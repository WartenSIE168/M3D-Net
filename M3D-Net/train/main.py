import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from config import device, split_ratios, random_seed, class_names
from dataset import TemporalDrivingDataset, get_txt_files, split_dataset
from model import StableCNNLSTM
from trainer import train_epoch, validate_epoch
from utils import configure_optimizer, find_matching_indices

def main():
    img_files, _ = get_txt_files('..\\processData\\dealImgData')
    eyeTxt_files, _ = get_txt_files('..\\processData\\dealEyeData')
    flightTxt_files, _ = get_txt_files('..\\processData\\dealFlightData')

    full_dataset = TemporalDrivingDataset(
        img_files=img_files,
        eyeTxt_files=eyeTxt_files,
        flightTxt_files=flightTxt_files,
        phase='train',
        seq_length=90
    )

    train_subset, val_subset, test_subset = split_dataset(
        full_dataset, split_ratios, seed=random_seed
    )

    print(f"Training samples: {len(train_subset)}, Validation samples: {len(val_subset)}, Test samples: {len(test_subset)}")

    dataloader_train = DataLoader(train_subset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    dataloader_val = DataLoader(val_subset, batch_size=16, shuffle=False, num_workers=4)
    dataloader_test = DataLoader(test_subset, batch_size=16, shuffle=False, num_workers=4)

    if torch.cuda.device_count() >= 2:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = StableCNNLSTM()
        model = nn.DataParallel(model)
        device = torch.device('cuda')
    else:
        print("Not enough GPUs available, using CPU or single GPU for training")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = StableCNNLSTM()

    model = model.to(device)

    optimizer, scheduler = configure_optimizer(model, max_epochs=20, steps_per_epoch=len(dataloader_train))
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    epochs = 50
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    train_name = 'MobNet_BiLSTM_ConvLSTMpro'

    for epoch in range(epochs):
        print(f"\nEpoch [{epoch + 1}/{epochs}]")
        train_loss, train_acc = train_epoch(model, dataloader_train, optimizer, criterion, device)
        val_loss, val_acc = validate_epoch(model, dataloader_val, criterion, device)

        scheduler.step(val_acc)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f" Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'.\\model\\best_{train_name}_model.pth')
            print("Saved new best model!")

        torch.cuda.empty_cache()

    epochs_range = range(1, len(train_accuracies) + 1)
    plt.figure(figsize=(12, 5))

    total_epochs = len(train_accuracies)
    if total_epochs > 20:
        tick_step = max(2, total_epochs // 10)
        tick_positions = list(range(1, total_epochs + 1, tick_step))
        if total_epochs not in tick_positions:
            tick_positions.append(total_epochs)
    else:
        tick_positions = epochs_range

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.xticks(tick_positions)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Acc')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Val Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.xticks(tick_positions)
    plt.tight_layout()
    plt.savefig(f'.\\train_img\\{train_name}_training_metrics.png')

    model.load_state_dict(torch.load(f'.\\model\\best_{train_name}_model.pth'))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader_test, desc='Testing', leave=True)
        for batch in progress_bar:
            if len(batch) == 4:
                img_inputs, eye_txt_inputs, flight_txt_inputs, labels = batch
                img_inputs = img_inputs.to(device).float()
                eye_txt_inputs = eye_txt_inputs.to(device).float()
                flight_txt_inputs = flight_txt_inputs.to(device).float()
                outputs = model(img_inputs, eye_txt_inputs, flight_txt_inputs)
            else:
                pass
            labels = labels.to(device)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\nTest Results:")
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Overall Accuracy: {accuracy:.4f}")
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)

    report_path = f'.\\class_report\\{train_name}_classification_report.txt'
    with open(report_path, 'w') as f:
        f.write("Test Results:\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n")
        f.write(report)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(f'.\\train_img\\{train_name}_confusion_matrix.png')

if __name__ == "__main__":
    main()