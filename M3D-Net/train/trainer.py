import torch
from tqdm import tqdm
from config import device
from model import StableCNNLSTM

def train_epoch(model, dataloader, optimizer, criterion, device=device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc='Training', leave=False)
    for batch in progress_bar:
        if len(batch) == 4:
            img_inputs, eye_txt_inputs, flight_txt_inputs, labels = batch
        else:
            img_inputs = batch[0]
            eye_txt_inputs = batch[1]
            flight_txt_inputs = batch[2]
            labels = batch[3]

        img_inputs = img_inputs.to(device).float()
        eye_txt_inputs = eye_txt_inputs.to(device).float()
        flight_txt_inputs = flight_txt_inputs.to(device).float()
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(img_inputs, eye_txt_inputs, flight_txt_inputs)

        loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        progress_bar.set_postfix({
            'Loss': loss.item(),
            'Acc': f'{correct/total:.2%}'
        })

    avg_loss = total_loss / total if total > 0 else 0
    avg_acc = correct / total if total > 0 else 0
    return avg_loss, avg_acc

def validate_epoch(model, dataloader, criterion, device=device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Validating', leave=False)
        for img_inputs, eye_txt_inputs, flight_txt_inputs, labels in progress_bar:
            img_inputs = img_inputs.to(device).float()
            eye_txt_inputs = eye_txt_inputs.to(device).float()
            flight_txt_inputs = flight_txt_inputs.to(device).float()
            labels = labels.to(device)

            outputs = model(img_inputs, eye_txt_inputs, flight_txt_inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix({
                'Val Loss': loss.item(),
                'Val Acc': f'{correct/total:.2%}'
            })

    avg_loss = total_loss / total if total > 0 else 0
    avg_acc = correct / total if total > 0 else 0
    return avg_loss, avg_acc