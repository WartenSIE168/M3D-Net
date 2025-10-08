import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from config import data_transforms, sequence_length

class TemporalDrivingDataset(Dataset):
    def __init__(self, img_files, eyeTxt_files, flightTxt_files, phase='train', seq_length=sequence_length):
        self.samples = []
        self.phase = phase
        self.seq_length = seq_length

        assert seq_length == 90, "Sequence length must be 90"
        assert len(img_files) == len(eyeTxt_files) == len(flightTxt_files), \
            "Image and text annotation files count mismatch"

        for img_file, eyeText_file, flightTxt_file in zip(img_files, eyeTxt_files, flightTxt_files):
            with open(img_file, 'r') as f:
                img_lines = [line.strip() for line in f.readlines() if line.strip()]
                assert len(img_lines) % seq_length == 0, \
                    f"Image file {img_file} line count must be divisible by {seq_length}"

            with open(eyeText_file, 'r') as f:
                eye_text_lines = [line.strip() for line in f.readlines() if line.strip()]
                assert len(eye_text_lines) >= len(img_lines), \
                    f"Eye text file {eyeTxt_files} has fewer lines than image file"

                eye_text_samples = []
                for i in range(0, len(img_lines), seq_length):
                    text_chunk = eye_text_lines[i:i+seq_length]
                    text_features = []
                    for line in text_chunk:
                        parts = line.split(',')
                        if len(parts) >= 4:
                            try:
                                features = [float(x) for x in parts[1:4]]
                                text_features.append(features)
                            except ValueError:
                                print(f"Invalid text features in {eyeText_file}: {line}")
                                break
                    if len(text_features) == seq_length:
                        eye_text_samples.append(np.array(text_features, dtype=np.float32))
                    else:
                        print(f"Warning: Invalid eye text sample at line {i} in {eyeText_file}")

            with open(flightTxt_file, 'r') as f:
                flight_text_lines = [line.strip() for line in f.readlines() if line.strip()]
                assert len(flight_text_lines) >= len(img_lines), \
                    f"Flight text file {flightTxt_files} has fewer lines than image file"

                flight_text_samples = []
                for i in range(0, len(img_lines), seq_length):
                    text_chunk = flight_text_lines[i:i+seq_length]
                    text_features = []
                    for line in text_chunk:
                        parts = line.split(',')
                        if len(parts) >= 4:
                            try:
                                features = [float(x) for x in parts[1:4]]
                                text_features.append(features)
                            except ValueError:
                                print(f"Invalid text features in {flightTxt_file}: {line}")
                                break
                    if len(text_features) == seq_length:
                        flight_text_samples.append(np.array(text_features, dtype=np.float32))
                    else:
                        print(f"Warning: Invalid flight text sample at line {i} in {flightTxt_file}")

            for i in range(0, len(img_lines), seq_length):
                img_chunk = img_lines[i:i+seq_length]
                frame_paths = []
                label = None
                for j, line in enumerate(img_chunk):
                    parts = line.split(',')
                    if len(parts) >= 3:
                        frame_path = parts[1].strip()
                        frame_paths.append(frame_path)
                        if j == seq_length - 1:
                            label = int(parts[2].strip())

                sample_idx = i // seq_length
                if sample_idx < len(eye_text_samples) and sample_idx < len(flight_text_samples):
                    eye_text_features = eye_text_samples[sample_idx]
                    flight_text_features = flight_text_samples[sample_idx]

                    if (len(frame_paths) == seq_length and 
                        len(eye_text_features) == seq_length and 
                        len(flight_text_features) == seq_length and 
                        label is not None):
                        self.samples.append((frame_paths, eye_text_features, flight_text_features, label))
                    else:
                        print(f"Warning: Invalid aligned sample at line {i} in {img_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame, eyeText, flightText, label = self.samples[idx]
        frames = []
        for path in frame:
            try:
                img = Image.open(path).convert('RGB')
                img = data_transforms[self.phase](img)
                frames.append(img)
            except Exception as e:
                print(f"Error loading {path}: {str(e)}")
                return None
        eyeText_tensors = [torch.tensor(item, dtype=torch.float32) for item in eyeText]
        flightText_tensors = [torch.tensor(item, dtype=torch.float32) for item in flightText]
        return torch.stack(frames), torch.stack(eyeText_tensors), torch.stack(flightText_tensors), torch.tensor(label)

def get_txt_files(directory=None):
    if directory is None:
        directory = os.getcwd()

    txt_files = []
    name_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                full_path = os.path.join(root, file)
                txt_files.append(full_path)
                name_files.append(file)

    return txt_files, name_files

def split_dataset(dataset, ratios, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    total = len(dataset)
    indices = list(range(total))
    np.random.shuffle(indices)

    train_end = int(ratios[0] * total)
    val_end = train_end + int(ratios[1] * total)

    return [
        torch.utils.data.Subset(dataset, indices[:train_end]),
        torch.utils.data.Subset(dataset, indices[train_end:val_end]),
        torch.utils.data.Subset(dataset, indices[val_end:])
    ]