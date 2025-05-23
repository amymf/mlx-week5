import torch
import torch.nn.functional as F
import librosa
import numpy as np
from torch.utils.data import Dataset
from skimage.transform import resize
import os


class UrbanSoundDataset(Dataset):
    def __init__(self, df, dataset_path, sr=22050, n_mels=128, duration=4.0):
        self.df = df
        self.dataset_path = dataset_path
        self.sr = sr
        self.n_mels = n_mels
        self.duration = duration
        self.samples = int(sr * duration)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(
            self.dataset_path, f"fold{row.fold}", row.slice_file_name
        )

        # Load audio file
        y, _ = librosa.load(file_path, sr=self.sr)
        # Pad or trim
        if len(y) < self.samples:
            y = np.pad(y, (0, self.samples - len(y)))
        else:
            y = y[: self.samples]

        # Compute mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_mels=self.n_mels
        )  # (128, num_frames)
        mel_db = librosa.power_to_db(mel)
        # Normalize, convert to tensor and resize
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
        mel_tensor = torch.tensor(mel_db).unsqueeze(0).float()  # (1, 128, num_frames)

        label = torch.tensor(row.classID).long()
        return mel_tensor, label
