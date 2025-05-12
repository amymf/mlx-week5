import os
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset import UrbanSoundDataset


DATASET_PATH = "data"
metadata_path = os.path.join(DATASET_PATH, "metadata/UrbanSound8K.csv")
audio_path = os.path.join(DATASET_PATH, "audio")

metadata = pd.read_csv(metadata_path)
metadata = metadata.sample(frac=1, random_state=42).reset_index(drop=True)

train_df, temp_test_df = train_test_split(metadata, test_size=0.2, random_state=42)
test_df, val_df = train_test_split(temp_test_df, test_size=0.5, random_state=42)

train_dataset = UrbanSoundDataset(train_df, audio_path)
test_dataset = UrbanSoundDataset(test_df, audio_path)
val_dataset = UrbanSoundDataset(val_df, audio_path)
