import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from import_data import test_dataset
from cnn import UrbanSoundCNN as CNN

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = CNN().to(device)
model.load_state_dict(torch.load("models/cnn.pth"))
model.eval()
test_loss = 0.0
test_acc = 0.0
criterion = torch.nn.CrossEntropyLoss()

for i, (mel, label) in enumerate(test_dataloader):
    mel = mel.to(device)
    label = label.to(device)
    with torch.no_grad():
        output = model(mel)
        loss = criterion(output, label)
        test_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        test_acc += (predicted == label).sum().item()
test_loss /= len(test_dataloader)
test_acc /= len(test_dataset)
print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")