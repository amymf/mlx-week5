from import_data import train_dataset, val_dataset
from cnn import UrbanSoundCNN as CNN
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

wandb.init(project="UrbanSound8K")

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    for i, (mel, label) in enumerate(tqdm(train_dataloader)):
        mel = mel.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        output = model(mel)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        train_acc += (predicted == label).sum().item()
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataset)

    val_loss = 0.0
    val_acc = 0.0
    model.eval()
    for i, (mel, label) in enumerate(tqdm(val_dataloader)):
        mel = mel.to(device)
        label = label.to(device)
        with torch.no_grad():
            output = model(mel)
            loss = criterion(output, label)
            val_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            val_acc += (predicted == label).sum().item()
    val_loss /= len(val_dataloader)
    val_acc /= len(val_dataset)
        
    wandb.log({"train_loss": train_loss, "train_acc": train_acc})
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    wandb.log({"val_loss": val_loss, "val_acc": val_acc})
    print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

torch.save(model.state_dict(), "models/cnn.pth")
wandb.save("cnn.pth")
print("Model saved as models/cnn.pth")