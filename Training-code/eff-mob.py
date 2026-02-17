import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import time
import os
import copy
import zipfile
from google.colab import drive
from google.colab import files




print("Mounting Google Drive...")
drive.mount('/content/drive')


ZIP_PATH = '/content/drive/MyDrive/plants.zip' 
EXTRACT_PATH = '/content/dataset'
BATCH_SIZE = 32
NUM_EPOCHS = 10  
INPUT_SIZE = 224 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")


if not os.path.exists(EXTRACT_PATH):
    print(f"Unzipping {ZIP_PATH}...")
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_PATH)
        print("Unzipping complete!")
    except FileNotFoundError:
        print(f"ERROR: Could not find {ZIP_PATH}")
        print("Please make sure 'plants.zip' is uploaded to your Google Drive.")
        raise
else:
    print("Dataset already unzipped. Skipping.")


DATA_DIR = '/content/dataset/dataset' 

print(f"Dataset location: {DATA_DIR}")


if not os.path.exists(DATA_DIR):
    print("❌ ERROR: The path isn't found. Please run the 'Fix-It' script again.")
else:
    print("✅ Path found! Loading images...")


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Loading images...")
full_dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms['train'])
class_names = full_dataset.classes
print(f"Classes found: {class_names}")


train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True),
    'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
}
dataset_sizes = {'train': train_size, 'val': val_size}



def get_model(model_name, num_classes):
    if model_name == 'efficientnet':
        model = models.efficientnet_b0(weights='DEFAULT')
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'mobilenet':
        model = models.mobilenet_v3_large(weights='DEFAULT')
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, num_classes)
    return model.to(DEVICE)

def train_model(model, criterion, optimizer, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model, best_acc, time_elapsed



print("\n>>>>>>>> STARTING EFFICIENTNET TRAINING <<<<<<<<")
model_eff = get_model('efficientnet', len(class_names))
criterion = nn.CrossEntropyLoss()
optimizer_eff = optim.Adam(model_eff.parameters(), lr=0.001)
best_model_eff, acc_eff, time_eff = train_model(model_eff, criterion, optimizer_eff, NUM_EPOCHS)

print("\n>>>>>>>> STARTING MOBILENET TRAINING <<<<<<<<")
model_mob = get_model('mobilenet', len(class_names))
optimizer_mob = optim.Adam(model_mob.parameters(), lr=0.001)
best_model_mob, acc_mob, time_mob = train_model(model_mob, criterion, optimizer_mob, NUM_EPOCHS)



print("\n==========================================")
print("             FINAL COMPARISON             ")
print("==========================================")
print(f"EfficientNet | Accuracy: {acc_eff*100:.2f}% | Time: {time_eff:.0f}s")
print(f"MobileNetV3  | Accuracy: {acc_mob*100:.2f}% | Time: {time_mob:.0f}s")
print("==========================================")


if acc_eff > acc_mob:
    winner = "EfficientNet"
    best_model = best_model_eff
    save_name = 'efficientnet_plants.pth'
else:
    winner = "MobileNet"
    best_model = best_model_mob
    save_name = 'mobilenet_plants.pth'

print(f"\nWinning Model: {winner}")
print(f"Saving {winner} model to {save_name}...")

torch.save(best_model.state_dict(), save_name)

print(f"Model saved! Downloading to your computer...")
files.download(save_name)

print("\nDone! Please copy these class names for your local test script:")
print(class_names)