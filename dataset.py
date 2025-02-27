import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
class CustData(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        
        image = Image.open(self.image_paths[idx]).convert("RGB") 
        lab = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, lab
transformer = transforms.Compose([
    transforms.Resize((144, 144)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
  

train_images, val_images = images_t[:split_idx], images_t[split_idx:]
train_labels, val_labels = labels_fin[:split_idx], labels_fin[split_idx:]
print(train_images, train_labels)
train_dataset = CustData(train_images, train_labels, transform=transformer)
val_dataset = CustData(val_images, val_labels, transform=transformer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,num_workers=4)

data_iter = iter(train_loader)
images, lab = next(data_iter)