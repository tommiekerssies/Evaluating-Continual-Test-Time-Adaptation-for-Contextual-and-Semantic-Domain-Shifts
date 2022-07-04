from torchvision import transforms
import torch
import torch.nn as nn
from torchvision.models import resnet50
import os
from PIL import Image
from torch.utils.data import Dataset

num_classes = 126
bottleneck_dim = 256

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]
)


class Model(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.to(device)
        model = resnet50(pretrained=True).to(device)
        model.fc = nn.Linear(model.fc.in_features, bottleneck_dim)
        bn = nn.BatchNorm1d(bottleneck_dim)
        self.encoder = nn.Sequential(model, bn)
        self.fc = nn.Linear(bottleneck_dim, num_classes)
        self.fc = nn.utils.weight_norm(self.fc, dim=0)

    def forward(self, x):
        return self.fc(torch.flatten(self.encoder(x), 1))

    def load_state_dict(self, checkpoint):
        state_dict = {}
        for name, param in checkpoint["state_dict"].items():
            # get rid of 'module.' prefix brought by DDP
            name = name.replace("module.", "")
            state_dict[name] = param
        super().load_state_dict(state_dict)


class Dataset(Dataset):
    def __init__(
        self,
        root: str,
        domains,
        transform=None,
    ):
        self.transform = transform

        lines = []
        for domain in domains:
          with open(os.path.join(root, f"{domain}_list.txt"), "r") as fd:
              lines += fd.readlines()
        lines = [line.strip() for line in lines if line]

        self.item_list = []
        for item in lines:
            img_file, label = item.split()
            img_path = os.path.join(root, img_file)
            label = int(label)
            self.item_list.append((img_path, label, img_file))

    def __getitem__(self, idx):
        """Retrieve data for one item.
        Args:
            idx: index of the dataset item.
        Returns:
            img: <C, H, W> tensor of an image
            label: int or <C, > tensor, the corresponding class label. when using raw label
                file return int, when using pseudo label list return <C, > tensor.
        """
        img_path, label, _ = self.item_list[idx]
        img = Image.open(img_path)
        img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        
        return img, label

    def __len__(self):
        return len(self.item_list)