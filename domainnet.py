from copy import deepcopy
from torchvision import transforms
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import os
from PIL import Image
from torch.utils.data import Dataset

num_classes = 126
bottleneck_dim = 256
img_size = 224
train_ratio = 0.9

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ]
)


class Model(nn.Module):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.to(device)
        model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
        model.fc = nn.Linear(model.fc.in_features, bottleneck_dim)
        bn = nn.BatchNorm1d(bottleneck_dim)
        self.encoder = nn.Sequential(model, bn)
        self.fc = nn.Linear(bottleneck_dim, num_classes)
        self.fc = nn.utils.weight_norm(self.fc, dim=0)

    def forward(self, x):
        return self.fc(torch.flatten(self.encoder(x), 1))

    def load_state_dict(self, checkpoint, *args, **kwargs):
        if "state_dict" in checkpoint:
          checkpoint = checkpoint["state_dict"]
        state_dict = {}
        for name, param in checkpoint.items():
            # get rid of 'module.' prefix brought by DDP
            name = name.replace("module.", "")
            state_dict[name] = param
        super().load_state_dict(state_dict, *args, **kwargs)

    def __deepcopy__(self, memo):
      """deepcopy on weightnorm gives an error, so fix it by detaching before and reattaching after"""
      self.fc.weight = torch._weight_norm(self.fc.weight_v, self.fc.weight_g, dim=0).detach()

      deepcopy_method = self.__deepcopy__
      self.__deepcopy__ = None
      cp = deepcopy(self, memo)

      self.fc.weight = torch._weight_norm(self.fc.weight_v, self.fc.weight_g, dim=0) #reattach
      cp.fc.weight = torch._weight_norm(cp.fc.weight_v, cp.fc.weight_g, dim=0) #reattach

      self.__deepcopy__ = deepcopy_method
      cp.__deepcopy__ = deepcopy_method

      return cp

class Dataset(Dataset):
    def __init__(
        self,
        root: str,
        domains,
        transform,
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