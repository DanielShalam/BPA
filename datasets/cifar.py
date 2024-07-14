import os.path as osp
from PIL import Image
import os
from torch.utils.data import Dataset
from torchvision import transforms


class CIFAR(Dataset):

    def __init__(self, data_path: str, setname: str, backbone: str, augment: bool):
        d = osp.join(data_path, setname)
        dirs = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d, o))]

        data = []
        label = []
        lb = -1

        for d in dirs:
            lb += 1
            for image_name in os.listdir(d):
                path = osp.join(d, image_name)
                data.append(path)
                label.append(lb)

        self.data = data
        self.label = label

        mean = [x / 255.0 for x in [129.37731888, 124.10583864, 112.47758569]]
        std = [x / 255.0 for x in [68.20947949, 65.43124043, 70.45866994]]
        normalize = transforms.Normalize(mean=mean, std=std)

        self.image_size = 32
        if augment and setname == 'train':
            transforms_list = [
                transforms.RandomResizedCrop(self.image_size),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        else:
            transforms_list = [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]

        self.transform = transforms.Compose(
            transforms_list + [normalize]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label


def get_transform(img_size: int, split_name: str):
    mean = [x / 255.0 for x in [129.37731888, 124.10583864, 112.47758569]]
    std = [x / 255.0 for x in [68.20947949, 65.43124043, 70.45866994]]
    normalize = transforms.Normalize(mean=mean, std=std)

    if split_name == 'train':
        return transforms.Compose([
            # transforms.RandomResizedCrop((img_size, img_size), scale=(0.05, 1.0)),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize
        ])
