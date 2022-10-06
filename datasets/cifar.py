import os.path as osp
from PIL import Image
import os
from torch.utils.data import Dataset
from torchvision import transforms


class CIFAR(Dataset):

    def __init__(self, setname):
        d = osp.join('data', 'cifar', setname)
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
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,
                                 std=std)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label, path

