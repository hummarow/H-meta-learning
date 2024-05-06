import os

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from data.dataset_utils import DatasetEnum
from SimCLR.data_aug.gaussian_blur import GaussianBlur


# ["ilsvrc", "omniglot_py", "AIRCRAFT", "BIRD", "TEXTURE", "QuickDraw", "FUNGI", "VGG", "Flower", "Traffic signs", "MSCOCO"]

SPLIT_MAPPING = {"valid": "val", "train": "train", "test": "test"}

DS_STAT_DICT = {
    DatasetEnum.miniimagenet.name: ([0.4721, 0.4533, 0.4099], [0.2771, 0.2677, 0.2844]),
    DatasetEnum.BIRD_CDFSL.name: ([0.4184, 0.4179, 0.3608], [0.2695, 0.2661, 0.2724]),
    DatasetEnum.CARS.name: ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    DatasetEnum.PLACES.name: ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    DatasetEnum.PLANTAE.name: ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}

default_stat = ([0.5074, 0.4867, 0.4411], [0.2675, 0.2566, 0.2763])

PATH = "/data/bjk_data/"

"""
from: https://github.com/huaxiuyao/HSML/
"""


class CDFSL(Dataset):

    def __init__(self, split, ds_name, contrastive):
        self.split = SPLIT_MAPPING[split]
        self.is_train = split == "train"

        self.labels = None
        self.samples = None

        self.ds_name = ds_name
        # def get_dataset_path(ds_name):
        data_root = os.path.join(PATH, DatasetEnum.get_value_by_name(ds_name=ds_name))
        data_path = os.path.join(data_root, "{}".format(self.split))

        # build transformers
        self.transform = None
        mean, std = DS_STAT_DICT.get(ds_name, default_stat)
        normalize_transform = transforms.Normalize(mean=mean, std=std)

        resize_image = 84
        if self.is_train:
            if contrastive:
                contrastive_transform = transforms.Compose(
                    [
                        transforms.Resize((resize_image, resize_image)),
                        transforms.RandomResizedCrop(size=resize_image),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply(
                            [transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8
                        ),
                        transforms.GaussianBlur(
                            kernel_size=int(0.1 * resize_image) // 2 * 2 + 1
                        ),
                        transforms.ToTensor(),
                        normalize_transform,
                    ]
                )
                self.transform = transforms.Lambda(
                    lambda x: torch.stack(
                        [contrastive_transform(x), contrastive_transform(x)], dim=0
                    )
                )
            else:
                self.transform = transforms.Compose(
                    [
                        transforms.Resize((resize_image, resize_image)),
                        transforms.ToTensor(),
                        normalize_transform,
                    ]
                )
        else:
            if contrastive:
                contrastive_transform = transforms.Compose(
                    [
                        transforms.Resize((resize_image, resize_image)),
                        transforms.RandomResizedCrop(size=resize_image),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply(
                            [transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8
                        ),
                        transforms.GaussianBlur(
                            kernel_size=int(0.1 * resize_image) // 2 * 2 + 1
                        ),
                        transforms.ToTensor(),
                        normalize_transform,
                    ]
                )
                self.transform = transforms.Lambda(
                    lambda x: torch.stack(
                        [contrastive_transform(x), contrastive_transform(x)], dim=0
                    )
                )
            else:
                self.transform = transforms.Compose(
                    [
                        transforms.Resize((resize_image, resize_image)),
                        transforms.ToTensor(),
                        normalize_transform,
                    ]
                )

        self.m_dataset = ImageFolder(data_path, self.transform)

        self.samples = list(self.m_dataset.samples)
        self.labels = list(self.m_dataset.targets)

        # print("loaded dataset: {}".format(self.__str__()))

    def __getitem__(self, index):
        img, label = self.m_dataset[index]
        return img, label

    def __len__(self):
        return len(self.labels)

    def get_label_num(self):
        return len(set(self.labels))

    def __str__(self):
        return "{}\t{}\t#samples: {}\t#classes: {}".format(
            self.ds_name, self.split, len(self.samples), len(set(self.labels))
        )
