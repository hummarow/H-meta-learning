import os
import torch

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from data.dataset_utils import load_data, DatasetEnum

PATH = "/data2/jjlee_datasets/meta_datasets/XB_MAML_datasets/"
PATH = "/data/bjk_data/meta_datasets/"


"""
from: https://github.com/kjunelee/MetaOptNet/ 
"""


class CifarFS_dataset(Dataset):
    def __init__(self, split, contrastive=False):
        self.split = split
        self.is_train = split == "train"

        # load data
        self.ds_name = DatasetEnum.CIFAR_FS.name
        data_path = os.path.join(PATH + self.ds_name, "{}.pickle".format(self.split))
        data_all = load_data(data_path)
        data_label = data_all["labels"]
        label_set = set(data_label)
        label_dict = dict(zip(label_set, range(len(label_set))))
        self.samples = data_all["data"]
        self.labels = [label_dict[x] for x in data_label]

        # build transformers
        self.transform = None
        mean = [0.5074, 0.4867, 0.4411]
        std = [0.2675, 0.2566, 0.2763]
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
        print("loaded dataset: {}".format(self.__str__()))

    def __getitem__(self, index):
        img, label = self.samples[index], self.labels[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, 0

    def __len__(self):
        return len(self.labels)

    def get_label_num(self):
        return len(set(self.labels))

    def __str__(self):
        return "{}\t{}\t#samples: {}\t#classes: {}".format(
            self.ds_name, self.split, len(self.samples), len(set(self.labels))
        )
