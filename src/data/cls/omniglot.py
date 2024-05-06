import os
import torch

from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from data.dataset_utils import DatasetEnum
from data.dataset_utils import compute_mean_std

# from utils.path_utils import PathUtils


default_stat = ([0.5074, 0.4867, 0.4411], [0.2675, 0.2566, 0.2763])
PATH = "/data2/jjlee_datasets/meta_datasets/XB_MAML_datasets/"
PATH = "/data/bjk_data/meta_datasets/"


class Omniglot(Dataset):
    """
    refer to https://github.com/google-research/meta-dataset
    """

    def __init__(self, split, ds_name=DatasetEnum.omniglot_py.name, contrastive=False):
        self.split = split
        self.is_train = split == "train"

        self.labels = None
        self.samples = None

        self.ds_name = ds_name
        data_root = os.path.join(PATH, DatasetEnum.get_value_by_name(ds_name=ds_name))
        data_path = os.path.join(data_root, "{}".format(self.split))

        # build transformers
        self.transform = None
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
                        transforms.Resize([resize_image, resize_image]),
                        transforms.ToTensor(),
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
                        transforms.Resize([resize_image, resize_image]),
                        transforms.ToTensor(),
                    ]
                )

        self.m_dataset = ImageFolder(data_path, self.transform)

        self.samples = list(self.m_dataset.samples)
        self.labels = list(self.m_dataset.targets)

        print("loaded dataset: {}".format(self.__str__()))

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
