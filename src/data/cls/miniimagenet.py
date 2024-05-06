from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from data.dataset_utils import DatasetEnum

import os
import torch

PATH = "/data2/jjlee_datasets/meta_datasets/XB_MAML_datasets/"
PATH = "/data/bjk_data/meta_datasets/"


class MiniImageNetDataset(Dataset):
    def __init__(self, split, contrastive=False):

        self.name = DatasetEnum.miniimagenet.name
        self.ds_name = PATH + DatasetEnum.miniimagenet.name
        self.split = split

        csv_path = os.path.join(self.ds_name, "split", "{}.csv".format(split))
        images_path = os.path.join(self.ds_name, "images")

        lines = [x.strip() for x in open(csv_path).readlines()[1:]]

        self.samples = []
        self.labels = []
        label_dict = {}
        label_index = 0
        self.label_index_2_name = []
        for e in lines:
            image_name, label_name = e.split(",")
            if label_name not in label_dict:
                label_dict[label_name] = label_index
                label_index += 1
                self.label_index_2_name.append(label_name)

            self.samples.append(os.path.join(images_path, image_name))
            self.labels.append(label_dict[label_name])

        is_train = split == "train"
        mean, std = [0.4721, 0.4533, 0.4099], [0.2771, 0.2677, 0.2844]
        resize_image = 84
        normalize_transform = transforms.Normalize(mean=mean, std=std)

        # transforms follows: https://github.com/kjunelee/MetaOptNet/blob/master/data/mini_imagenet.py
        self.transform = None
        if is_train:
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
                        # normalize_transform,
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
                        # normalize_transform,
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
                        # normalize_transform,
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
                        # normalize_transform,
                        # transforms.Resize(resize_image),
                        # transforms.CenterCrop(resize_image),
                        # transforms.ToTensor(),
                        # normalize_transform,
                    ]
                )
        print("loaded dataset: {}".format(self.__str__()))

    def __getitem__(self, index):
        image_path, label = self.samples[index], self.labels[index]
        image = self.transform(Image.open(image_path).convert("RGB"))

        return image, label

    def __len__(self):
        return len(self.labels)

    def get_label_num(self):
        return len(set(self.labels))

    def __str__(self):
        return "{}\t{}\t#samples: {}\t#classes: {}".format(
            self.name, self.split, len(self.samples), len(set(self.labels))
        )
