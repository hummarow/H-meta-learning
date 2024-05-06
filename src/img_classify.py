import numpy
import torch

from omniglot import Omniglot
import torchvision.transforms as transforms
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
import os.path
import numpy as np
import random
from collections import defaultdict

from SimCLR.data_aug.view_generator import ContrastiveLearningViewGenerator
from SimCLR.data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from SimCLR.data_aug.gaussian_blur import GaussianBlur


# 参考正确学长版
# from omniglotNShot_new import OmniglotNShot


class ImgClassification:
    def __init__(self, args):
        """
        Different from mnistNShot, the
        :param root:
        :param batchsz: task num
        :param n_way:
        :param k_shot:
        :param k_qry:
        :param imgsz:
        """

        print("进入数据集加载中")

        # self.resize = args.imgsz
        # print(f"self.resize: {self.resize}")

        if args.dataset == "omniglot":
            self.resize = 28
        elif args.dataset == "imagenet-1k" or args.dataset == "miniimagenet":
            self.resize = 84
        elif args.dataset == "domainnet":
            self.resize = 224

        # print(f"args.dataset: {args.dataset}")
        # print(f"self.resize: {self.resize}")
        # exit()

        self.dataset = args.dataset
        self.channel = args.channel
        self.n_views = 3
        self.contrastive = args.contrastive
        if args.dataset == "omniglot":
            self.x = self.create_Omniglot(args.root)
        elif args.dataset == "miniimagenet":
            self.x = self.create_mini_imagenet(args.root)
        elif args.dataset == "imagenet-1k":
            self.x = self.create_imagenet_1k(args.root)
        elif args.dataset == "domainnet":
            self.x = self.create_domainnet(args.root)
        self.x = self.create_domainnet(args.root)
        # exit()

        # [1623, 20, 84, 84, 1]
        # TODO: can not shuffle here, we must keep training and test set distinct!

        # 当为 domainnet，(5, 304, 20, 3, 224, 224)

        # 对类别的分类，
        self.train_split = args.train_split
        self.val_split = args.val_split

        # self.x_train, self.x_test = self.x[:1200], self.x[1200:]

        if args.dataset == "domainnet":
            # pass
            # self.x_train, self.x_test = (
            #     self.x[:, : self.train_split],
            #     self.x[:, self.train_split :],
            # )
            self.x_train, self.x_test = (
                self.x[: self.train_split],
                self.x[self.train_split :],
            )
            self.n_cls = self.x.shape[1]
        else:
            self.x_train, self.x_test = (
                self.x[: self.train_split],
                self.x[self.train_split :],
            )
            self.n_cls = self.x.shape[0]

        # print(f"self.x_train: {self.x_train.shape}  self.x_test: {self.x_test.shape} ")

        # exit()
        # self.x_train, self.x_test = self.x[: self.train_split], self.x[self.train_split:]

        # self.normalization()
        # self.n_cls = self.x.shape[0]  # 1623

        self.n_way = args.n_way  # n way
        self.task_cluster_batch_num = args.task_cluster_batch_num
        self.task_num = args.task_num
        self.task_spt = args.task_spt  # 此参数没用了
        self.task_qry = args.task_qry
        self.k_shot = args.k_spt  # k shot
        self.k_query = args.k_qry  # k query
        self.if_val = args.if_val
        self.test_spt_task_from = args.test_spt_task_from
        self.candidate_num = args.candidate_num
        assert (
            self.k_shot + self.k_query
        ) <= 20  # 为什么呢？ 每个类别下的图片数量为 20

        # save pointer of current read batch in total cache
        self.indexes = {"train": 0, "test": 0}
        self.datasets = {
            "train": self.x_train,
            "test": self.x_test,
        }  # original data cached
        print("DB: train", self.x_train.shape, "test", self.x_test.shape)

        # 当为 domainnet 时， self.x_train: (5, 270, 20, 3, 224, 224)
        #                    self.x_test:  (5, 34, 20, 3, 224, 224)

        self.datasets_cache = {"train": [], "test": []}

    def normalization(self):
        """
        Normalizes our data, to have a mean of 0 and sdt of 1
        """
        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)
        # print("before norm:", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)
        self.x_train = (self.x_train - self.mean) / self.std
        self.x_test = (self.x_test - self.mean) / self.std

        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)

    def construct_task(self, data_pack, task_num):
        """
        :param data_pack: [图片类别数量，对应类别下的图片数量，图片（占三位）]，numpy.array 类型
        :param task_num: 任务数量
        :return: x_spt_task_cluster, x_qry_task_cluster, y_spt_task_cluster, y_qry_task_cluster
                x_spt_task_cluster 和 x_qry_task_cluster 为 [任务数量，图片数量，图片（占三位）] 图片数量为 n_way * k_shot 或 n_way * k_qry

        """

        (
            x_spt_task_cluster,
            x_qry_task_cluster,
            y_spt_task_cluster,
            y_qry_task_cluster,
        ) = ([], [], [], [])
        for t in range(task_num):
            selected_cls = np.random.choice(data_pack.shape[0], self.n_way, False)
            # try:
            #     selected_cls = np.random.choice(data_pack.shape[0], self.n_way, False)
            # except:
            #     breakpoint()
            x_spt, y_spt, x_qry, y_qry = [], [], [], []
            for j, cur_class in enumerate(selected_cls):
                selected_img = np.random.choice(20, self.k_shot + self.k_query, False)
                # meta-training and meta-test
                x_spt.append(data_pack[cur_class][selected_img[: self.k_shot]])
                x_qry.append(data_pack[cur_class][selected_img[self.k_shot :]])
                y_spt.append([j for _ in range(self.k_shot)])
                y_qry.append([j for _ in range(self.k_query)])

            # shuffle inside a batch
            perm = np.random.permutation(self.n_way * self.k_shot)

            # 这里针对数据集的通道数进行区分

            # x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot, 1, self.resize, self.resize)[perm]
            # print(f"self.resize: {self.resize}")
            if self.dataset == "omniglot":
                if not self.contrastive:
                    x_spt = np.array(x_spt).reshape(
                        self.n_way * self.k_shot, 1, self.resize, self.resize
                    )[perm]
                # NOTE: Modification for contrastive learning.
                else:
                    x_spt = np.array(x_spt).reshape(
                        self.n_way * self.k_shot,
                        self.n_views,
                        1,
                        self.resize,
                        self.resize,
                    )[perm]
            else:
                if not self.contrastive:
                    x_spt = np.array(x_spt).reshape(
                        self.n_way * self.k_shot, 3, self.resize, self.resize
                    )[perm]
                else:
                    x_spt = np.array(x_spt).reshape(
                        self.n_way * self.k_shot,
                        self.n_views,
                        3,
                        self.resize,
                        self.resize,
                    )[perm]

            y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)[perm]
            perm = np.random.permutation(self.n_way * self.k_query)

            # x_qry = np.array(x_qry).reshape(self.n_way * self.k_query, 1, self.resize, self.resize)[perm]
            if self.dataset == "omniglot":
                if not self.contrastive:
                    x_qry = np.array(x_qry).reshape(
                        self.n_way * self.k_query, 1, self.resize, self.resize
                    )[perm]
                else:
                    x_qry = np.array(x_qry).reshape(
                        self.n_way * self.k_query,
                        self.n_views,
                        1,
                        self.resize,
                        self.resize,
                    )[perm]
            else:
                if not self.contrastive:
                    x_qry = np.array(x_qry).reshape(
                        self.n_way * self.k_query, 3, self.resize, self.resize
                    )[perm]
                else:
                    x_qry = np.array(x_qry).reshape(
                        self.n_way * self.k_query,
                        self.n_views,
                        3,
                        self.resize,
                        self.resize,
                    )[perm]

            y_qry = np.array(y_qry).reshape(self.n_way * self.k_query)[perm]

            x_spt_task_cluster.append(x_spt)
            x_qry_task_cluster.append(x_qry)
            y_spt_task_cluster.append(y_spt)
            y_qry_task_cluster.append(y_qry)

        return (
            x_spt_task_cluster,
            x_qry_task_cluster,
            y_spt_task_cluster,
            y_qry_task_cluster,
        )

    def unpack_hie_data(self, eight_cluster_batch, train_or_test):
        eight_cluster_mass = [0, 0, 0, 0, 0, 0, 0, 0]

        eight_cluster_mass[:4] = [
            one_cluster_batch.reshape(
                (-1, one_cluster_batch.shape[2], self.channel, self.resize, self.resize)
            )
            for one_cluster_batch in eight_cluster_batch[:4]
        ]
        eight_cluster_mass[4:] = [
            one_cluster_batch.reshape((-1, one_cluster_batch.shape[2]))
            for one_cluster_batch in eight_cluster_batch[4:]
        ]

        if train_or_test == "train":
            x_spt = np.vstack((eight_cluster_mass[0], eight_cluster_mass[2]))
            x_qry = np.vstack((eight_cluster_mass[1], eight_cluster_mass[3]))
            y_spt = np.vstack((eight_cluster_mass[4], eight_cluster_mass[6]))
            y_qry = np.vstack((eight_cluster_mass[5], eight_cluster_mass[7]))

        else:  # 测试用cluster中的spt_task不是目标对象，在当前方案下，直接舍弃
            x_spt = eight_cluster_mass[2]
            y_spt = eight_cluster_mass[6]
            x_qry = eight_cluster_mass[3]
            y_qry = eight_cluster_mass[7]

        return x_spt, x_qry, y_spt, y_qry

    def load_data_cache_hie(self, data_pack: numpy.ndarray, mode, sample_mode):
        """
        :param data_pack: [图片类别数量，对应类别下的图片数量，图片（占三位）]，numpy.array 类型
                    当为 domainnet 时，为 [domain数量，domain下的类别数量，一个类别下的图片数量，图片（3位）]
        :param mode:
        :param sample_mode:
        :return: x_spt_task_spt_set_batch, x_spt_task_qry_set_batch, x_qry_task_spt_set_batch, x_qry_task_qry_set_batch,
                y_spt_task_spt_set_batch, y_spt_task_qry_set_batch, y_qry_task_spt_set_batch, y_qry_task_qry_set_batch

                x_spt_task_spt_set_batch 为 [cluster数量，任务数量，图片数量，图片（占3位）]
        """

        """
        Collects several batches data for N-shot learning
        :param data_pack: [cls_num, 20, 84, 84, 1]
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        #  take 5 way 1 shot as example: 5 * 1
        data_cache = []

        for sample in range(1):  # 缓存为1，要不然显存受不了
            (
                x_spt_task_spt_set_batch,
                x_spt_task_qry_set_batch,
                x_qry_task_spt_set_batch,
                x_qry_task_qry_set_batch,
                y_spt_task_spt_set_batch,
                y_spt_task_qry_set_batch,
                y_qry_task_spt_set_batch,
                y_qry_task_qry_set_batch,
            ) = ([], [], [], [], [], [], [], [])

            # x = np.array([1])

            # domain 的数量大于等于 cluster 的数量，那么为 不可重复取样，否则为可重复取样
            if self.dataset == "domainnet":
                # breakpoint()
                if data_pack.shape[0] >= self.task_cluster_batch_num:
                    domain_selected = np.random.choice(
                        range(data_pack.shape[0]),
                        self.task_cluster_batch_num,
                        replace=False,
                    )
                else:
                    domain_selected = np.random.choice(
                        range(data_pack.shape[0]),
                        self.task_cluster_batch_num,
                        replace=True,
                    )
                for i in range(self.task_cluster_batch_num):
                    index_selected = domain_selected[i]

                    if mode == "train":
                        if bool(self.if_val):
                            # spt_pack = data_pack[:800]  # 如果是train,传入的data_pack则为前1200个类
                            # qry_pack = data_pack[800:]

                            spt_pack = data_pack[index_selected][
                                : self.val_split
                            ]  # 如果是train,传入的data_pack则为前1200个类
                            qry_pack = data_pack[index_selected][self.val_split :]

                        else:
                            spt_pack = qry_pack = data_pack[index_selected]
                    else:
                        if bool(self.if_val):
                            # spt_pack = data_pack[: 800]  # 如果是test，传入的则是训练集的前800个和测试集的400多个的拼接
                            # qry_pack = data_pack[800:]

                            spt_pack = data_pack[index_selected][
                                : self.val_split
                            ]  # 如果是test，传入的则是训练集的前800个和测试集的400多个的拼接
                            qry_pack = data_pack[index_selected][self.val_split :]

                        elif (
                            not bool(self.if_val)
                            and self.test_spt_task_from == "meta_train"
                        ):
                            # spt_pack = data_pack[:1200]
                            # qry_pack = data_pack[1200:]

                            spt_pack = data_pack[index_selected][: self.train_split]
                            qry_pack = data_pack[index_selected][self.train_split :]
                        else:
                            spt_pack = qry_pack = data_pack[index_selected]

                    if sample_mode == "hierarchical_concentrate":
                        sub_index = random.sample(
                            range(len(spt_pack)), self.candidate_num
                        )
                        spt_pack = spt_pack[sub_index]
                        sub_index = random.sample(
                            range(len(qry_pack)), self.candidate_num
                        )
                        qry_pack = qry_pack[sub_index]
                    else:
                        assert sample_mode == "hierarchical_uniform"

                    (
                        x_spt_task_spt_set_cluster,
                        x_spt_task_qry_set_cluster,
                        y_spt_task_spt_set_cluster,
                        y_spt_task_qry_set_cluster,
                    ) = self.construct_task(spt_pack, self.task_spt)
                    (
                        x_qry_task_spt_set_cluster,
                        x_qry_task_qry_set_cluster,
                        y_qry_task_spt_set_cluster,
                        y_qry_task_qry_set_cluster,
                    ) = self.construct_task(qry_pack, self.task_qry)

                    x_spt_task_spt_set_batch.append(x_spt_task_spt_set_cluster)
                    x_spt_task_qry_set_batch.append(x_spt_task_qry_set_cluster)
                    x_qry_task_spt_set_batch.append(x_qry_task_spt_set_cluster)
                    x_qry_task_qry_set_batch.append(x_qry_task_qry_set_cluster)
                    y_spt_task_spt_set_batch.append(y_spt_task_spt_set_cluster)
                    y_spt_task_qry_set_batch.append(y_spt_task_qry_set_cluster)
                    y_qry_task_spt_set_batch.append(y_qry_task_spt_set_cluster)
                    y_qry_task_qry_set_batch.append(y_qry_task_qry_set_cluster)

            else:
                for i in range(self.task_cluster_batch_num):  # one batch means one set
                    if mode == "train":
                        if bool(self.if_val):
                            # spt_pack = data_pack[:800]  # 如果是train,传入的data_pack则为前1200个类
                            # qry_pack = data_pack[800:]

                            spt_pack = data_pack[
                                : self.val_split
                            ]  # 如果是train,传入的data_pack则为前1200个类
                            qry_pack = data_pack[self.val_split :]

                        else:
                            spt_pack = qry_pack = data_pack
                    else:
                        if bool(self.if_val):
                            # spt_pack = data_pack[: 800]  # 如果是test，传入的则是训练集的前800个和测试集的400多个的拼接
                            # qry_pack = data_pack[800:]

                            spt_pack = data_pack[
                                : self.val_split
                            ]  # 如果是test，传入的则是训练集的前800个和测试集的400多个的拼接
                            qry_pack = data_pack[self.val_split :]

                        elif (
                            not bool(self.if_val)
                            and self.test_spt_task_from == "meta_train"
                        ):
                            # spt_pack = data_pack[:1200]
                            # qry_pack = data_pack[1200:]

                            spt_pack = data_pack[: self.train_split]
                            qry_pack = data_pack[self.train_split :]
                        else:
                            spt_pack = qry_pack = data_pack
                    if sample_mode == "hierarchical_concentrate":
                        sub_index = random.sample(
                            range(len(spt_pack)), self.candidate_num
                        )
                        spt_pack = spt_pack[sub_index]
                        sub_index = random.sample(
                            range(len(qry_pack)), self.candidate_num
                        )
                        qry_pack = qry_pack[sub_index]
                    else:
                        assert sample_mode == "hierarchical_uniform"
                    (
                        x_spt_task_spt_set_cluster,
                        x_spt_task_qry_set_cluster,
                        y_spt_task_spt_set_cluster,
                        y_spt_task_qry_set_cluster,
                    ) = self.construct_task(spt_pack, self.task_spt)
                    (
                        x_qry_task_spt_set_cluster,
                        x_qry_task_qry_set_cluster,
                        y_qry_task_spt_set_cluster,
                        y_qry_task_qry_set_cluster,
                    ) = self.construct_task(qry_pack, self.task_qry)

                    x_spt_task_spt_set_batch.append(x_spt_task_spt_set_cluster)
                    x_spt_task_qry_set_batch.append(x_spt_task_qry_set_cluster)
                    x_qry_task_spt_set_batch.append(x_qry_task_spt_set_cluster)
                    x_qry_task_qry_set_batch.append(x_qry_task_qry_set_cluster)
                    y_spt_task_spt_set_batch.append(y_spt_task_spt_set_cluster)
                    y_spt_task_qry_set_batch.append(y_spt_task_qry_set_cluster)
                    y_qry_task_spt_set_batch.append(y_qry_task_spt_set_cluster)
                    y_qry_task_qry_set_batch.append(y_qry_task_qry_set_cluster)

            # [c_b, t_n, setsz, 1, 25, 25]
            x_spt_task_spt_set_batch = np.asarray(x_spt_task_spt_set_batch).astype(
                np.float32
            )
            x_spt_task_qry_set_batch = np.asarray(x_spt_task_qry_set_batch).astype(
                np.float32
            )
            x_qry_task_spt_set_batch = np.asarray(x_qry_task_spt_set_batch).astype(
                np.float32
            )
            x_qry_task_qry_set_batch = np.asarray(x_qry_task_qry_set_batch).astype(
                np.float32
            )
            y_spt_task_spt_set_batch = np.asarray(y_spt_task_spt_set_batch).astype(int)
            y_spt_task_qry_set_batch = np.asarray(y_spt_task_qry_set_batch).astype(int)
            y_qry_task_spt_set_batch = np.asarray(y_qry_task_spt_set_batch).astype(int)
            y_qry_task_qry_set_batch = np.asarray(y_qry_task_qry_set_batch).astype(int)

            data_cache.append(
                [
                    x_spt_task_spt_set_batch,
                    x_spt_task_qry_set_batch,
                    x_qry_task_spt_set_batch,
                    x_qry_task_qry_set_batch,
                    y_spt_task_spt_set_batch,
                    y_spt_task_qry_set_batch,
                    y_qry_task_spt_set_batch,
                    y_qry_task_qry_set_batch,
                ]
            )

        return data_cache

    def load_data_cache(self, data_pack):
        """
        Collects several batches data for N-shot learning
        :param data_pack: [cls_num, 20, 84, 84, 1]
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        #  take 5 way 1 shot as example: 5 * 1
        setsz = self.k_shot * self.n_way
        querysz = self.k_query * self.n_way
        data_cache = []

        # print('preload next 50 caches of batchsz of batch.')
        for sample in range(10):  # num of episodes
            x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
            for i in range(self.task_num):  # one batch means one set
                x_spt, y_spt, x_qry, y_qry = [], [], [], []
                selected_cls = np.random.choice(data_pack.shape[0], self.n_way, False)

                for j, cur_class in enumerate(selected_cls):
                    selected_img = np.random.choice(
                        20, self.k_shot + self.k_query, False
                    )

                    # meta-training and meta-test
                    x_spt.append(data_pack[cur_class][selected_img[: self.k_shot]])
                    x_qry.append(data_pack[cur_class][selected_img[self.k_shot :]])
                    y_spt.append([j for _ in range(self.k_shot)])
                    y_qry.append([j for _ in range(self.k_query)])

                # shuffle inside a batch
                perm = np.random.permutation(self.n_way * self.k_shot)
                x_spt = np.array(x_spt).reshape(
                    self.n_way * self.k_shot, 1, self.resize, self.resize
                )[perm]
                y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)[perm]
                perm = np.random.permutation(self.n_way * self.k_query)
                x_qry = np.array(x_qry).reshape(
                    self.n_way * self.k_query, 1, self.resize, self.resize
                )[perm]
                y_qry = np.array(y_qry).reshape(self.n_way * self.k_query)[perm]

                # append [sptsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
                x_spts.append(x_spt)
                y_spts.append(y_spt)
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)

            # [b, setsz, 1, 84, 84]
            x_spts = (
                np.array(x_spts)
                .astype(np.float32)
                .reshape(self.task_num, setsz, 1, self.resize, self.resize)
            )
            y_spts = np.array(y_spts).astype(np.int).reshape(self.task_num, setsz)
            # [b, qrysz, 1, 84, 84]
            x_qrys = (
                np.array(x_qrys)
                .astype(np.float32)
                .reshape(self.task_num, querysz, 1, self.resize, self.resize)
            )
            y_qrys = np.array(y_qrys).astype(np.int).reshape(self.task_num, querysz)

            data_cache.append([x_spts, y_spts, x_qrys, y_qrys])

        return data_cache

    def next_hierarchical(self, mode="train", sample_mode=""):
        """
        Gets next batch from the dataset with name.
        :param mode: The name of the splitting (one of "train", "val", "test")
        :return:
        """
        # update cache if indexes is larger cached num
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0

            if sample_mode == "hierarchical_uniform":
                if mode == "train":
                    self.datasets_cache[mode] = self.load_data_cache_hie(
                        self.datasets[mode], mode, sample_mode
                    )
                else:
                    if bool(self.if_val):
                        # self.datasets_cache[mode] = self.load_data_cache_hie(np.vstack((
                        #     self.datasets['train'][:800], self.datasets[mode])), mode,
                        #     sample_mode)  # 前800是spt_task，后面的是qry_tak

                        if self.dataset != "domainnet":
                            self.datasets_cache[mode] = self.load_data_cache_hie(
                                np.vstack(
                                    (
                                        self.datasets["train"][: self.val_split],
                                        self.datasets[mode],
                                    )
                                ),
                                mode,
                                sample_mode,
                            )  # 前800是spt_task，后面的是qry_tak
                        else:
                            temp_list = []

                            assert (
                                self.datasets["train"].shape[0]
                                == self.datasets[mode].shape[0]
                            )

                            for i in range(self.datasets["train"].shape[0]):
                                temp = np.vstack(
                                    (
                                        self.datasets["train"][i][: self.val_split],
                                        self.datasets[mode][i],
                                    )
                                )
                                temp_list.append(temp)

                            temp_array = np.array(temp_list)
                            # print(temp_array.shape)
                            self.datasets_cache[mode] = self.load_data_cache_hie(
                                temp_array, mode, sample_mode
                            )

                    elif (
                        not bool(self.if_val)
                        and self.test_spt_task_from == "meta_train"
                    ):
                        # self.datasets_cache[mode] = self.load_data_cache_hie(np.vstack((  # 前1200是spt_task，后面的是qry_tak
                        #     self.datasets['train'], self.datasets[mode])), mode, sample_mode)

                        if self.dataset != "domainnet":
                            self.datasets_cache[mode] = self.load_data_cache_hie(
                                np.vstack(
                                    (self.datasets["train"], self.datasets[mode])
                                ),
                                mode,
                                sample_mode,
                            )
                        else:
                            temp_list = []

                            assert (
                                self.datasets["train"].shape[0]
                                == self.datasets[mode].shape[0]
                            )

                            for i in range(self.datasets["train"].shape[0]):
                                temp = np.vstack(
                                    (self.datasets["train"][i], self.datasets[mode][i])
                                )
                                temp_list.append(temp)

                            temp_array = np.asarray(temp_list)
                            # print(temp_array.shape)
                            self.datasets_cache[mode] = self.load_data_cache_hie(
                                temp_array, mode, sample_mode
                            )

                    else:
                        self.datasets_cache[mode] = self.load_data_cache_hie(
                            self.datasets[mode], mode, sample_mode
                        )
                    # breakpoint()

            else:
                assert (
                    sample_mode == "hierarchical_concentrate"
                )  # hierarchical_concentrate

                if mode == "train":
                    self.datasets_cache[mode] = self.load_data_cache_hie(
                        self.datasets[mode], mode, sample_mode
                    )
                else:
                    if bool(self.if_val):
                        # self.datasets_cache[mode] = self.load_data_cache_hie(np.vstack((
                        #     self.datasets['train'][:800], self.datasets[mode])), mode, sample_mode,
                        #     self.candidate_num)  # 前800是spt_task，后面的是qry_tak

                        if self.dataset != "domainnet":
                            self.datasets_cache[mode] = self.load_data_cache_hie(
                                np.vstack(
                                    (
                                        self.datasets["train"][: self.val_split],
                                        self.datasets[mode],
                                    )
                                ),
                                mode,
                                sample_mode,
                            )  # 前800是spt_task，后面的是qry_tak
                        else:
                            temp_list = []

                            assert (
                                self.datasets["train"].shape[0]
                                == self.datasets[mode].shape[0]
                            )

                            for i in range(self.datasets["train"].shape[0]):
                                temp = np.vstack(
                                    (
                                        self.datasets["train"][i][: self.val_split],
                                        self.datasets[mode][i],
                                    )
                                )
                                temp_list.append(temp)

                            temp_array = np.array(temp_list)
                            # print(temp_array.shape)
                            self.datasets_cache[mode] = self.load_data_cache_hie(
                                temp_array, mode, sample_mode
                            )
                            ###

                    elif (
                        not bool(self.if_val)
                        and self.test_spt_task_from == "meta_train"
                    ):
                        # self.datasets_cache[mode] = self.load_data_cache_hie(np.vstack((  # 前1200是spt_task，后面的是qry_tak
                        #     self.datasets['train'], self.datasets[mode])), mode, sample_mode)

                        if self.dataset != "domainnet":
                            self.datasets_cache[mode] = self.load_data_cache_hie(
                                np.vstack(
                                    (self.datasets["train"], self.datasets[mode])
                                ),
                                mode,
                                sample_mode,
                            )
                        else:
                            temp_list = []

                            assert (
                                self.datasets["train"].shape[0]
                                == self.datasets[mode].shape[0]
                            )

                            for i in range(self.datasets["train"].shape[0]):
                                temp = np.vstack(
                                    (self.datasets["train"][i], self.datasets[mode][i])
                                )
                                temp_list.append(temp)

                            temp_array = np.asarray(temp_list)
                            # print(temp_array.shape)
                            self.datasets_cache[mode] = self.load_data_cache_hie(
                                temp_array, mode, sample_mode
                            )

                    else:
                        self.datasets_cache[mode] = self.load_data_cache_hie(
                            self.datasets[mode], mode, sample_mode
                        )
                    # breakpoint()
        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1
        return next_batch

    def next_uniform(self, mode="train"):
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode])

        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch

    # 在 csv 文件中是 filename -> label
    def loadCSV(self, csvf):
        """
        return a dict saving the information of csv
        :param splitFile: csv file name
        :return: {label:[file1, file2 ...]}

        @param csvf:  csv文件图片
        @return: label -> [filename1, filename2, ...]  标签到文件列表的映射
        """
        # label -> [filename1, filename2, ...]  标签到文件列表的映射
        dictLabels = {}

        import csv

        with open(csvf) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=",")
            next(csvreader, None)  # skip (filename, label)
            for i, row in enumerate(csvreader):
                filename = row[0]
                label = row[1]
                # append filename to current label
                if label in dictLabels.keys():
                    dictLabels[label].append(filename)
                else:
                    dictLabels[label] = [filename]
        return dictLabels

    def create_Omniglot(self, root):
        file_name = (
            "omniglot.npy" if not self.contrastive else "omniglot_contrastive.npy"
        )
        if not os.path.isfile(os.path.join(root, file_name)):
            # 加载数据集

            # if root/data.npy does not exist, just download it
            """Return a set of data augmentation transformations as described in the SimCLR paper."""
            s = 1.0
            color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
            basic_transforms = transforms.Compose(
                [
                    lambda x: Image.open(x).convert("L"),
                    lambda x: x.resize((self.resize, self.resize)),
                    lambda x: np.reshape(x, (self.resize, self.resize, 1)),
                    lambda x: np.transpose(
                        x, [2, 0, 1]
                    ),  # 轴变换，应该是将通道数换到第一维度
                    lambda x: x / 255.0,
                ]
            )
            data_transforms = transforms.Compose(
                [
                    lambda x: Image.open(x).convert("L"),
                    lambda x: x.resize((self.resize, self.resize)),
                    transforms.RandomResizedCrop(size=self.resize),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([color_jitter], p=0.8),
                    GaussianBlur(kernel_size=int(0.1 * self.resize), channel=1),
                    transforms.ToTensor(),
                    # lambda x: x / 255.0,
                ]
            )

            x = Omniglot(
                root,
                download=True,
                transform=(
                    transforms.Compose(
                        [
                            ContrastiveLearningViewGenerator(
                                base_transform=basic_transforms,
                                extra_transform=data_transforms,
                                n_views=3,
                            ),
                        ]
                    )
                    if self.contrastive
                    else basic_transforms
                ),
            )

            # temp = (
            #     dict()
            # )  # {label:img1, img2..., 20 imgs, label2: img1, img2,... in total, 1623 label}
            temp = defaultdict(list)
            for img, label in x:
                # print(type(img))  # 28 * 28
                # exit(0)
                # label 就是一个 int 数字

                temp[label].append(img)

            x = []
            for (
                label,
                imgs,
            ) in temp.items():  # labels info deserted , each label contains 20imgs
                # print('label {} 的图片数量是 {} '.format(label, len(imgs)))
                # 由上面的代码知  每个 label 下的图片数量是一样的耶

                x.append(np.array(imgs))
            # 经过上面的操作， self.x 是五维的

            # 下面的 class 应该是指 label
            # as different class may have different number of imgs
            x = np.array(x).astype(float)  # [[20 imgs],..., 1623 classes in total]

            # 上面是错的？
            # 每个 label 下的图片数量是一样的耶

            # each character contains 20 imgs
            print("data shape:", x.shape)  # [1623, 20, 84, 84, 1]
            # (1623, 20, 1, 64, 64) 上面后面的 通道数的位置错了？
            # (图片类型数量, 每个类型下的图片数量, 图片(占3位))

            # exit(0)

            temp = []  # Free memory
            # save all dataset into npy file.
            np.save(os.path.join(root, file_name), x)
            print(f"write into {file_name}.")
        else:
            # if data.npy exists, just load it.
            x = np.load(os.path.join(root, file_name))
            print(f"load from {file_name}.")

        return x

    def create_mini_imagenet(self, root):
        # temp = root
        # root = './'
        npy_path = os.path.join(
            root,
            (
                "miniimagenet.npy"
                if not self.contrastive
                else "miniimagenet_contrastive.npy"
            ),
        )
        if not os.path.isfile(npy_path):
            # root = temp

            mode = "all"
            if mode == "all":
                csvdatas = [
                    self.loadCSV(os.path.join(root, "train.csv")),
                    self.loadCSV(os.path.join(root, "val.csv")),
                    self.loadCSV(os.path.join(root, "test.csv")),
                ]
            else:
                # label -> [filename1, filename2, ...]  标签到文件列表的映射，标签即为图片类别
                csvdatas: dict = [
                    self.loadCSV(os.path.join(root, mode + ".csv"))
                ]  # csv path

            data = []  # [[filename1, filename2, ...], [filename3, filename4, ...], ...]

            # 数据集中的 长字符串 label（如n01532829） 到数字label 的映射
            img2label = {}

            startidx = 2
            for csvdata in csvdatas:
                for i, (k, v) in enumerate(csvdata.items()):
                    # k 是 label，类似于 n01532829 这样的字符串，
                    # v 是 [filename1, filename2, ...]
                    img2label[k] = (
                        i + startidx
                    )  # {"img_name[:9]":label}   # 这里为什么要加上一个 startidx
                    data.append(v)  # [[img1, img2, ...], [img111, ...]]
                startidx += len(csvdata)

            x = []
            path = os.path.join(root, "images")  # image path

            basic_transform = transforms.Compose(
                [
                    lambda x: Image.open(x).convert("RGB"),
                    transforms.Resize((self.resize, self.resize)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
            extra_transform = transforms.Compose(
                [
                    lambda x: Image.open(x).convert("RGB"),
                    transforms.RandomResizedCrop(size=self.resize),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply(
                        [transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8
                    ),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
            if self.contrastive:
                transform = ContrastiveLearningViewGenerator(
                    base_transform=basic_transform,
                    extra_transform=extra_transform,
                    n_views=3,
                )
            else:
                transform = basic_transform
            # 在 mini-imagenet 中transform区分为train和test，但两个是相同的

            # print(len(data)) # 100

            for filename_list in data:
                # 外层循环，遍历图片类别
                img_list = []

                for filename in filename_list:
                    # 内层循环，遍历该类别下的图片名称
                    file_path = os.path.join(path, filename)

                    # print(file_path)
                    # exit()

                    img = transform(file_path)
                    img_list.append(np.array(img))

                x.append(np.array(img_list))
                # print(np.array(img_list).shape)  # (600, 3, 84, 84)

            x = np.array(x).astype(float)
            print("data shape:", x.shape)  # # (100, 600, 3, 84, 84)

            # exit(0)

            # np.save(os.path.join(root, "miniimagenet.npy"), x)
            np.save(npy_path, x)
            print(f"write into {npy_path}.")
        else:
            x = np.load(npy_path)
            print(f"load from {npy_path}")

        return x

    def create_imagenet_1k(self, root):
        # 参数root暂时不起作用

        # 222.20.97.89:11224
        save_path = os.path.join(root, "imagenet-1k.npy")

        # 222.20.96.147:15000
        # save_path = "../../../gyc_data/MAML/imagenet/imagenet-1k.npy"
        # print(save_path)
        # print("进入create")

        if not os.path.isfile(save_path):
            root = os.path.join(root, "val")
            # transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
            #                                 transforms.Resize((self.resize, self.resize)),
            #                                 transforms.ToTensor(),
            #                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            #                                 ])
            transform = ContrastiveLearningViewGenerator(
                base_transform=ContrastiveLearningDataset.get_simclr_pipeline_transform(
                    self.resize
                ),
                n_views=2,
            )

            x = []
            class_list = os.listdir(root)  # 图片类型列表
            i = 1
            for class_name in class_list:
                print(i)
                i += 1

                # 类型路径
                class_path = os.path.join(root, class_name)

                # 图片名称列表
                img_name_list = os.listdir(class_path)

                # 存放张量形式的图片的列表
                img_array_list = []

                for img_name in img_name_list:
                    img_path = os.path.join(class_path, img_name)

                    img_array = transform(img_path)

                    img_array_list.append(np.array(img_array))

                # print(np.array(img_array_list).shape) # (50, 3, 84, 84)
                # exit()

                x.append(np.array(img_array_list))

            x = np.array(x).astype(float)

            print("data shape:", x.shape)

            np.save(save_path, x)
            print("write into imagenet-1k.npy.")
        else:
            # print("存在")
            x = np.load(save_path)
            print("data shape:", x.shape)
            print("load from imagenet-1k.npy.")

        return x

    def create_domainnet(self, root):
        # 参数root暂时不起作用

        # save_path = '../data/DomainNet.npy'
        file_name = (
            "DomainNet.npy" if not self.contrastive else "DomainNet_contrastive.npy"
        )
        save_path = f"../data/DomainNet/{file_name}"

        if not os.path.isfile(save_path):
            #
            data_path = "../data/DomainNet/data"

            domain_list = [f for f in os.listdir(data_path) if not f.startswith(".")]
            domain_list = list(filter(lambda x: x != "Clipart", domain_list))

            print(domain_list)

            class_list_join = []

            class_to_min_num = {}

            min_num = 100

            for domain in domain_list:
                class_path = os.path.join(data_path, domain.lower())
                class_list = os.listdir(class_path)

                class_list_join.append("".join(class_list))

                print(f"{domain} 下有{len(class_list)} 个类别")

                class_to_num = {}

                for class_name in class_list:
                    image_list_path = os.path.join(class_path, class_name)

                    image_list = os.listdir(image_list_path)

                    # print(len(image_list))
                    if min_num > len(image_list):
                        min_num = len(image_list)

                    # 如果不存在或者更小
                    if (
                        class_name not in class_to_min_num
                        or len(image_list) < class_to_min_num[class_name]
                    ):
                        class_to_min_num[class_name] = len(image_list)

                    class_to_num[class_name] = len(image_list)

                print(class_to_num)

            print(f"类别中含有的图片数量最少为{min_num}")

            # 输出 4 个 True，说明每个 domain 下的图片类别是相同的，同时类别排列的顺序是相同的
            for i in range(1, len(class_list_join)):
                print(class_list_join[i] == class_list_join[i - 1], end=" ")

                # print(class_list)

            # 使用字典结构统计每个类别下的最小值，之后按值进行排序
            class_to_min_num_sorted = dict(
                sorted(class_to_min_num.items(), key=lambda x: x[1])
            )
            print(class_to_min_num_sorted)

            min_class_set = set()

            for class_name, num in class_to_min_num_sorted.items():
                if num < 20:
                    min_class_set.add(class_name)
                else:
                    break

            print(min_class_set)
            print(f"图片数量少于20的类别有{len(min_class_set)}个")

            print("\n============================\n")

            # 定义图片变换，源自 drt https://github.com/liyunsheng13/DRT
            basic_transform = transforms.Compose(
                [
                    lambda x: Image.open(x).convert("RGB"),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            s = 1.0
            color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
            extra_transform = transforms.Compose(
                [
                    lambda x: Image.open(x).convert("RGB"),
                    transforms.Resize(256),
                    transforms.RandomResizedCrop(size=224),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([color_jitter], p=0.8),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            if self.contrastive:
                transform = ContrastiveLearningViewGenerator(
                    base_transform=basic_transform,
                    extra_transform=extra_transform,
                    n_views=3,
                )
            else:
                transform = basic_transform

            # 根据上文，类别排列的顺序是相同的，只需要按顺序读取并过滤就好了

            domain_array = []
            if self.contrastive:
                domain_array = np.zeros((6, 299, 20, 3, 3, 224, 224), dtype=np.float32)
            else:
                domain_array = np.zeros((6, 299, 20, 3, 224, 224), dtype=np.float32)

            for d_i, domain in enumerate(domain_list):
                class_path = os.path.join(data_path, domain.lower())
                class_list = os.listdir(class_path)

                class_array = []
                filtered_class_list = [
                    cls for cls in class_list if cls not in min_class_set
                ]
                for c_i, class_name in enumerate(filtered_class_list):
                    # if class_name in min_class_set:
                    #     continue

                    # 存放张量形式的图片的列表，[20（图片数量），图片（3位）]
                    img_array_list = []

                    image_list_path = os.path.join(class_path, class_name)

                    image_list = os.listdir(image_list_path)

                    # 在每个类别下读取 20 张图片即可
                    for i in range(20):
                        image_path = os.path.join(image_list_path, image_list[i])

                        # print(image_path)
                        img_array = transform(image_path)

                        # img_array_list.append(np.array(img_array))
                        domain_array[d_i, c_i, i] = np.array(img_array)

                    # class_array.append(np.asarray(img_array_list))
                    # print(np.array(img_array_list).shape)
                    # exit()
                # print(np.asarray(class_array).shape)
                print(f"完成 domain {domain}")
                # exit()

                # domain_array.append(np.array(class_array))

            # domain_array = np.asarray(domain_array).astype(float)

            print("domain_array.shape：", domain_array.shape)
            np.save(save_path, domain_array)
            print(f"write into {file_name} ")
        else:
            # print("存在")
            domain_array = np.load(save_path)
            print("data shape:", domain_array.shape)
            print(f"load from {file_name}.")
        return domain_array


class MyArgs:
    epoch = 60000
    n_way = 3
    k_spt = 3
    k_qry = 15
    imgsz = 224
    imgc = 3
    task_num = 4
    meta_lr = 1e-3
    update_lr = 0.01
    update_step = 5
    update_step_test = 10

    dataset = "domainnet"
    root = "domainnet"
    val_split = 240
    train_split = 270
    task_cluster_batch_num = 2
    task_spt = 2
    task_qry = 2
    if_val = 0
    test_spt_task_from = "meta_train"
    seed = 222

    candidate_num = 20


if __name__ == "__main__":
    print("hello world")

    args = MyArgs()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    dataset1 = ImgClassification(args)

    # exit()

    temp = dataset1.next_hierarchical(mode="train", sample_mode="hierarchical_uniform")

    print(type(temp))  # list  这里为什么是list
    print(len(temp))  # 8
    print(type(temp[0]))  # numpy.ndarray
    print(temp[0].shape)  # (2, 2, 9, 3, 84, 84)
    # (2, 2, 9, 3, 224, 224)
    torch.save(temp, "./miniDomainnet.pt")

    # y = torch.load("./myTensor.pt")
    exit()
