from torch.utils.data.dataloader import DataLoader
from data.get_dataset import get_multi_dataset
from data.multi_dataset_sampler import MultipleDatasetSampler


MetaBTAF = ["BIRD", "TEXTURE", "AIRCRAFT", "FUNGI"]
MetaCIO = ["CIFAR_FS", "miniimagenet", "omniglot_py"]
MetaABF = ["AIRCRAFT", "BIRD", "FUNGI"]
MetaDataset = [
    "ilsvrc",
    "omniglot_py",
    "AIRCRAFT",
    "BIRD",
    "TEXTURE",
    "QuickDraw",
    "FUNGI",
    "VGG",
    "Flower",
    "Traffic signs",
    "MSCOCO",
]  # ilsvrc, quickdraw, vgg, flower, trafficsign, mscoco
miniimagenet = ["miniimagenet"]
tiereimagenet = ["tieredimagenet"]
omniglot = ["omniglot_py"]
CIFAR_FS = ["CIFAR_FS"]
CDFSL = ["BIRD_CDFSL", "CARS", "PLACES", "PLANTAE", "miniimagenet"]
metadatasets = {
    "MetaBTAF": MetaBTAF,
    "MetaCIO": MetaCIO,
    "MetaABF": MetaABF,
    "miniimagenet": miniimagenet,
    "tieredimagenet": tiereimagenet,
    "CIFAR_FS": CIFAR_FS,
    "omniglot_py": omniglot,
    "CDFSL": CDFSL,
}
domainnet = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]


class MetaDatasetsGenerator:
    def __init__(self, args, contrastive=False, batch=2):
        self.stages = {"train": "train", "valid": "valid", "test": "test"}

        self.ds_name = args.datasets
        self.cluster_name = metadatasets[self.ds_name]
        self.test_ds_name = args.test_dataset
        self.test_cluster_name = metadatasets[self.test_ds_name]

        self.m_dataset = {
            stage: get_multi_dataset(
                args, ds_name=self.ds_name, split=stage, contrastive=contrastive
            )
            for stage in self.stages
        }
        if "test" in self.m_dataset.keys():
            self.m_dataset["test"] = get_multi_dataset(
                args, ds_name=self.test_ds_name, split="test", contrastive=contrastive
            )
        self.n_cluster = len(self.m_dataset[self.stages["train"]].cluster_info)
        self.test_n_cluster = len(self.m_dataset["test"].cluster_info)
        max_steps = {
            "train": args.epoch,
            "valid": args.max_test_task * self.n_cluster,
            "test": args.max_test_task * self.test_n_cluster,
        }  # *self.n_cluster*args.batch_size
        shuffling = {"train": True, "valid": False, "test": False}
        # if contrastive:
        #     n_way = 16
        #     k_shot = 24
        # else:
        n_way = args.num_ways
        k_shot = args.num_shots + args.num_shots_test

        self.m_sampler = {
            stage: MultipleDatasetSampler(
                args,
                self.m_dataset[stage],
                total_steps=max_steps[stage],
                n_way=n_way if not contrastive  else args.contrastive_batch_size,
                k_shot=k_shot if not contrastive else 1,
                is_train=shuffling[stage],
                is_random_classes=False,
                contrastive=contrastive,
                batch=batch,
            )
            for stage in self.stages
        }

        self.m_dataloader = {
            stage: DataLoader(
                self.m_dataset[stage],
                batch_sampler=self.m_sampler[stage],
                num_workers=8,
                pin_memory=True,
            )
            for stage in self.stages
        }

        stat_keys_temp = ["loss", "acc", "error_norm"]
        self.stat_keys = []
        for i in range(self.n_cluster + 1):
            if i == self.n_cluster:
                for x in stat_keys_temp:
                    self.stat_keys.append("{}_{}".format(i, x))
            else:
                for x in stat_keys_temp:
                    self.stat_keys.append("{}_{}".format(self.cluster_name[i], x))
        
        for i in range(self.test_n_cluster + 1):
            if i == self.test_n_cluster:
                for x in stat_keys_temp:
                    self.stat_keys.append("test_{}_{}".format(i, x))
            else:
                for x in stat_keys_temp:
                    self.stat_keys.append("test_{}_{}".format(self.test_cluster_name[i], x))

        for x in stat_keys_temp:
            self.stat_keys.append(x)
