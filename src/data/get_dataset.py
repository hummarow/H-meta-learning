from data.cls.meta_dataset import MetaDatasets
from data.cls.miniimagenet import MiniImageNetDataset

# from torchmeta.datasets import MiniImagenet as MiniImageNetDataset
# from data.cls.mini_tiered_CIFAR import get_meta_dataset
from data.cls.tieredimagenet import TieredImageNetDataset
from data.cls.omniglot import Omniglot
from data.cls.cdfsl import CDFSL
from data.dataset_utils import DatasetEnum
from data.multi_dataset_sampler import MultipleDataset

# from data.reg.lines import LinesDataSet
from data.cls.cifar_fs import CifarFS_dataset


def get_dataset(ds_name, split, contrastive=False):
    """
    get a traditional dataset
    :param ds_name:
    :param split:
    :return:
    """
    #####################################
    # cls
    #####################################
    if ds_name == DatasetEnum.miniimagenet.name:
        return MiniImageNetDataset(split=split, contrastive=contrastive)
    elif ds_name == DatasetEnum.tieredimagenet.name:
        return TieredImageNetDataset(split=split, contrastive=contrastive)
    elif ds_name == DatasetEnum.CIFAR_FS.name:
        return CifarFS_dataset(split=split, contrastive=contrastive)
    elif ds_name in (
        DatasetEnum.FUNGI.name,
        DatasetEnum.AIRCRAFT.name,
        DatasetEnum.BIRD.name,
        DatasetEnum.TEXTURE.name,
    ):
        return MetaDatasets(ds_name=ds_name, split=split, contrastive=contrastive)
    elif ds_name == DatasetEnum.omniglot_py.name:
        return Omniglot(split=split, contrastive=contrastive)
    #####################################
    # reg
    #####################################
    # elif ds_name == DatasetEnum.Lines.name:
    #     return LinesDataSet()
    elif ds_name in (
        DatasetEnum.BIRD_CDFSL.name,
        DatasetEnum.CARS.name,
        DatasetEnum.PLACES.name,
        DatasetEnum.PLANTAE.name,
    ):
        return CDFSL(split=split, ds_name=ds_name, contrastive=contrastive)
    else:
        raise ValueError("unknown dataset: {}, {}".format(ds_name, split))


def get_multi_dataset(args, ds_name, split="train", contrastive=False):
    """
     get a meta-dataset
    :param ds_name:
    :param split:
    :return:
    """
    # MODIFICATION
    # Test domain adopted. Held-out domain is designated for testing and is
    # specified in the args.test_domain_selected. The rest of the domains are
    # used for training.
    metadatasets = []
    # print(ds_name, DatasetEnum.MINI_IMAGENET.name)

    def _get_datasets(dataset_enum_list):
        for ds_enum in dataset_enum_list:
            metadatasets.append(
                (
                    ds_enum.name,
                    get_dataset(
                        ds_name=ds_enum.name, split=split, contrastive=contrastive
                    ),
                )
            )

    # get a DatasetEnum object from the string ds_name
    if split == "train" or split == "valid":
        try:
            ds_enum = DatasetEnum(ds_name)
        except ValueError:
            raise ValueError("unknown meta-dataset: {}".format(ds_name))

        all_clusters = ds_enum.get_clusters()
        if args.holdout:
            all_clusters.remove(DatasetEnum(args.test_domain))
        _get_datasets(all_clusters)
    elif split == "test":
        _get_datasets([DatasetEnum(args.test_domain)])
    else:
        raise ValueError("unknown split: {}".format(split))

    multiple_ds = MultipleDataset(metadatasets)
    return multiple_ds
