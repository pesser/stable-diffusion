import webdataset as wds
from PIL import Image
import io
import os
import torchvision
from PIL import Image
import glob
import random
import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm
from omegaconf import OmegaConf
from einops import rearrange
import torch


from ldm.util import instantiate_from_config


def dict_collation_fn(samples, combine_tensors=True, combine_scalars=True):
    """Take a list  of samples (as dictionary) and create a batch, preserving the keys.
    If `tensors` is True, `ndarray` objects are combined into
    tensor batches.
    :param dict samples: list of samples
    :param bool tensors: whether to turn lists of ndarrays into a single ndarray
    :returns: single sample consisting of a batch
    :rtype: dict
    """
    batched = {key: [] for key in samples[0]}
    # assert isinstance(samples[0][first_key], (list, tuple)), type(samples[first_key])

    for s in samples:
        [batched[key].append(s[key]) for key in batched]


    result = {}
    for key in batched:
        if isinstance(batched[key][0], (int, float)):
            if combine_scalars:
                result[key] = np.array(list(batched[key]))
        elif isinstance(batched[key][0], torch.Tensor):
            if combine_tensors:
                # import torch

                result[key] = torch.stack(list(batched[key]))
        elif isinstance(batched[key][0], np.ndarray):
            if combine_tensors:
                result[key] = np.array(list(batched[key]))
        else:
            result[key] = list(batched[key])
        # result.append(b)
    return result


class WebDataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, tar_base, batch_size, train=None, validation=None,
                 test=None, num_workers=4, load_ddp=True, n_nodes=1,
                 **kwargs):
        super().__init__(self)
        print(f'Setting tar base to {tar_base}')
        self.tar_base = tar_base
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train = train
        self.validation = validation
        self.test = test
        self.load_ddp = load_ddp
        self.multinode = n_nodes > 1
        self.n_nodes = n_nodes  # n gpu ??

    def make_loader(self, dataset_config, train=True):
        if 'image_transforms' in dataset_config:
            image_transforms = [instantiate_from_config(tt) for tt in dataset_config.image_transforms]
        else:
            image_transforms = []

        image_transforms.extend([torchvision.transforms.ToTensor(),
                                 torchvision.transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        image_transforms = torchvision.transforms.Compose(image_transforms)

        if 'transforms' in dataset_config:
            transforms_config = OmegaConf.to_container(dataset_config.transforms)
        else:
            transforms_config = dict()

        transform_dict = {dkey: load_partial_from_config(transforms_config[dkey]) if transforms_config[
                                                                                         dkey] != 'identity' else identity
                          for dkey in transforms_config}
        img_key = dataset_config.get('image_key', 'jpeg')
        transform_dict.update({img_key: image_transforms})

        shuffle = dataset_config.get('shuffle', 0)

        # TODO fid strategy when n exmples not known beforehand
        n_examples = dataset_config.get('n_examples', 1e6) // self.n_nodes

        shards_to_load = dataset_config.shards
        dset_name = 'unknown'
        if isinstance(shards_to_load, str):
            print(f'Loading tars based on the string {shards_to_load}')
            tars = os.path.join(self.tar_base, shards_to_load)
            start_shard_id, end_shard_id = dataset_config.shards.split('{')[-1].split('}')[0].split('..')
            n_shards = int(end_shard_id) - int(start_shard_id) + 1
            dset_name = dataset_config.shards.split('-')[0]
        elif isinstance(shards_to_load, int):
            print(f'Creating tar list, max shard is {shards_to_load}')
            try:
                tars = [tf for tf in natsorted(glob(os.path.join(self.tar_base, '*.tar'))) if
                        int(tf.split('/')[-1].split('.')[0]) < shards_to_load]
                n_shards = len(tars)
                random.shuffle(tars)

            except ValueError as e:
                print('tarfile names should follow the pattern <zero_padded_number>.tar . Check names of the files')
                raise e
        else:
            raise ValueError(
                'shards should be either a string containing consecutive shards or an int defining the max shard number')

        print(f'Got {n_shards} shard files in datafolder for {"training" if train else "validation"}')

        # if self.num_workers > 0:
        #     assert n_shards % self.num_workers == 0 , f'Number of workers which is {self.num_workers} does not evenly divide number of shards which is {n_shards}'
        print(f'Loading webdataset based dataloader based on {n_shards} of {dset_name} dataset.')

        # start creating the dataset
        nodesplitter = wds.shardlists.split_by_node if self.multinode else wds.shardlists.single_node_only
        epoch_length = n_examples // (self.batch_size)

        dset = wds.WebDataset(tars, nodesplitter=nodesplitter).shuffle(shuffle)

        with_epoch_args = {'nsamples': n_examples, 'nbatches': epoch_length}

        if 'filters' in dataset_config:
            for stage in tqdm(dataset_config.filters,
                              desc=f'Applying the following filters: {[f for f in dataset_config.filters]}'):
                f = getattr(dset, stage)
                dset = f(dset, *dataset_config.filters[stage].args,
                         **dataset_config.filters[stage].get('kwargs', dict()))

        print(f'Dataset holding {len(dset.pipeline[0].urls)} shards')

        def ignore_me(*args, **kwargs):
            pass

        dset = (dset
                .decode('pil', handler=ignore_me)
                # .to_tuple("jpg;png;jpeg pickle cls hls")
                # .map_tuple(image_transforms,load_partial_from_config(nns_transform) if 'target' in nns_transform else identity,identity,identity)
                .map_dict(**transform_dict)
                .repeat()
                .batched(self.batch_size, partial=False,
                         collation_fn=dict_collation_fn)
                .with_length(n_examples)
                .with_epoch(**with_epoch_args)
                )

        loader = wds.WebLoader(dset, batch_size=None, shuffle=False,
                               num_workers=self.num_workers)

        return loader, n_examples

    def train_dataloader(self):
        assert self.train is not None
        loader, dset_size = self.make_loader(self.train)
        # if self.load_ddp:
        #     loader = loader.ddp_equalize(dset_size // self.batch_size)
        return loader

    def val_dataloader(self):
        assert self.train is not None
        loader, _ = self.make_loader(self.validation, train=False)
        return loader

    def test_dataloader(self):
        assert self.train is not None
        loader, _ = self.make_loader(self.test, train=False)
        return loader


if __name__ == "__main__":
    url = "pipe:aws s3 cp s3://s-datasets/laion5b/laion2B-data/000000.tar -"
    dataset = wds.WebDataset(url)
    example = next(iter(dataset))
    for k in example:
        print(k, type(example[k]))

    print(example["__key__"])
    for k in ["json", "txt"]:
        print(example[k].decode())

    image = Image.open(io.BytesIO(example["jpg"]))
    outdir = "tmp"
    os.makedirs(outdir, exist_ok=True)
    image.save(os.path.join(outdir, example["__key__"] + ".png"))


    def load_example(example):
        return {
            "key": example["__key__"],
            "image": Image.open(io.BytesIO(example["jpg"])),
            "text": example["txt"].decode(),
        }


    for i, example in tqdm(enumerate(dataset)):
        ex = load_example(example)
        print(ex["image"].size, ex["text"])
        if i >= 100:
            break
