from importlib import import_module
import itertools
from typing import Callable, Union, List, Iterable
from torch.utils.data import DataLoader
from models.encoders.audio_encoders import VGGishEncoder
from tqdm.auto import tqdm
from tqdm.utils import _screen_shape_wrapper
import sys


def fold_generator(dataset, n_runs = 1):

    splits_keys = set([obj['split'] for obj in dataset])
    n_splits = len(splits_keys)

    # Allows for multiple runs with the same splits
    for _ in range(n_runs):
        if all(isinstance(key, int) for key in splits_keys):
            splits = [[] for _ in range(n_splits)]

            for obj in dataset:
                splits[obj['split']].append(obj)

            for i in range(n_splits-1):
                i = 0
                train_splits = splits[:i] + splits[i+2:]
                test = splits[i]
                valid = splits[i+1]
                train = list(itertools.chain(*train_splits))

                yield train, valid, test
        elif all(isinstance(key, str) for key in splits_keys):
            splits = {key:[] for key in splits_keys}

            for obj in dataset:
                splits[obj['split']].append(obj)

            yield splits['train'], splits['validation'], splits['test']
        else:
            assert False, "Unsupported splits definition in Dataset File"


def batchify(
    dataset,
    *, 
    batch_size: int = 16,
    epochs: int = 1,
    load_audio: bool = False,
    ):
    if load_audio:
        dataset = LoadAudioLive(dataset)
    
    loader = MultiEpochsDataLoader(
        dataset=dataset,  
        batch_size=batch_size, 
        collate_fn=lambda list_x: [x for x in list_x] ,
        num_workers=4,
        num_epochs=epochs,
        prefetch_factor=4,
        shuffle=True,
    )
    for batch in loader:
        # assert len(batch) > 0, "HERE IS THE ERROR"
        yield batch


def import_class(
    class_str: str
):
    hierarchy = class_str.split(".")
    module_hierarchy = ".".join(hierarchy[:-1])
    class_name = hierarchy[-1]

    module = import_module(module_hierarchy)
    return getattr(module, class_name)


class LoadAudioLive:

    def __init__(self, dataset, load_fn = None):
        self.dataset = dataset
        if load_fn:
            self.load_fn = load_fn
        else:
            self.load_fn = VGGishEncoder.obj_to_audio

    def __len__(self):
        return len(self.dataset)
    
    #! Adding support for dictionary converstion and ** dereferencing
    def keys(self):
        return self.dataset.keys()

    def __getitem__(self, idx):
        obj = self.dataset[idx].copy()
        
        if "tensor" not in obj:
            try:
                obj['tensor'] = self.load_fn(obj)
            except Exception as e:
                print(e)
                pass

        return obj

class MultiEpochsDataLoader(DataLoader):

    def __init__(self, num_epochs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.num_epochs = num_epochs
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)*self.num_epochs

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class multiline_tqdm(tqdm):

    def __init__(self, *args, desc="", **kwargs):
        
        self.header_line = tqdm(
            bar_format="{desc}", 
            desc=desc, 
            leave=kwargs.get("leave", True),
        )
        super().__init__(*args, **kwargs)

        # Tries to close the unused progress bar in the Header Bar
        if hasattr(self.header_line, 'container'):
            self.header_line.container.children[1].close()

    def set_description(self, *args, **kwargs):
        self.header_line.set_description(*args, **kwargs)

    def close(self, *args, **kwargs):
        super().close(*args, **kwargs)
        self.header_line.close(*args, **kwargs)
