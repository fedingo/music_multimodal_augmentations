from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List
from torch.utils.data import DataLoader
from models.encoders.audio_encoders import VGGishEncoder
from tqdm.auto import tqdm
from models.text_representation import TextRepresentation
import torch
import json
import os


class AbstractTask(ABC):

    def __init__(self, path: str) -> None:
        # LOADING DATASET
        data_dir = "/home/user/projects/MusicTasksAsTextToText/tasks/data/"
        with open(os.path.join(data_dir, path)) as inp_file:
            self.data = json.load(inp_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __getiter__(self) -> Iterable:
        return self.data
    
    #! Adding support for dictionary converstion and ** dereferencing
    def keys(self):
        return self.data[0].keys()

    @abstractmethod
    def eval(self, batch: List[Dict[str, Any]], predictions) -> Dict[str, float]:
        pass

    @abstractmethod
    def format_targets(self, batch) -> torch.Tensor:
        pass

    @abstractmethod
    def get_model_config(self) -> Dict[str, Any]:
        pass

    def preload_audio(self) -> None:
        class Wrapper:
            def __init__(self, target_dict) -> None:
                self.list_dict = target_dict
            def __getitem__(self, idx) -> dict:
                if 'tensor' in self.list_dict[idx]:
                    return self.list_dict[idx]['tensor']
                else:
                    return VGGishEncoder.obj_to_audio(self.list_dict[idx])
            def __len__(self):
                return len(self.list_dict)

        loader = DataLoader(
                            dataset=Wrapper(self.data),  
                            batch_size=1, 
                            collate_fn=lambda list_x: [x for x in list_x] ,
                            num_workers=4,
                        )
        
        for idx, batch in enumerate(tqdm(loader)):
            self.data[idx]['tensor'] = batch[0].clone().numpy()

    def preload_text(self, encoders):
        encoder = TextRepresentation(encoders=encoders)
        for obj in tqdm(self, leave=False):
            obj['text_representations'] = encoder.encode(obj)