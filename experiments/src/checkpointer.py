import uuid
import torch
import os
from concurrent.futures import ThreadPoolExecutor


class Checkpointer:

    def __init__(
        self,
        model: torch.nn.Module,
        minimize: bool = True
    ) -> None:
        self.model = model

        script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
        path_cache = ".cache/"
        self.cache_path = os.path.join(script_dir, path_cache)

        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        self.sort_fn = min if minimize else max

        self.checkpoints = []

    def __enter__(self):
        # ? Support for with statements
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # ? Support for with statements, with clean_up on exit
        self.load_best()
        self.clean_cache()

        # * If exceptions happended, pass them throught to main execution 
        if exc_type is not None:
            return False 

        return True
        
    

    def __get_model_path(self, model_id) -> str:
        # Get path from a Model ID
        return os.path.join(self.cache_path, model_id+".pt")

    def __save_model(self, model_id) -> None:
        # Saves in cache the model ID
        pt_obj = {
            'model_state_dict': self.model.state_dict(),
        }
        model_path = self.__get_model_path(model_id)

        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(torch.save, pt_obj, model_path)

    def __load_model(self, model_id) -> torch.nn.Module:
        # Load from cache the weights into the current model
        checkpoint = torch.load(self.__get_model_path(model_id))
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def clean_cache(self):
        # Clear cache folder of all the checkpoints (maybe keep best?)
        for obj in self.checkpoints:
            os.remove(self.__get_model_path(obj['model_id']))

        self.checkpoints = []        
        
    def submit_score(self, score: float) -> None:
        new_model_id = str(uuid.uuid4())
        self.__save_model(new_model_id)

        self.checkpoints.append({
            'model_id': new_model_id,
            'score': score
        })

    def load_best(self) -> torch.nn.Module:
        
        if self.checkpoints:
            # Extract Model id with the best Score 
            # [either min or max depending on minimize param]
            best_model_id = self.sort_fn(
                self.checkpoints, 
                key= lambda obj: obj['score']
            )['model_id']
            self.__load_model(best_model_id)


