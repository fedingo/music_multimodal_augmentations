from . import AbstractEncoder
import os
import json


class MSDAEncoder(AbstractEncoder):

    NOT_FOUND: str = "Bio: NOT FOUND"

    def __init__(self):
        
        script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
        rel_path = "data/msda_bio_corpus.json"
        abs_file_path = os.path.join(script_dir, rel_path)

        with open(abs_file_path) as infile:
            self.corpus = json.load(infile)

    def __call__(self, obj: dict) -> str:

        author_bio = self.corpus.get(obj["author_name"].lower())
        if not author_bio:
            return self.NOT_FOUND
                
        return self._clean_text(author_bio)