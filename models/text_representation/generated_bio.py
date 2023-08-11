from . import AbstractEncoder
import os
import json


class GeneratedBiographyEncoder(AbstractEncoder):

    NOT_FOUND: str = "Bio: NOT FOUND"

    def __init__(self):
        
        script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
        rel_path = "data/generated_biographies.json"
        abs_file_path = os.path.join(script_dir, rel_path)

        with open(abs_file_path) as infile:
            artist_corpus = json.load(infile)

        self.corpus = {
            obj['author_name'].lower(): obj
                for obj in artist_corpus.values()
        }

    def __call__(self, obj: dict) -> str:

        author_obj = self.corpus.get(obj["author_name"].lower())
        if not author_obj:
            return self.NOT_FOUND
        
        author_bio = author_obj['biography']
        final_passage = f"Bio: {author_bio}"

        return self._clean_text(final_passage)