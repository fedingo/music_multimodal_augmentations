from . import AbstractEncoder
import os
import json


class BiographyEncoder(AbstractEncoder):

    NOT_FOUND: str = "Bio: NOT FOUND"

    def __init__(self):
        
        script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
        rel_path = "data/author_corpus.json"
        abs_file_path = os.path.join(script_dir, rel_path)

        with open(abs_file_path) as infile:
            artist_corpus = json.load(infile)

        self.corpus = {
            obj['author_name'].lower(): obj
                for obj in artist_corpus
        }

    def __call__(self, obj: dict) -> str:

        author_obj = self.corpus.get(obj["author_name"].lower())
        if not author_obj:
            return self.NOT_FOUND
        
        passages = [self._clean_text(bio) for bio in author_obj['biographies'] if bio]
        # Give higher priority to longer passages
        passages.sort(reverse=True, key=lambda s: len(s))

        # Remove Duplicates
        passages = list(set(passages))

        if "" in passages:
            passages.remove("")

        final_passage = "\n".join([f"Bio {n}: {passage}" for n, passage in enumerate(passages)])

        return self._clean_text(final_passage)