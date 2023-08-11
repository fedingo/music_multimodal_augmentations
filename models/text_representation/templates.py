from string import Template
from . import AbstractEncoder


class TemplateEncoder(AbstractEncoder):

    def __init__(self, pattern) -> None:
        self.template = Template(pattern)


    def __call__(self, obj) -> str:
        representation = self.template.substitute(obj)
        
        return self._clean_text(representation)
    

class MetadataEncoder(TemplateEncoder):

    NOT_FOUND: str = "Author: Unknown\nTitle: Unknown"

    def __init__(self) -> None:
        pattern = """Author: $author_name\nTitle: $title"""
        super().__init__(pattern)


class LyricsEncoder(TemplateEncoder):

    NOT_FOUND: str = "Lyrics: No lyrics."

    def __init__(self) -> None:
        pattern = """Lyrics: $lyrics"""
        super().__init__(pattern)
