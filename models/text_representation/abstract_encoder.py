from abc import ABC, abstractmethod
from bs4 import BeautifulSoup


class AbstractEncoder(ABC):

    def _clean_text(self, passage: str) -> str:
        # First, lets strip HTML tags out
        soup = BeautifulSoup(passage, features="lxml")
        passage = soup.get_text()

        # If very long bio, take first passage
        if len(passage) > 1000:
            passage = passage.split("\n")[0]

        # If very long passage, take first 4 sentences
        if len(passage) > 1000:
            sentences = passage.split(".")
            passage = ".".join(sentences[:4])

        return passage.strip()
    
    @abstractmethod
    def __call__(self, obj: dict) -> str:
        pass