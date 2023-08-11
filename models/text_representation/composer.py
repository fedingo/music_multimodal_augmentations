from typing import List
from . import AbstractEncoder


class Composer:

    def __init__(self, encoder_list: List[AbstractEncoder] = []):
        self.encoder_list = []

        # Force checks when building object
        for encoder in encoder_list:
            self.append(encoder)

    def append(self, encoder: AbstractEncoder):

        assert issubclass(encoder, AbstractEncoder), f"Encoder {encoder.__name__}  is not a subclass of AbstractEncoder"
        self.encoder_list.append(encoder())

    def __call__(self, obj):
        return [encoder(obj) for encoder in self.encoder_list]