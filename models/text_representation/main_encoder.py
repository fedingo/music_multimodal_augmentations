from .composer import Composer
from typing import List, Type


class TextRepresentation:

    def __init__(
        self,
        encoders: List[Type]
    ) -> None:

        assert len(encoders) > 0, "No Encoder specified"
        self.encoder = Composer(encoders)

    def encode(self, obj):
        return self.encoder(obj)
