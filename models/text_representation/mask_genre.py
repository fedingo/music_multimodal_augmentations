from . import AbstractEncoder


def MaskGenre(encoder: AbstractEncoder):

    assert issubclass(encoder, AbstractEncoder), f"Encoder {encoder.__name__}  is not a subclass of AbstractEncoder"

    class MaskGenreEncoder(AbstractEncoder):

        def __init__(self) -> None:
            self.encoder = encoder()

            if hasattr(encoder, "NOT_FOUND"):
                self.NOT_FOUND = encoder.NOT_FOUND

        def __call__(self, obj: dict) -> str:

            target_genre = obj['genre'].lower()
            tmp_out = self.encoder(obj).lower()

            return tmp_out.replace(target_genre, "MASK")
        
    return MaskGenreEncoder