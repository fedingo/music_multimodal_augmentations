from transformers import BertTokenizer, BertModel
from typing import List
from .classification_head import ClassificationHead
from abc import ABC, abstractmethod
from .encoders.text_encoders import AbstractTextEncoder, T5Encoder

class AbstractClassification(ABC):

    def __init__(self, enc_dim: int, **kwargs) -> None:
        super().__init__()
        self.classifier = ClassificationHead(enc_dim, **kwargs)

    @abstractmethod
    def forward(self, **kwargs):
        pass


class BERTClassification(AbstractClassification, AbstractTextEncoder):

    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased',
        do_lower_case = True,
        max_length = 512,
        truncation=True,
    )

    def __init__(self, **kwargs) -> None:
        super().__init__(768, **kwargs)

        self.model = BertModel.from_pretrained(
            'bert-base-uncased')

    def forward(self, labels=None, **kwargs):
        text_encoded  = self.model(**kwargs).pooler_output

        return self.classifier(text_encoded, labels)


class T5EncoderClassification(AbstractClassification, T5Encoder):

    def __init__(self, pretrained=True, **kwargs) -> None:
        T5Encoder.__init__(self, pretrained)
        AbstractClassification.__init__(self, 1024, **kwargs)
        

    def forward(self, labels=None, **kwargs):
        sequence_vector  = T5Encoder.forward(self, **kwargs).pooler_output

        return self.classifier(sequence_vector, labels)