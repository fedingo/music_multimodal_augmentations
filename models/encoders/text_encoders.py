from torch import nn, mean
from transformers import T5Tokenizer, T5EncoderModel, BertTokenizer, BertModel
from abc import ABC, abstractmethod
from transformers.modeling_outputs import BaseModelOutputWithPooling
from typing import List


class AbstractTextEncoder(nn.Module, ABC):

    def __init__(self) -> None:
        super().__init__()

        assert hasattr(type(self), "tokenizer"), \
                f"Tokenizer should be defined for class {type(self)}"

    @classmethod
    def preprocess_inputs(cls, batch: List[dict], *, device: str = "cpu"):
        text_samples = ["\n".join(obj['text_representations']) for obj in batch]
        inputs = cls.tokenizer(
            text_samples,
            padding=True,
            return_tensors="pt",
            truncation=True,
        ).to(device)
        return dict(inputs)

    @abstractmethod
    def forward(self, **kwargs):
        pass


t5_version = "t5-large"
class T5Encoder(AbstractTextEncoder):

    OUTPUT_SHAPE = 1024

    tokenizer = T5Tokenizer.from_pretrained(
        t5_version,
        do_lower_case = True,
        max_length = 512,
        truncation=True,
    )

    def __init__(self, pretrained=True) -> None:
        super().__init__()
        self.model = T5EncoderModel.from_pretrained(t5_version)

        if not pretrained:
            self.model = T5EncoderModel(config=self.model.config)

    def forward(self, **kwargs):
        output = self.model(**kwargs)

        sequence_vector = mean(output.last_hidden_state, axis = 1)
        return BaseModelOutputWithPooling(
            pooler_output=sequence_vector, 
            last_hidden_state=output.last_hidden_state
        )


class BERTEncoder(AbstractTextEncoder):

    OUTPUT_SHAPE = 768

    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased',
        do_lower_case = True,
        max_length = 512,
        truncation=True,
    )

    def __init__(self) -> None:
        super().__init__()

        self.model = BertModel.from_pretrained(
            'bert-base-uncased')

    def forward(self, **kwargs):
        return self.model(**kwargs)