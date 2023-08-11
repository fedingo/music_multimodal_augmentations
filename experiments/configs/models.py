from models import *
from models.text_representation import *
from models.multimodal_head import *
from models.text_representation import *


encoder_specific_lr = [
    ("audio_encoder", 1e-3),
    ("text_encoder",  5e-5),
    ("multimodal_head", 1e-5),
    ("classifier", 1e-3),
    ("modal_classifiers", 1e-3)
]

encoder_specific_lr_2 = [
    ("multi_modal_encoder", [
                                ("audio_encoder", 1e-3),
                                ("text_encoder", 5e-5),
                                ("multimodal_head", 1e-3),
                            ]),
    ("classifier", 1e-3),
]

fixed_high_lr = 1e-3
fixed_low_lr = 5e-5


#! MODEL SPECIFIC CONFIGS
multimodal_base_config = {
    "model_class": MultiModalForClassification,
    "load_audio": True,
    "batch_size": 4,
    "learning_rate": encoder_specific_lr,
    'model_kwargs': {
        'modal_regularization': None,
        'modal_fusion_head': ConcatHead,
    }
}

vggish_config = {
    "model_class": VGGishForClassification,
    "load_audio": True,
    "batch_size": 8,
    "learning_rate": fixed_high_lr,
}

t5_base_config = {
    "model_class": T5EncoderClassification,
    "load_audio": False,
    "batch_size": 8,
    "learning_rate": fixed_low_lr,
}


t5_models_list = [
    # {
    #     **t5_base_config,
    #     "text_representation": [MetadataEncoder],
    # },
    # {
    #     **t5_base_config,
    #     "text_representation": [BiographyEncoder],
    # },
    # {
    #     **t5_base_config,
    #     "text_representation": [WikipediaAuthorEncoder],
    # },
    {
        **t5_base_config,
        "text_representation": [LyricsEncoder],
    },
    {
        **t5_base_config,
        "text_representation": [
                                    MetadataEncoder, 
                                    BiographyEncoder, 
                                    WikipediaAuthorEncoder,
                                    LyricsEncoder,
                                ],
    },
]

multimodal_models_list = [
    {
        **multimodal_base_config,
        "text_representation": [MetadataEncoder],
    },
    {
        **multimodal_base_config,
        "text_representation": [BiographyEncoder],
    },
    {
        **multimodal_base_config,
        "text_representation": [WikipediaAuthorEncoder],
    },
    {
        **multimodal_base_config,
        "text_representation": [LyricsEncoder],
    },
    {
        **multimodal_base_config,
        "text_representation": [
                                    MetadataEncoder, 
                                    BiographyEncoder, 
                                    WikipediaAuthorEncoder,
                                    LyricsEncoder,
                                ],
    },
]




models_list = [
    # vggish_config,
    # *t5_models_list,
    *multimodal_models_list,
]