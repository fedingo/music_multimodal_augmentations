from tasks import GenreClassification, EmotionRecognition, MusicTagging

#! DATASET SPECIFIC CONFIGS

#? Genre Classification
GTZAN_official_config = {
    "task_class": GenreClassification,
    "dataset_path": "GTZAN_official.json",
    "maximize_metric": "Accuracy",
    "epochs": 50,
}

FMA_config = {
    "task_class": GenreClassification,
    "dataset_path": "fma_genre_filtered.json",
    "maximize_metric": "Accuracy",
    "epochs": 10,
}

#? Emotion Recognition
EmoMusic_config = {
    "task_class": EmotionRecognition,
    "dataset_path": "emoMusic.json",
    "maximize_metric": "R2Score",
    "epochs": 25,
}

Deezer_config = {
    "task_class": EmotionRecognition,
    "dataset_path": "deezer.json",
    "maximize_metric": "R2Score",
    "epochs": 10,
}

#? Music Tagging
MTAT_config = {
    "task_class": MusicTagging,
    "dataset_path": "mtat_unfiltered.json",
    "maximize_metric": "mAP",
    "epochs": 10,
}

JamendoMT_config = {
    "task_class": MusicTagging,
    "dataset_path": "jamendo-mt_low_crop.json",
    "maximize_metric": "mAP",
    "epochs": 10,
}



datasets_list = [
    GTZAN_official_config, 
    FMA_config,
    EmoMusic_config,
    Deezer_config,
    MTAT_config,
    JamendoMT_config,
]