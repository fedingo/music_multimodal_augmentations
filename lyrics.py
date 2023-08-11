#!/usr/bin/env -S PYTHONPATH=/home/user/projects/MusicTasksAsTextToText python

import click
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import json
import os
from tqdm.auto import tqdm
import warnings
from transformers.utils import logging
from torch.utils.data import DataLoader

logging.set_verbosity_error()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

from experiments.src.utils import LoadAudioLive


class Transcriber():

    def __init__(self) -> None:
        # load model and processor
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-large")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large").cuda()

        self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language = "en", task = "transcribe")


    def __call__(self, obj):
        audio = obj['tensor']
        inputs = self.processor(audio, sampling_rate=16_000, return_tensors="pt").to("cuda")

        predicted_ids = self.model.generate(**inputs) 
        lyrics = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        if len(lyrics) < 10:
            return "No lyrics."
        else:
            return lyrics


def annotate_lyrics(
    dataset_path: str,
):
    # Load Dataset Object
    title = "Lyrics Augmentations Script!"
    print("#", "-"*len(title), "#")
    print("|", title, "|")
    print("#", "-"*len(title), "#")

    with open(dataset_path) as f:
        dataset = json.load(f)

    loader = DataLoader(
        LoadAudioLive(dataset),
        batch_size=1,
        collate_fn=lambda x: x[0], # Skip Collation
        num_workers=16,
    )

    print("Loading models...")
    extract_lyrics = Transcriber()

    try:
        for obj, loaded_obj in tqdm(zip(dataset,loader), total=len(dataset)):
            # If lyrics already in Object, skip
            # This allows for jobs to continue if for any reason they crush
            if "lyrics" in obj:
                continue
            obj['lyrics'] = extract_lyrics(loaded_obj)
    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        print("Canceled, exiting...")

    lyrics_count = 0
    failed_lyrics_count = 0
    for obj in dataset:
        if "lyrics" in obj:
            lyrics_count += 1
            if obj['lyrics'] == "No lyrics.":
                failed_lyrics_count += 1

    print(f"N. items: {len(dataset)}")
    print(f"N. annotate lyrics: {lyrics_count}")
    print(f"N. of failed lyrics: {failed_lyrics_count}")

    # Saves new dataset in place 
    # (This is meant to be an augmentation of the already existing dataset)
    with open(dataset_path, "w") as f:
        json.dump(dataset, f)


@click.command()
@click.option(
    "--dataset_path",
    required=True,
    help="Path to dataset to augment"
)
def run_annotate_lyrics(
    **kwargs
):
    annotate_lyrics(**kwargs)


if __name__ == "__main__":
    run_annotate_lyrics()
