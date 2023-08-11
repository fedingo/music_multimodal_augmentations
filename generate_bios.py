#!/usr/bin/env -S PYTHONPATH=/home/user/projects/MusicTasksAsTextToText python3.9

import sys
import signal
import time
import openai
import json
from tasks import *
from tqdm import tqdm

class InterruptException(Exception):
    pass

def signal_handler(sig, frame):    
    raise InterruptException()

# Registers handler for SIGINT
signal.signal(signal.SIGINT, signal_handler)


openai.api_key = ""


def generate_biography(artist_name):

  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
          {"role": "system", "content": "I am a music expert, that is specialized in music biographies."},
          {"role": "user", "content": f"Generate a 100 words biography of the music artists {artist_name}:\n\n"},
          ],
    temperature=0.7,
    max_tokens=128,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )

  return response["choices"][0]["message"]["content"]



with open('generated_biographies.json') as infile:
    generated_bios = json.load(infile)

starting_elements = len(generated_bios)

task_list = [
    GenreClassification("GTZAN_official.json"),
    GenreClassification("fma_genre_filtered.json"),
    EmotionRecognition("emoMusic.json"),
    EmotionRecognition("deezer.json"),
    MusicTagging("mtat.json"),
    MusicTagging("jamendo-mt_low_crop.json"),
]

try:

    for idx, task in enumerate(task_list):
        print(f"Running Task {idx}")
        artists = list(set([obj['author_name'] for obj in task]))

        for artist in tqdm(artists):
            if artist == "" or artist.lower() in generated_bios:
                continue
            try:
                bio = generate_biography(artist)
                generated_bios[artist.lower()] = {
                    "author_name": artist,
                    "biography": bio
                }
            except InterruptException as e:
                print("Interrupt")
                raise e
            except:
                print(f"Failed to retrieve biography for artist {artist}, skipping...")

except:
    # Regardless of why we close the script, we want to save the current generated bios
    pass

print(f"Generated {len(generated_bios) - starting_elements} biographies")

with open('generated_biographies.json', "w") as outfile:
    generated_bios = json.dump(generated_bios, outfile)

print("Saved current progress! Closing...")