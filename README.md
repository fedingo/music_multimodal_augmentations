# music_multimodal_augmentations



## Downloading the datasets

The project relies on a set of .json files that contain the information for each dataset.

Each contains a list of dictionary objects of the following format:

```
{
   "author_name":"John Lee Hooker",
   "title":"One Bourbon, One Scotch And One Beer",
   "genre":"blues",
   "filename":"/home/user/projects/subtasks/GTZAN/wav/blues.00000.wav",
   "split":"validation",
   "lyrics":" One bourbon, one scotch and one bill Hey Mr. Bartender, come here I want another drink and I want it now My baby she gone, she been gone tonight I ain't seen my baby since night before One bourbon, one scotch and one bill",
   "sample_id":"blues.00000.wav"
}
```

To use the provided .json files that contain extra annotations (like lyrics) it will be necessary to substitute the filename with the actual position of the files for each object. Alternatively it's possible to generate new .json files that directly point to the correct audio files. Then it's necessary to point to the new files in the [config_file](experiments/configs/datasets.py)

To use the Wikipedia Artist Retriever, it is necessary to build the kilt index using pyserini [Kilt](https://github.com/facebookresearch/KILT) [Pyserini](https://github.com/castorini/pyserini)


## Resource Files:

- Web Bios Corpus [File](models/text_representation/data/author_corpus.json)
- ChatGPT Generated Bios Corpus [File](models/text_representation/data/generated_biographies.json)
- WikiData Artists Data [File](wikidata_artists.json)

## TODOS:
- Add requirements file
- Centralize and make configurable path of data sources 