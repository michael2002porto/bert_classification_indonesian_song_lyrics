from typing import List, Set
import json
import pandas as pd
import re
import random

from pydantic import BaseModel, conlist, conset
from groq import Groq

client = Groq(
    api_key="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",
)


# Data model for LLM to generate
class Song(BaseModel):
    title: str
    lyric: Set[str]
    # lyric: conset(str, min_items=10, max_items=10)
    # lyric: conlist(str, min_items=10, unique_items=True)
    # lyric: List[str]


class Album(BaseModel):
    age_class_tag: str
    songs: List[Song]
    # songs: conset(Song, min_items=10, max_items=10)
    # songs: conlist(Song, min_items=10)
    # songs: List[Song]


def convert_existing_dataset():
    dataset = pd.read_excel("data/dataset_lyrics.xlsx")
    dataset = dataset[["Title", "Lyric", "Age Class tag"]]

    # Initialize a dictionary to hold the albums
    albums = {}

    # Process each row in the dataset
    for _, row in dataset.iterrows():
        title = row["Title"].replace("Lirik Lagu ", "")
        lyrics = set(re.findall(r'[A-Z][^A-Z]*', row["Lyric"]))
        age_class_tag = row["Age Class tag"]

        # Create a Song object
        song = Song(title=title, lyric=lyrics)

        # Add the song to the corresponding album
        if age_class_tag not in albums:
            albums[age_class_tag] = Album(age_class_tag=age_class_tag, songs=set())
        albums[age_class_tag].songs.append(song)

    # Convert albums to JSON
    json_data = {age_class_tag: album.json(indent=2, ensure_ascii=False) for age_class_tag, album in albums.items()}
    
    return json_data


def synthesize_album(age_class_tag: str, num_songs: int, seen_titles, existing_dataset) -> Album:
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"You are an Indonesian creative songwriter tasked with producing an album containing {num_songs} unique songs."
                f" You will generate multiple unique songs based on the following existing dataset: {json.dumps(existing_dataset)}."
                f" Each song should have a different title and exactly 16 unique lyric lines."
                f" You need to create distinct song titles and lyrics for four age categories: children, adolescent, adult, and all ages.\n"
                # Pass the json schema to the model. Pretty printing improves results.
                f" The output should be in JSON format."
                f" The JSON object must use the schema: {json.dumps(Album.schema(), indent=2)}",
            },
            {
                "role": "user",
                "content": f"Synthesize an Indonesian album for {age_class_tag}."
                f" Make sure the song titles are not repeated from the existing dataset or among the newly generated songs.",
            },
        ],
        model="llama3-70b-8192",
        temperature=1,
        # Streaming is not supported in JSON mode
        stream=False,
        # Enable JSON mode by setting the response format
        response_format={"type": "json_object"},
    )
    return chat_completion.choices[0].message.content


class SynthesizeAlbum():
    # 1. def __init__()
    # 2. def setup()

    def __init__(self, age_class_tag = "adult", num_songs = 20, seen_titles = set()):
        super(SynthesizeAlbum, self).__init__()
        self.age_class_tag = age_class_tag
        self.num_songs = num_songs
        self.seen_titles = seen_titles

    def setup(self):
        # save existing dataset to json
        # json_data = convert_existing_dataset()
        # for age_class_tag, album_json in json_data.items():
        #     with open(f"data/existing_dataset/{age_class_tag}_album.json", "w") as outfile:
        #         outfile.write(album_json)

        label = self.age_class_tag.replace(" ", "_")
        existing_dataset = json.load(open(f"data/existing_dataset/{label}.json"))

        # Get the existing songs
        songs = existing_dataset["songs"]

        # Randomly select 20 items from the songs list
        # Karena keterbatasan prompt LLM (muncul error jika terlalu banyak)
        if len(songs) >= 20:
            selected_songs = random.sample(songs, 20)
        else:
            selected_songs = songs

        # Update the existing_dataset with the selected songs
        existing_dataset["songs"] = selected_songs
        # print(existing_dataset)

        return synthesize_album(self.age_class_tag, self.num_songs, self.seen_titles, existing_dataset)


# album = synthesize_album("adult")
# print(album)