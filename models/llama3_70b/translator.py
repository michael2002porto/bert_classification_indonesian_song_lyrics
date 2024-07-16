from typing import List, Set
import json

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
    songs: Set[Song]
    # songs: conset(Song, min_items=10, max_items=10)
    # songs: conlist(Song, min_items=10)
    # songs: List[Song]


def translate_album(english_album) -> Album:
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a professional song translator tasked with translating English album to Indonesian album."
                f" Preserving the original meaning and flow of the lyrics.\n"
                # Pass the json schema to the model. Pretty printing improves results.
                f" The output should be in JSON format and in Indonesian language."
                f" The JSON object must use the schema: {json.dumps(Album.schema(), indent=2)}",
            },
            {
                "role": "user",
                "content": f"Translate this JSON album: {json.dumps(english_album)}",
            },
        ],
        model="llama3-8b-8192",
        temperature=0.7,
        # Streaming is not supported in JSON mode
        stream=False,
        # Enable JSON mode by setting the response format
        response_format={"type": "json_object"},
    )
    return chat_completion.choices[0].message.content


class TranslateAlbum():
    # 1. def __init__()
    # 2. def setup()

    def __init__(self, english_album = json.load(open("data/generative/english_album.json"))):
        super(TranslateAlbum, self).__init__()
        self.english_album = english_album

    def setup(self):
        return translate_album(self.english_album)


# english_album = json.load(open("data/generative/english_album.json"))
# indonesian_album = translate_album(english_album)
# print_album(album)
# print(indonesian_album)