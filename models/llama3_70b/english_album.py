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


def get_album(age_class_tag: str, num_songs: int, seen_titles) -> Album:
    # new_prompt = ""
    # if seen_titles:
    #     new_prompt = f" Do not generate these song titles again: {list(seen_titles)}."
    #     # new_prompt = f" Do not generate these song titles again: {', '.join(seen_titles)}."
    #     print(new_prompt)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"You are a creative songwriter tasked with producing an album containing {num_songs} unique songs."
                f" Each song should have a different title and exactly 16 unique lyric lines."
                f" You need to create distinct song titles and lyrics for four age categories: children, adolescent, adult, and all ages.\n"
                # Pass the json schema to the model. Pretty printing improves results.
                f" The output should be in JSON format."
                f" The JSON object must use the schema: {json.dumps(Album.schema(), indent=2)}",
            },
            {
                "role": "user",
                "content": f"Fetch an album for {age_class_tag}."
                # f"{new_prompt}",
            },
        ],
        model="llama3-70b-8192",
        # Controls randomness: lowering results in less random completions.
        # As the temperature approaches zero, the model will become deterministic
        # and repetitive.
        temperature=1,
        # Streaming is not supported in JSON mode
        stream=False,
        # Enable JSON mode by setting the response format
        response_format={"type": "json_object"},
    )
    return chat_completion.choices[0].message.content


class GetAlbum():
    # 1. def __init__()
    # 2. def setup()

    def __init__(self, age_class_tag = "adult", num_songs = 20, seen_titles = set()):
        super(GetAlbum, self).__init__()
        self.age_class_tag = age_class_tag
        self.num_songs = num_songs
        self.seen_titles = seen_titles

    def setup(self):
        return get_album(self.age_class_tag, self.num_songs, self.seen_titles)


# album = get_album("adult", 20)
# print(album)