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


def get_album(age_class_tag: str) -> Album:
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an Indonesian creative songwriter tasked with producing an album containing 20 unique songs."
                f" Each song should have a different title and exactly 12 unique lyric lines.\n"
                # Pass the json schema to the model. Pretty printing improves results.
                f" The output should be in JSON format."
                f" The JSON object must use the schema: {json.dumps(Album.schema(), indent=2)}",
            },
            {
                "role": "user",
                "content": f"Fetch an album for {age_class_tag} in Indonesian language",
            },
        ],
        model="llama3-70b-8192",
        temperature=0,
        # Streaming is not supported in JSON mode
        stream=False,
        # Enable JSON mode by setting the response format
        response_format={"type": "json_object"},
    )
    return chat_completion.choices[0].message.content


album = get_album("adult")
# print_album(album)
print(album)