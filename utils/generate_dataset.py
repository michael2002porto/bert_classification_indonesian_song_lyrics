import sys
import json
import pandas as pd

# access the parent folder
sys.path.append(".")

from models.llama3_70b.english_album import *
from models.llama3_70b.translator import TranslateAlbum

if __name__ == '__main__':
    songs_data = []

    get_album = GetAlbum(age_class_tag = "children", num_songs = 10)
    english_album = get_album.setup()
    print(english_album)

    translate_album = TranslateAlbum(english_album = english_album)
    indonesian_album = translate_album.setup()
    print(indonesian_album)

    # with open("data/generated_lyrics.json", "w") as outfile:
    #     json.dump(json.loads(indonesian_album), outfile)

    generated_album = json.loads(indonesian_album)

    for i, song in enumerate(generated_album["songs"], start = 1):
        song_dict = {
            "No": i,
            "Title": song["title"],
            "Lyric": " ".join(song["lyric"]),
            "Age Class tag": "anak"
        }
        songs_data.append(song_dict)

    df = pd.DataFrame(songs_data)

    # Save DataFrame to Excel file
    output_file = "data/generated_lyrics.xlsx"
    df.to_excel(output_file, index=False)