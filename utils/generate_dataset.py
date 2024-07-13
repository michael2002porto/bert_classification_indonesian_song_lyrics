import sys
import json
import pandas as pd

# access the parent folder
sys.path.append(".")

from models.llama3_70b.english_album import *
from models.llama3_70b.translator import TranslateAlbum

if __name__ == '__main__':
    songs_data = []
    seen_english_titles = set()  # To track seen titles
    seen_indonesian_titles = set()  # To track seen titles
    i = 1

    label = {
        'all ages': 'semua usia',
        'children': 'anak',
        'adolescent': 'remaja',
        'adult': 'dewasa'
    }

    for key, value in label.items():
        num_songs_per_label = 0

        while num_songs_per_label < 100:
            get_album = GetAlbum(age_class_tag = key, num_songs = 10, seen_titles = seen_english_titles)
            english_album = get_album.setup()
            print(english_album)

            generated_english_album = json.loads(english_album)

            for song in generated_english_album["songs"]:
                if song["title"] not in seen_english_titles:
                    seen_english_titles.add(song["title"])

            translate_album = TranslateAlbum(english_album = english_album)
            indonesian_album = translate_album.setup()
            print(indonesian_album)

            # with open("data/generated_lyrics.json", "w") as outfile:
            #     json.dump(json.loads(indonesian_album), outfile)

            generated_indonesian_album = json.loads(indonesian_album)

            for song in generated_indonesian_album["songs"]:
                if song["title"] in seen_indonesian_titles:
                    continue

                seen_indonesian_titles.add(song["title"])
                song_dict = {
                    "No": i,
                    "Title": song["title"],
                    "Lyric": " ".join(song["lyric"]),
                    "Age Class tag": value
                }
                songs_data.append(song_dict)
                num_songs_per_label += 1
                i += 1

    df = pd.DataFrame(songs_data)

    # Save DataFrame to Excel file
    output_file = "data/generated_lyrics.xlsx"
    df.to_excel(output_file, index=False)