import argparse
import sys
import json
import pandas as pd
from tqdm import tqdm

# access the parent folder
sys.path.append(".")

from models.llama3_70b.english_album import *
from models.llama3_70b.translator import TranslateAlbum

def collect_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_songs_per_label", type=int, default=100)
    parser.add_argument("--num_songs_per_llm_batch", type=int, default=10)

    return parser.parse_args()

if __name__ == '__main__':
    args = collect_parser()

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

    total_songs = args.num_songs_per_label * 4  # Total number of songs to generate (100 per label)
    pbar = tqdm(total=total_songs, desc="Generating songs")

    for key, value in label.items():
        num_songs_per_label = 0

        while num_songs_per_label < args.num_songs_per_label:
            try:
                get_album = GetAlbum(
                    age_class_tag = key,
                    num_songs = args.num_songs_per_llm_batch,
                    seen_titles = seen_english_titles
                )
                english_album = get_album.setup()
                # print(english_album)

                generated_english_album = json.loads(english_album)

                for song in generated_english_album["songs"]:
                    if song["title"] not in seen_english_titles:
                        seen_english_titles.add(song["title"])

                translate_album = TranslateAlbum(english_album = english_album)
                indonesian_album = translate_album.setup()
                # print(indonesian_album)

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
                    pbar.update(1)

            except Exception as e:
                print(f"Error processing songs: {e}")
                # Save songs_data in case of error (optional)
                df = pd.DataFrame(songs_data)
                df.to_excel("data/generated_lyrics_partial.xlsx", index=False)

    pbar.close()
    df = pd.DataFrame(songs_data)

    # Save DataFrame to Excel file
    output_file = "data/generated_lyrics.xlsx"
    df.to_excel(output_file, index=False)