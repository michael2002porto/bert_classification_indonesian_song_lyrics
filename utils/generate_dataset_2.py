import argparse
import sys
import json
import pandas as pd
from tqdm import tqdm
from collections import deque

# access the parent folder
sys.path.append(".")

from models.llama3_70b.gen_indo_album import GetAlbum

def collect_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_songs_per_label", type=int, default=100)
    parser.add_argument("--num_songs_per_llm_batch", type=int, default=10)

    return parser.parse_args()

if __name__ == '__main__':
    args = collect_parser()

    songs_data = []
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
                    seen_titles = seen_indonesian_titles
                )
                indonesian_album = get_album.setup()
                print(indonesian_album)

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