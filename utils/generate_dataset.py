import sys
import json

# access the parent folder
sys.path.append(".")

from models.llama3_70b.english_album import *
from models.llama3_70b.translator import TranslateAlbum

if __name__ == '__main__':
    get_album = GetAlbum(age_class_tag = "all ages", num_songs = 10)
    english_album = get_album.setup()
    print(english_album)

    translate_album = TranslateAlbum(english_album = english_album)
    indonesian_album = translate_album.setup()
    print(indonesian_album)

    with open("data/generated_lyrics.json", "w") as outfile:
        json.dump(json.loads(indonesian_album), outfile)