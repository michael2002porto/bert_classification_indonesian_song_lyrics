import os
import requests

if __name__ == '__main__':
    # Create 'checkpoints' directory if it doesn't exist
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    # List of URLs to download
    urls = [
        'https://github.com/michael2002porto/bert_classification_indonesian_song_lyrics/releases/download/pretrained_checkpoints/split_synthesized.ckpt',
        'https://github.com/michael2002porto/bert_classification_indonesian_song_lyrics/releases/download/pretrained_checkpoints/split_generated_1.ckpt',
        'https://github.com/michael2002porto/bert_classification_indonesian_song_lyrics/releases/download/pretrained_checkpoints/split_generated_2.ckpt',
        'https://github.com/michael2002porto/bert_classification_indonesian_song_lyrics/releases/download/pretrained_checkpoints/split_full_combination.ckpt',
    ]

    # Loop over each URL and download the corresponding file
    for url in urls:
        # Extract the filename from the URL (the part after the last '/')
        filename = url.split('/')[-1]
        # Full path to save the file
        filepath = os.path.join("checkpoints", filename)
        
        # Download the file
        print(f"Downloading {filename}...")
        r = requests.get(url)
        
        # Save the file in 'checkpoints/' directory
        with open(filepath, 'wb') as f:
            f.write(r.content)
        
        print(f"{filename} downloaded successfully!")
