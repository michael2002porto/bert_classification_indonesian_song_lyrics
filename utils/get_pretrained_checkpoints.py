import os
import requests
from tqdm import tqdm

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
        
        # Send a GET request
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))  # Get total file size
        
        # Download and save the file with progress bar
        with open(filepath, 'wb') as file:
            # Set up the progress bar
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename, ncols=100) as pbar:
                for data in response.iter_content(1024):  # Download in chunks of 1024 bytes
                    file.write(data)
                    pbar.update(len(data))  # Update the progress bar by the chunk size

        print(f"{filename} downloaded successfully!")
