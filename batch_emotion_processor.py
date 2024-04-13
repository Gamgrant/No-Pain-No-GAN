
from emotion_analyzer import EmotionAnalyzer
import matplotlib.pyplot as plt     
from vault_secrets import HUME_API_KEY
import pandas as pd
import os

csv_path = 'album_metadata.csv'  # Adjust as necessary
audio_directory = 'audio'  # Adjust as necessary
analyzer = EmotionAnalyzer(HUME_API_KEY)

def validate_csv_and_audio_files(csv_path, audio_directory):
    df = pd.read_csv(csv_path)

    # check for uniqueness of 'song_id' and 'album_id'
    if df['song_id'].nunique() != len(df) or df['album_id'].nunique() != len(df):
        raise ValueError("song_id or album_id has duplicates in the CSV file.")

    # check that each song ID has a corresponding mp3 file in the audio directory
    missing_files = []
    for song_id in df['song_id']:
        mp3_file = f"{song_id}.mp3"
        if not os.path.exists(os.path.join(audio_directory, mp3_file)):
            missing_files.append(mp3_file)

    if missing_files:
        print("The following files are listed in the CSV but missing in the audio directory:", missing_files)
    else:
        print("All song IDs from the CSV have corresponding audio files in the directory.")

def process_files_in_batches(files, batch_size):
    # Split files into batches
    for i in range(0, len(files), batch_size):
        yield files[i:i + batch_size]

def update_csv_with_emotions(csv_path, audio_directory, analyzer, batch_size=10):
    df = pd.read_csv(csv_path)

    # new column for emotions
    df['Top Emotions'] = ""

    file_paths = []

    for index, row in df.iterrows():
        song_id = row['song_id']
        audio_path = os.path.join(audio_directory, song_id + '.mp3')

        # Check if file exists
        if not os.path.exists(audio_path):
            df.at[index, 'Top Emotions'] = "N/A"
            continue

        # Add file path for batch processing
        file_paths.append(audio_path)

    # Process files in batches
    for batch in process_files_in_batches(file_paths, batch_size):
        results = analyzer.run_emotion_analysis(batch)
        
        # Update the DataFrame with the results for each file in the batch
        for song_path, emotions in results:
            song_id = os.path.basename(song_path).replace('.mp3', '')
            emotion_list = [emotion[0] for emotion in emotions]  # Extract only the emotion names
            df.loc[df['song_id'] == song_id, 'Top Emotions'] = ', '.join(emotion_list)

    df.to_csv('updated_data_with_emotions.csv', index=False)

validate_csv_and_audio_files(csv_path, audio_directory)
# update_csv_with_emotions(csv_path, audio_directory, analyzer, batch_size=10)



