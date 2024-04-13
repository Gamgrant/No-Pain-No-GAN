
from emotion_analyzer import EmotionAnalyzer
import matplotlib.pyplot as plt     
from vault_secrets import HUME_API_KEY
import pandas as pd
import os

analyzer = EmotionAnalyzer(HUME_API_KEY)

def process_files_in_batches(files, batch_size):
    # Split files into batches
    for i in range(0, len(files), batch_size):
        yield files[i:i + batch_size]

def update_csv_with_emotions(csv_path, audio_directory, batch_size=10):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Add a new column for the emotions
    df['Top Emotions'] = ""

    # List to collect all file paths
    file_paths = []

    # Loop through the DataFrame and check each file
    for index, row in df.iterrows():
        song_id = row['song_id']
        audio_path = os.path.join(audio_directory, song_id + '.mp3')

        # Check if the file exists
        if not os.path.exists(audio_path):
            df.at[index, 'Top Emotions'] = "N/A"
            continue

        # Add file path to the list for batch processing
        file_paths.append(audio_path)

    # Process files in batches
    for batch in process_files_in_batches(file_paths, batch_size):
        results = analyzer.run_emotion_analysis(batch)
        
        # Update the DataFrame with the results for each file in the batch
        for song_path, emotions in results:
            song_id = os.path.basename(song_path).replace('.mp3', '')
            emotion_list = [emotion[0] for emotion in emotions]  # Extract only the emotion names
            df.loc[df['song_id'] == song_id, 'Top Emotions'] = ', '.join(emotion_list)

    # Save the updated DataFrame to a new CSV file
    df.to_csv('updated_emotions.csv', index=False)

# Example usage
update_csv_with_emotions('path_to_your_csv.csv', 'path_to_your_audio_directory', batch_size=10)



