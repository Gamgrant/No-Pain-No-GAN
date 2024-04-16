
from emotion_analyzer import EmotionAnalyzer
from error_record_formatter import error_record_formatter
import matplotlib.pyplot as plt     
from vault_secrets import HUME_API_KEYS 
import pandas as pd
import os
import json
import traceback
import time

def validate_csv_and_audio_files(csv_path, audio_directory):
    df = pd.read_csv(csv_path)
    if df['song_id'].nunique() != len(df) or df['album_id'].nunique() != len(df):
        raise ValueError("song_id or album_id has duplicates in the CSV file.")
    missing_files = []
    found_files = []
    for song_id in df['song_id']:
        mp3_file = os.path.join(audio_directory, f"{song_id}.mp3")
        if not os.path.exists(mp3_file):
            missing_files.append(mp3_file)
        else:
            found_files.append(mp3_file)
    if missing_files:
        print(f"{len(missing_files)} files are missing in the audio directory.")
    print(f"{len(found_files)} files are found and will be processed.")

def process_files_in_batches(files, batch_size):
    for i in range(0, len(files), batch_size):
        yield files[i:i + batch_size]

def update_csv_with_emotions(csv_path, audio_directory, analyzer, batch_size=10, k=3):
    batch_num = 0
    error_count = 0
    df = pd.read_csv(csv_path)
    if 'Top Emotions' not in df.columns:
        df['Top Emotions'] = ""
    file_paths = [
        os.path.join(audio_directory, f"{row['song_id']}.mp3") 
        for index, row in df.iterrows()
        if os.path.exists(os.path.join(audio_directory, f"{row['song_id']}.mp3"))
        and (pd.isna(row['Top Emotions']) or row['Top Emotions'].strip() == "")
    ]
    print(f"Found {len(file_paths)} files to process.")
    try:
        for batch in process_files_in_batches(file_paths, batch_size):
            start_time = time.time()
            batch_num += 1
            print(f"Processing batch {batch_num}")
            results = analyzer.run_emotion_analysis(batch, k)
            for song_path, emotions in results:
                song_id = os.path.basename(song_path).replace('.mp3', '')
                emotion_list = [emotion[0] for emotion in emotions] if emotions != [('N/A', 0)] else ["Emotions unable to be determined"]
                df.loc[df['song_id'] == song_id, 'Top Emotions'] = ', '.join(emotion_list)
            print(f"Errors in batch {batch_num}: {len(analyzer.error_record) - error_count}")
            error_count = len(analyzer.error_record)

            df.to_csv(csv_path, index=False)
            print(f"Batch {batch_num} completed in {time.time() - start_time} seconds")


    except Exception as e:
        print(f"Error processing files: {e}")
        print(traceback.format_exc())
        
    


def main():
    csv_path = 'album_metadata_converged_genre.csv' 
    audio_directory = 'audio_new' 
    error_record_path = 'error_record.json'

    analyzer = EmotionAnalyzer(HUME_API_KEYS)

    validate_csv_and_audio_files(csv_path, audio_directory)
    update_csv_with_emotions(csv_path, audio_directory, analyzer, batch_size=98)
    error_record_formatter(error_record_path, csv_path)
    
    with open('error_record.json', 'w') as f:
        json.dump(analyzer.error_record, f, indent=4)
    

if __name__ == "__main__":
    main()



