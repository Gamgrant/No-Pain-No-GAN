import pandas as pd
import json

def error_record_formatter(json_file_path, csv_file_path):
    df = pd.read_csv(csv_file_path)
    
    # Load the JSON data
    with open(json_file_path, 'r') as file:
        json_data = json.load(file)

    # Update the JSON data with information from the CSV based on song_id
    for song_id, errors in json_data.items():
        if errors: 
            song_data = df[df['song_id'] == song_id].iloc[0]
            error_message = errors[0]['message']  

            # Update the JSON entry with additional data from the CSV
            json_data[song_id][0].update({
                "song_name": song_data['song_name'],
                "album_name": song_data['album_name'],
                "artist_name": song_data['artist'],
                "genre": song_data['singular_genre'],
                "error_message": error_message  
            })

    # Save the modified JSON back to the same file
    with open(json_file_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

def main():
    json_file_path = 'error_record.json' 
    csv_file_path = 'album_metadata_converged_genre_small_test.csv'  

    error_record_formatter(json_file_path, csv_file_path)

if __name__ == "__main__":
    main()

