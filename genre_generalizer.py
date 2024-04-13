genres = ['Classical', 'Rock','Alternative', 'Pop', 'Jazz', 'Blues', 'Country', 'Electronic', 'Folk', 'Hip-Hop', 'R&B', 'Reggae']

import pandas as pd

# Load the CSV file
file_path = 'genres_merged.csv'  # Change this to the path of your CSV file
df = pd.read_csv(file_path)

# Keep only "genre" and "single_genre" columns
df = df[['genre', 'single_genre']]

# Count the total number of rows before deleting "N/A"
total_rows_before = df.shape[0]
print(f"Total rows before deletion: {total_rows_before}")

# Delete all rows where "single_genre" is "N/A"
df_filtered = df[df['single_genre'] != "N/A"]

# Count the total number of rows after deleting "N/A"
total_rows_after = df_filtered.shape[0]
print(f"Total rows after deletion: {total_rows_after}")

# Optionally, save the filtered dataframe to a new CSV
filtered_file_path = 'genres_merged_and_cleaned.csv'  # Change this to your desired path
df_filtered.to_csv(filtered_file_path, index=False)
