import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import googleapiclient.discovery
from vault_secrets import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, YOUTUBE_API_KEY

# Spotify API setup
client_id = SPOTIFY_CLIENT_ID
client_secret = SPOTIFY_CLIENT_SECRET
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# YouTube API setup
youtube_api_key = YOUTUBE_API_KEY
youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey=youtube_api_key)

def get_album_uri(album_name, artist_name):
    results = sp.search(q=f'album:{album_name} artist:{artist_name}', type='album')
    if results['albums']['items']:
        album = results['albums']['items'][0]
        return album['uri']
    else:
        return None

def get_album_tracks(album_uri):
    album = sp.album(album_uri)
    return album['tracks']['items']

def get_youtube_video_views(song_title, artist_name):
    search_query = f"{song_title} {artist_name} official song"
    search_response = youtube.search().list(
        q=search_query,
        type='video',
        part='id,snippet',
        maxResults=1
    ).execute()

    if search_response['items']:
        video_id = search_response['items'][0]['id']['videoId']
        video_response = youtube.videos().list(
            part='statistics',
            id=video_id
        ).execute()

        if 'viewCount' in video_response['items'][0]['statistics']:
            return int(video_response['items'][0]['statistics']['viewCount'])
        else:
            return 0
    else:
        return 0

# Main script
album_name = 'Thriller'
artist_name = 'Michael Jackson'

popularity_record = []

# Get the album URI
album_uri = get_album_uri(album_name, artist_name)
if album_uri:
    # Get the album tracks
    tracks = get_album_tracks(album_uri)

    # Get the popularity (YouTube views) for each track
    for track in tracks:
        song_title = track['name']
        youtube_views = get_youtube_video_views(song_title, artist_name)
        if "popularity" in track:
            spotify_popularity = track["popularity"]
        else:
            spotify_popularity = 0
        popularity_record.append((song_title, spotify_popularity, youtube_views))

    # Print the popularity record
    print(f"Album '{album_name}' by '{artist_name}' has the following popularity records:")
    for song_title, spotify_popularity, youtube_views in popularity_record:
        print(f"- {song_title}: {youtube_views} views, Spotify popularity: {spotify_popularity}")

    # Print the most popular track
    most_popular_track = max(popularity_record, key=lambda x: x[1])
    print(f"The most popular track based on Spotify popularity is '{most_popular_track[0]}' with {most_popular_track[1]} popularity.")
    most_popular_track = max(popularity_record, key=lambda x: x[2])
    print(f"The most popular track based on YouTube views is '{most_popular_track[0]}' with {most_popular_track[2]} views.")
else:
    print(f"Unable to find the album '{album_name}' by '{artist_name}'")
