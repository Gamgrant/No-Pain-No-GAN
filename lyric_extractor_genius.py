import requests
from bs4 import BeautifulSoup
from vault_secrets import GENIUS_CLIENT_ID, GENIUS_CLIENT_SECRET, GENIUS_CLIENT_ACCESS_TOKEN

def get_song_lyrics(artist, song_title):
    """
    Fetches song lyrics from the Genius API.
    
    Args:
        artist (str): The name of the artist.
        song_title (str): The title of the song.
    
    Returns:
        str: The lyrics of the song, or None if the lyrics could not be found.
    """
    client_id = GENIUS_CLIENT_ID
    client_secret = GENIUS_CLIENT_SECRET
    
    auth_response = requests.post("https://api.genius.com/oauth/token", data={
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    })
    
    if auth_response.status_code == 200:
        print("Authenticated with Genius API")
        access_token = auth_response.json()["access_token"]
        
        # song search
        search_response = requests.get(
            "https://api.genius.com/search",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"q": f"{artist} {song_title}"}
        )
        
        if search_response.status_code == 200:
            print("Found song on Genius")
            song_info = search_response.json()["response"]["hits"][0]["result"]
            song_url = song_info["url"]
            print(song_info)
            print(f"Song URL: {song_url}")
            print(f"Song title: {song_info['title']}")
            print(f"Song artist: {song_info['primary_artist']['name']}")
            print(f"Song album: {song_info['primary_artist']['album']['name']}")
            print(f"Song genre: {song_info['primary_artist']['genres']}")
            
            
            # fetch lyrics
            lyrics_response = requests.get(song_url)
            if lyrics_response.status_code == 200:
                soup = BeautifulSoup(lyrics_response.text, "html.parser")
                lyrics_div = soup.find("div", class_="lyrics")
                if lyrics_div:
                    lyrics = lyrics_div.get_text(strip=True)
                    return lyrics
        else:
            print("Could not find song on Genius")
    else:
        print("Could not authenticate with Genius API")
    
    return None


lyrics = get_song_lyrics("Michael Jackson", "Thriller")
print(lyrics)
