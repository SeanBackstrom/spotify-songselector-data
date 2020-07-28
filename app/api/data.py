import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
SPOTIPY_CLIENT_ID = os.getenv('SPOTIPY_CLIENT_ID')
SPOTIPY_CLIENT_SECRET = os.getenv('SPOTIPY_CLIENT_SECRET')

# use this variable to do commands
sp = spotipy.Spotify(
    client_credentials_manager=SpotifyClientCredentials())

# new_releaselist = sp.new_releases(country='US', limit=1,)

# print(sp.track('SOBBMDR12A8C13253B'))

songlist = []
offset = 0 
for songset in range(0, 3):     
    songs = sp.playlist_tracks(playlist_id='3YjUPovYVgaUHPTw2QUpMg',
                               offset=offset)
    offset += 99
    songlist.append(songs)

print(len(songlist[0]))
