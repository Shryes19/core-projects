# from bs4 import BeautifulSoup
# import requests
#
# songs = []
# client_id = "c43b35bdbe94495a85d432a0b2ef4ea7"
# client_secret = "525c587a955546b6ac958b0aebfe8635"
#
# date = input("Enter date in yyyy-mm-dd format\n")
#
# responce = requests.get(f"https://www.billboard.com/charts/hot-100/{date}")
# responce.raise_for_status()
# text = responce.text
#
#
# soup = BeautifulSoup(text,"html.parser")
# all_movies = soup.select("li ul li h3")
# for movie in all_movies:
#     songs.append(movie.getText().strip())
# print(songs)

client_id = "c43b35bdbe94495a85d432a0b2ef4ea7"
client_secret = "525c587a955546b6ac958b0aebfe8635"

import spotipy
from spotipy.oauth2 import SpotifyOAuth

scope =  "playlist-modify-private"

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope,
                                               client_id=client_id,
                                               client_secret=client_secret,
                                               redirect_uri="http://example.com",
                                               username = "31avcfdqgmk2lmzuzby34l3e5n6y",
                                               show_dialog=True,
                                               cache_path="token.txt",
                                               ))
user_id = sp.current_user()["id"]

