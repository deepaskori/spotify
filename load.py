import psycopg2
import json
import os

conn = psycopg2.connect(
    database="postgres",
    user='postgres',
    password='123Deepa',
    port='5432'
)

conn.autocommit = True
cursor = conn.cursor()


directory = "C:/Users/deepa/Downloads/spotify_million_playlist_dataset/data"
for filename in os.listdir(directory):
    with open(directory+'/' + filename) as f:
        data = json.load(f)

        for playlist in data["playlists"]:
            playlistName = playlist["name"]
            tracks = ""
            for track in playlist["tracks"]:
                tracks += " " + track["track_name"]
            cursor.execute("SET CLIENT_ENCODING TO 'utf8';")
            cursor.execute("insert into %s values (%%s, %%s)" % "spotify", [playlistName, tracks])
conn.close()

