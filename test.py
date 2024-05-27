import base64
import requests
import urllib.parse
from urllib.parse import urlparse, parse_qs
import pandas as pd
from transformers import pipeline


CLIENT_ID = 'b482975f9ea341a3a6d8e6b4d23082fc'
CLIENT_SECRET = 'cd3753d204dc400aa159ce1b6673c977'
data = {
    'grant_type': 'client_credentials',
    'client_id': CLIENT_ID,
    'client_secret': CLIENT_SECRET,
}

response = requests.post('https://accounts.spotify.com/api/token', data=data)
# print("temp_token:", response.json()['access_token'])

temp_token= "BQD-WwjX4y1g1YkExVb5og-4WrFYJLEkdsEycFQl3ScxKsR-9NcHgsJClNZvxmwqGMlqpQURYUzugkr9DyQjtShZfh1QWSE5xKV-6b-wkWgec5WheDE"
headers = {
    'Authorization': f'Bearer {temp_token}',
}
response = requests.get('https://api.spotify.com/v1/artists/4Z8W4fKeB5YxbusRsdQVPb', headers=headers)
# print("test response:", response.json())


REDIRECT_URI = 'http://localhost:8888/callback'
authorization_url = 'https://accounts.spotify.com/authorize'

query_params = {
    'response_type': 'code',
    'client_id': CLIENT_ID,
    'scope': 'playlist-read-private playlist-read-collaborative',
    'redirect_uri': REDIRECT_URI
}

authorization_redirect_url = authorization_url + '?' + urllib.parse.urlencode(query_params)
print(authorization_redirect_url)


authorization_code = "AQB9ktNTCoUVcOunrnzci1RPlIcmILj_VxW1L_v0gBQKqxZpd7PLeDuQd2ROzZzlOa1poUVphLkRsZ0E_7M60o_tArNSGHjLCFhX1S5Q5jVGsioR23BuJJFQfFCaXtnRESw52MhPi-VviGdybYTcWytj-x307t9R-mcfgHvfk262-NtISFV6eecoJ5kxS-Xj_KQ0YI2Nxaiv9Q1S466MN-6FB0TrJhdZVBmjJw1IPFAGdJ1KPWc"
token_url = 'https://accounts.spotify.com/api/token'
auth_header = f"{CLIENT_ID}:{CLIENT_SECRET}".encode('ascii')
auth_header_b64 = base64.b64encode(auth_header).decode('ascii')

token_data = {
    'grant_type': 'authorization_code',
    'code': authorization_code,
    'redirect_uri': REDIRECT_URI
}

token_headers = {
    'Authorization': f'Basic {auth_header_b64}',
    'Content-Type': 'application/x-www-form-urlencoded'
}

response = requests.post(token_url, data=token_data, headers=token_headers)
token_info = response.json()

print("access_token dictionary:", token_info)

access_token = "BQBn0jijDjEhnZt0xjLRoR5lFMypllrvYIrPhaHLxxJzqH4Kc4Y-omQBZM2Zv_kjjDX0hSxo_8SSuVaHsPH5o-nbiRawFiJ3sl7eJdiPvJ1OlztQJnAF3Avs8C4k9Y1iDRacCBUgdiQiZq4GSLN_1YA8FWL7FXwLLMm1iEwR6Dzv_r_WEZh-PeyoDN_S_plYew"

me_url = "https://api.spotify.com/v1/users/deepakori0/playlists"
headers = {
        'Authorization': f'Bearer {access_token}'
    }

response = requests.get(me_url, headers=headers)
data = response.json()
personalDf = pd.DataFrame(columns = ["playlist", "tracks"])
for item in data["items"]:
    dct = {}
    playlistName = item["name"]
    response = requests.get(item["tracks"]['href'], headers=headers)
    tracks = ""
    for potTrack in response.json()["items"]:
        if potTrack["track"]:
            tracks += potTrack["track"]["name"]
    dct['playlist'] = playlistName
    dct['tracks'] = tracks
    personalDf = personalDf._append(dct, ignore_index=True)

print(personalDf)



print("*************************")

pipe = pipeline("text2text-generation", model="deepakori/finetuned-spotify-t5")
print("lady gaga and conan mockasin", pipe(personalDf.iloc[1, 1]))
print("alex g and panchiko", pipe(personalDf.iloc[2, 1]))
print("indian playlist", pipe(personalDf.iloc[14, 1]))
print("taylor swift playlist", pipe(personalDf.iloc[19, 1]))
print("punk playlist", pipe(personalDf.iloc[26, 1]))
print("jazz playlist", pipe(personalDf.iloc[47, 1]))
