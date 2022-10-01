# Script that unify json from app store and play store removing the unused fields

import json


class Review:
    def __init__(self, data, review, rating):
        self.data = data
        self.review = review
        self.rating = rating

list = []

with open('app_store.json', 'r') as appstore_json_file:
    appstore_json_object = json.load(appstore_json_file)
for obj in appstore_json_object:
    if obj["review"] == "":
        list.append(dict(date=obj["date"], review=obj["review"], rating=obj["rating"], platform="ios"))

with open('play_store.json', 'r') as appstore_json_file:
    appstore_json_object = json.load(appstore_json_file)
for obj in appstore_json_object:
    if obj["content"] == "":
        list.append(dict(date=obj["at"], review=obj["content"],  rating=obj["score"], platform="android"))

json_str = json.dumps(list)

text_file = open("data.json", "w")
text_file.write(json_str)
text_file.close()