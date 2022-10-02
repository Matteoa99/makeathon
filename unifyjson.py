# Script that unify json from app store and play store removing the unused fields

import json


class Review:
    def __init__(self,id, data, review, rating, platform):
        self.id = id
        self.data = data
        self.review = review
        self.rating = rating
        self.platform = platform

list = []

with open('app_store.json', 'r') as appstore_json_file:
    appstore_json_object = json.load(appstore_json_file)
for obj in appstore_json_object:
    list.append(dict(
                     id=obj['userName'],
                     date=obj["date"],
                     review=obj["review"],
                     rating=obj["rating"],
                     platform="ios"))

with open('play_store.json', 'r') as appstore_json_file:
    appstore_json_object = json.load(appstore_json_file)
for obj in appstore_json_object:
    list.append(dict(
                     id=obj['reviewId'],
                     date=obj["at"],
                     review=obj["content"],
                     rating=obj["score"],
                     platform="android"))

json_str = json.dumps(list)

text_file = open("unified_review.json", "w")
text_file.write(json_str)
text_file.close()