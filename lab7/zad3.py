from twikit import Client
import json

client = Client("en-US")

with open("./pass/data.json", "r") as file:
    user_info = json.load(file)

client.login(auth_info_1=user_info["username"], password=user_info["password"])

client.save_cookies("cookies.json")

client.load_cookies(path="cookies.json")

user = client.get_user_by_screen_name("elonmusk")

tweets = user.get_tweets("Tweets", count=100)

tweets_to_store = []
for tweet in tweets:
    tweets_to_store.append(
        {
            "created_at": tweet.created_at,
            "favorite_count": tweet.favorite_count,
            "full_text": tweet.full_text,
        }
    )

with open("tweets.json", "w", encoding="utf-8") as json_file:
    json.dump(tweets_to_store, json_file, ensure_ascii=False, indent=4)
