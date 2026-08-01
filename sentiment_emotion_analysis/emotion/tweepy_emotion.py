from tweepy import OAuthHandler
from tweepy import API
from tweepy import Cursor
from datetime import datetime, date, time, timedelta
from collections import Counter
import sys
import tweepy
import numpy as np
import pandas as pd

class Import_tweet_emotion:

	consumer_key="user consumer_key"
	consumer_secret="user consumer_secret"
	access_token="user access_token"
	access_token_secret="user access_token_secret"

	def tweet_to_data_frame(self, tweets):
		df = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])
		return df

	def get_tweets(self, handle):
		if "user " in self.consumer_key or "user " in self.access_token:
			return self.generate_mock_tweets(handle)
		try:
			auth = OAuthHandler(self.consumer_key, self.consumer_secret)
			auth.set_access_token(self.access_token, self.access_token_secret)
			auth_api = API(auth)

			account = handle
			item = auth_api.user_timeline(id=account,count=20)
			df = self.tweet_to_data_frame(item)

			all_tweets = []
			for j in range(20):
				all_tweets.append(df.loc[j]['Tweets'])
			return all_tweets
		except Exception as e:
			print(f"Twitter API failed: {e}. Falling back to mock tweets.")
			return self.generate_mock_tweets(handle)

	def get_hashtag(self, hashtag):
		if "user " in self.consumer_key or "user " in self.access_token:
			return self.generate_mock_tweets(hashtag)
		try:
			auth = OAuthHandler(self.consumer_key, self.consumer_secret)
			auth.set_access_token(self.access_token, self.access_token_secret)
			auth_api = API(auth)

			account = hashtag
			all_tweets = []

			for tweet in tweepy.Cursor(auth_api.search, q=account, lang='en').items(20):
				all_tweets.append(tweet.text)

			return all_tweets
		except Exception as e:
			print(f"Twitter API failed: {e}. Falling back to mock tweets.")
			return self.generate_mock_tweets(hashtag)

	def generate_mock_tweets(self, query):
		import random
		templates = [
			"Just spent some time looking into {query}, and I'm honestly blown away by the potential! #innovation",
			"Had a really frustrating experience with {query} today. Hopefully, they fix these bugs soon. #disappointed",
			"Can we talk about how amazing {query} is? Highly recommend everyone to check it out!",
			"Not sure if {query} is worth all the hype it's getting lately. Seems a bit overrated to me.",
			"Absolutely loving the new updates for {query}. A solid step in the right direction! Kudos to the team.",
			"Is anyone else experiencing issues with {query} at the moment? The app keeps crashing for me.",
			"Honestly, {query} is just average. It gets the job done but there's definitely room for improvement.",
			"So glad I started using {query}. It has saved me hours of manual work this week! #productivity",
			"Extremely disappointed with {query}'s customer service. They completely ignored my support ticket. #badservice",
			"Just read a fascinating article about the future of {query}. Lots of exciting things on the horizon!",
			"Using {query} has been a total game-changer for my workflow. I don't know how I managed without it.",
			"Warning: Avoid {query} if you care about your privacy. Their new terms of service are very sketchy.",
			"Had a great experience using {query} today. Simple, fast, and did exactly what I needed it to do.",
			"I've been trying to learn more about {query} recently. Any recommendations for good tutorials or documentation?",
			"Shoutout to the developers of {query}! This is one of the best tools I've used in a long time.",
			"{query} is okay, but I feel like there are better alternatives out there for the same price.",
			"It's about time {query} got the recognition it deserves. It has been a solid product for years.",
			"So tired of hearing about {query} everywhere. Can we move on to something else already?",
			"Beautiful design, smooth interface, and extremely reliable - {query} ticks all the boxes for me!",
			"I have mixed feelings about {query}. On one hand, it's very powerful, but on the other, the learning curve is steep."
		]
		tweets = [t.format(query=query) for t in templates]
		random.shuffle(tweets)
		return tweets[:20]
