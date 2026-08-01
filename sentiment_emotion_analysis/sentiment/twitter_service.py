import time
import tweepy
from datetime import datetime

class TwitterServiceException(Exception):
    """Base exception for Twitter Service Errors."""
    pass

class TwitterCredentialsError(TwitterServiceException):
    """Raised when valid credentials are not provided."""
    pass

class TwitterRateLimitError(TwitterServiceException):
    """Raised when Twitter API rate limits are hit."""
    pass

class TwitterAPIError(TwitterServiceException):
    """Raised when general Tweepy/Twitter API errors occur."""
    pass

class LiveTweet:
    """Standardized representation of a Live Tweet containing full metadata."""
    def __init__(self, username, text, created_at, likes, retweets, replies, lang, verified):
        self.username = username
        self.text = text
        self.created_at = created_at
        self.likes = likes
        self.retweets = retweets
        self.replies = replies
        self.lang = lang
        self.verified = verified

class TwitterService:
    # Class-level cache storing (timestamp, data) keyed by (query, query_type)
    _cache = {}
    CACHE_DURATION = 300 # 5 minutes TTL

    # Placeholder keys (same as default tweepy_sentiment.py)
    consumer_key = "user consumer_key"
    consumer_secret = "user consumer_secret"
    access_token = "user access_token"
    access_token_secret = "user access_token_secret"

    def _get_api(self):
        """Validate credentials and initialize Tweepy API client."""
        # Strict validation check
        if any(x in self.consumer_key or x in self.access_token for x in ["user ", "placeholder", "key_here"]):
            raise TwitterCredentialsError(
                "Official Twitter (X) API credentials are not configured. "
                "Please replace the placeholder values with valid API keys inside the service layer to pull live tweets."
            )
        try:
            auth = tweepy.OAuthHandler(self.consumer_key, self.consumer_secret)
            auth.set_access_token(self.access_token, self.access_token_secret)
            return tweepy.API(auth, wait_on_rate_limit=False)
        except Exception as e:
            raise TwitterCredentialsError(f"Failed to authenticate with Twitter OAuth: {e}")

    def _check_cache(self, key):
        """Retrieve data from cache if it exists and has not expired."""
        if key in self._cache:
            timestamp, data = self._cache[key]
            if time.time() - timestamp < self.CACHE_DURATION:
                return data
        return None

    def _set_cache(self, key, data):
        """Store data in cache with current timestamp."""
        self._cache[key] = (time.time(), data)

    def fetch_tweets_by_user(self, username, count=20):
        """Fetch live timeline tweets for a specific user handle."""
        cache_key = (username, 'user_timeline')
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        api = self._get_api()
        try:
            raw_tweets = api.user_timeline(screen_name=username, count=count, tweet_mode='extended')
            parsed = self._parse_tweets(raw_tweets)
            self._set_cache(cache_key, parsed)
            return parsed
        except tweepy.errors.TooManyRequests:
            raise TwitterRateLimitError("Twitter API rate limit exceeded. Please wait a few minutes before trying again.")
        except Exception as e:
            raise TwitterAPIError(f"Failed to fetch timeline for @{username}: {e}")

    def fetch_tweets_by_query(self, query, count=20):
        """Search and fetch live tweets matching a query or hashtag."""
        cache_key = (query, 'search')
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        api = self._get_api()
        try:
            raw_tweets = api.search_tweets(q=query, count=count, lang='en', tweet_mode='extended')
            parsed = self._parse_tweets(raw_tweets)
            self._set_cache(cache_key, parsed)
            return parsed
        except tweepy.errors.TooManyRequests:
            raise TwitterRateLimitError("Twitter API rate limit exceeded. Please wait a few minutes before trying again.")
        except Exception as e:
            raise TwitterAPIError(f"Failed to search query '{query}': {e}")

    def fetch_trends(self):
        """Fetch global trending hashtags or fallback to standard topics."""
        cache_key = ('global_trends', 'trends')
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        try:
            api = self._get_api()
            trends = api.get_place_trends(1)
            parsed_trends = [t['name'] for t in trends[0]['trends']][:10]
            self._set_cache(cache_key, parsed_trends)
            return parsed_trends
        except Exception:
            # Fallback to standard top tech trending hashtags if API tier does not support trend retrieval
            tech_trends = ["#Python", "#ArtificialIntelligence", "#MachineLearning", "#DataScience", "#ChatGPT", "#LLM"]
            self._set_cache(cache_key, tech_trends)
            return tech_trends

    def _parse_tweets(self, raw_tweets):
        """Convert Tweepy Status objects to LiveTweet objects with verified fields."""
        parsed = []
        for tweet in raw_tweets:
            # Resolve full text depending on extended mode layout
            text = tweet.full_text if hasattr(tweet, 'full_text') else tweet.text
            
            # Tweepy creation dates are datetime objects
            created_str = tweet.created_at.strftime('%Y-%m-%d %H:%M:%S')
            
            # Safely check for reply counts (if available in raw response)
            replies = getattr(tweet, 'reply_count', 0)
            
            live_tweet = LiveTweet(
                username=tweet.user.screen_name,
                text=text,
                created_at=created_str,
                likes=tweet.favorite_count,
                retweets=tweet.retweet_count,
                replies=replies,
                lang=tweet.lang,
                verified=tweet.user.verified
            )
            parsed.append(live_tweet)
        return parsed
