# -*- coding: utf-8 -*-
"""
Created on Mon May 12 20:00:40 2025

@author: sqril
"""

'''Data Gathering and Filtering'''

import pandas as pd
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
# Initialize VADER
sia = SentimentIntensityAnalyzer()
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from wordcloud import WordCloud
from langdetect import detect
from collections import Counter


nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Set of stopwords
stop_words = set(stopwords.words('english'))

def detect_language_safe(text):
    try:
        if pd.isna(text) or str(text).strip() == '':
            return None
        return detect(str(text))
    except:
        return None

def filter_english_accounts(df, tweets_column='tweet', min_english_ratio=0.7):
    filtered_df = df.copy()

    def analyze_user_tweets(tweets):
        # Ensure tweets is a list
        if not isinstance(tweets, list):
            return 0

        # Detect language for each tweet
        languages = [detect_language_safe(tweet) for tweet in tweets if tweet]
        languages = [lang for lang in languages if lang]  # Remove None values

        # Calculate ratio of English tweets
        if not languages:
            return 0
        lang_counts = Counter(languages)
        english_ratio = lang_counts.get('en', 0) / len(languages)

        return english_ratio

    # Calculate English ratio for each user
    print("Analyzing tweets for language detection...")
    filtered_df['english_ratio'] = filtered_df[tweets_column].apply(analyze_user_tweets)

    # Filter users based on minimum English ratio
    english_accounts = filtered_df[filtered_df['english_ratio'] >= min_english_ratio]

    # Drop the temporary column if not needed
    english_accounts = english_accounts.drop('english_ratio', axis=1)

    print(f"Found {len(english_accounts)} accounts with at least {min_english_ratio*100}% English tweets")
    return english_accounts



def filtered_accounts(file_path):
    df = pd.read_json(file_path)
     # Limit the number of tweets per user
    df['tweet'] = df['tweet'].apply(lambda tweets: tweets[:20] if isinstance(tweets, list) else tweets)

    df = filter_english_accounts(df, tweets_column='tweet', min_english_ratio=0.7)
    return df


'''Filtering Accounts'''
test=filtered_accounts('test.json')
dev=filtered_accounts('dev.json')
train=filtered_accounts('train.json')

'''Saving Filtered Accounts to JSON'''
test.to_json('C:/Users/sqril/Twitter English Dataset (Human vs Bot) Classification Final/test_english.json')
dev.to_json('C:/Users/sqril/Twitter English Dataset (Human vs Bot) Classification Final/dev_english.json')
train.to_json('C:/Users/sqril/Twitter English Dataset (Human vs Bot) Classification Final/train_english.json')


def extract_features(df):
    # Extract profile features

    df['num_followers'] = pd.to_numeric(df['profile'].apply(lambda x: x.get('followers_count')), errors='coerce')
    df['num_friends'] = pd.to_numeric(df['profile'].apply(lambda x: x.get('friends_count')), errors='coerce')
    df['is_verified'] = df['profile'].apply(lambda x: x.get('verified')).str.strip().map({'True': 1, 'False': 0})
    df['account_age'] = df['profile'].apply(lambda x: 2020 - pd.to_datetime(x['created_at'].strip(), format="%a %b %d %H:%M:%S %z %Y").year)
    df['tweet_length'] = df['processed_tweet'].apply(len)
    df['statuses_count'] = pd.to_numeric(df['profile'].apply(lambda x: x.get('statuses_count')),errors= 'coerce')
    df['favorites_count'] = pd.to_numeric(df['profile'].apply(lambda x: x.get('favourites_count')),errors= 'coerce')
    df['listed_count'] = pd.to_numeric(df['profile'].apply(lambda x: x.get('listed_count')),errors= 'coerce')
    df['screen_name']=df['profile'].apply(lambda x: x.get('screen_name'))
    df['description']=df['profile'].apply(lambda x: x.get('description'))
    df['friends_follower_ratio'] = df.apply(
            lambda row: row['num_friends'] / row['num_followers']
            if row['num_followers'] > 0 else 0,
            axis=1     )

    df=df.drop(columns=['profile'],axis=1)
    df=df.drop(columns=['ID'],axis=1)


    return df


def process_twitter_data(file_path):
    df = pd.read_json(file_path)
    threshold = 0.8 * len(df)
    df = df.dropna(thresh=threshold, axis=1)
    df=df.drop(columns=['neighbor'],axis=1)

    columns_to_drop = ['id','id_str', 'name', 'location', 'profile_location','id',
                      'entities', 'url', 'utc_offset', 'time_zone',
                      'contributors_enabled', 'is_translator', 'is_translation_enabled',
                        'profile_background_color', 'profile_background_image_url',
                        'utc_offset', 'time_zone','lang','geo_enabled',
                        'profile_background_image_url_https', 'profile_background_tile',
                        'profile_image_url', 'profile_image_url_https','profile_background_image_url_https',
                        'profile_link_color','profile_sidebar_border_color', 'profile_sidebar_fill_color',
                        'profile_text_color', 'profile_use_background_image', 'has_extended_profile',
                      'default_profile', 'default_profile_image']

    df['profile'] = df['profile'].apply(lambda x: {k: v for k, v in x.items() if k not in columns_to_drop})




    # Process tweet text
    if 'tweet' in df.columns:
        tweet_features = df['tweet'].apply(analyze_tweet_content)
        df = pd.concat([df, tweet_features], axis=1)
        df['processed_tweet'] = df['tweet'].apply(lambda tweets: [preprocess_text(tweet) for tweet in tweets] if isinstance(tweets, list) else preprocess_text(tweets))
        
        df = df.drop(columns=['tweet','domain'])
        df = df.drop_duplicates(subset='processed_tweet')

        # Extract features
        df = extract_features(df)
        df = df.dropna(axis=1, how='all')

        return df


def analyze_tweet_content(tweet):
        if isinstance(tweet, list):
            tweet = ' '.join(tweet)

        # Convert to string if not already
        tweet = str(tweet)

        # Count URLs (links)
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        num_links = len(re.findall(url_pattern, tweet))

        # Count hashtags
        num_hashtags = len(re.findall(r'#\w+', tweet))

        # Count mentions
        num_mentions = len(re.findall(r'@\w+', tweet))

        # Check if it's a retweet
        num_retweets=len(re.findall(r'RT',tweet))

        # Count characters (excluding whitespace)
        num_chars = len(''.join(tweet.split()))

        return pd.Series({
            'num_links': num_links,
            'num_hashtags': num_hashtags,
            'num_mentions': num_mentions,
           # 'is_retweet': is_retweet,
            'num_chars': num_chars
        })


def preprocess_text(text):
    if text is None or not text:
        return '' # Return an empty string instead of an empty list

    if isinstance(text, list):
        text = ' '.join(text)

    text = re.sub(r'^RT @\w+: ', '', text)
    text = text.lower()

    # Remove links
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\bamp\b', '', text)
    text = re.sub(r'\bnot\b', 'not_', text)
    text = re.sub(r'\bno\b', 'no_', text)
    text = re.sub(r'\bnever\b', 'never_', text)
    text = re.sub(r'\bcannot\b', 'cannot_', text)
    text = re.sub(r'\bcan\'t\b', 'cannot_', text)
    text=re.sub(r'\bu\b', 'you', text)
    text=re.sub(r'\bim\b', 'i am', text)
    # Remove special characters (keep letters, numbers, and spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Remove stop words while maintaining sentence structure
    stop_words = set(nltk.corpus.stopwords.words('english'))
    important_words = {"not", "no", "never", "nor", "but",'do'}
    custom_stop_words = stop_words - important_words
    custom_stop_words.update({'one','well','new','w','v','day','im'})

    words = text.split()
    cleaned_words = [word for word in words if word not in custom_stop_words]

    # Join words to form cleaned text
    cleaned_text = ' '.join(cleaned_words)

    return cleaned_text



test_processed = process_twitter_data('test_english.json')
test_processed.head()

train_processed = process_twitter_data('train_english.json')
train_processed.head()

dev_processed = process_twitter_data('dev.json')
dev_processed.head()

all_words = ' '.join([' '.join(tweet) for tweet in train_processed['processed_tweet']])

'''Data Visualization - Word Clouds'''
# Create a WordCloud object
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)

# Plot the Word Cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # No axes for word cloud
plt.title('Word Cloud of Processed Tweets', fontsize=20)
plt.show()


df = pd.read_json('train.json')
# Convert each tweet to a string before joining.
# This ensures that even if a tweet contains other data types, it's treated as a string.
all_word = ' '.join([str(tweet) for tweet in df['tweet']])

# Create a WordCloud object
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_word)

# Plot the Word Cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # No axes for word cloud
plt.title('Word Cloud of Original Tweets', fontsize=20)
plt.show()


train_processed=pd.concat([train_processed,test_processed], ignore_index=True)
train_processed.to_csv('C:/Users/sqril/Twitter English Dataset (Human vs Bot) Classification Final/twitter_english_data.csv', index=False)
train_processed.head()

train_processed['label'].value_counts()


'''Performing Sentiment Analysis and scaling features down'''
# Load the CSV file
df = pd.read_csv('twitter_english_data.csv')

columns_to_scale = ['num_chars', 'num_mentions', 'num_hashtags', 'num_links', 'account_age', 'favorites_count', 'friends_follower_ratio', 'listed_count', 'num_followers', 'num_friends', 'tweet_length', 'statuses_count']


#Use robustscaler first to scale features down to prevent outliers from affecting it
robust_scaler = RobustScaler()
df[columns_to_scale] = robust_scaler.fit_transform(df[columns_to_scale])

# Apply MinMaxScaler to bring the range to [0, 1]
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
df[columns_to_scale] = min_max_scaler.fit_transform(df[columns_to_scale])

df.to_csv('C:/Users/sqril/Twitter English Dataset (Human vs Bot) Classification Final/english_data.csv', index=False)
df.head()


def analyze_sentiment(text):
    text = str(text) if not pd.isna(text) else ''
    scores = sia.polarity_scores(text)
    return scores

def perform_sentiment_analysis(df, text_column='processed_tweet'):
    df['sentiment_scores'] = df[text_column].apply(lambda tweets: analyze_sentiment(" ".join(tweets)) if isinstance(tweets, list) else analyze_sentiment(tweets))
    df['compound_sentiment'] = df['sentiment_scores'].apply(lambda x: x['compound'])
    df['sentiment_category'] = df['compound_sentiment'].apply(lambda score:
        'Positive' if score >= 0.05 else
        'Negative' if score <= -0.05 else
        'Neutral')
    return df
    
df = perform_sentiment_analysis(df)
df.isnull().sum()
df.columns
















