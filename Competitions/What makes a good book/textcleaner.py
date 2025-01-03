import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from collections import Counter
import maps as m
import textcleaner as tc
import re

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("opinion_lexicon")


def clean_text_simple(text):

    """
    Function converts all text to lower and removes non-alpha characters
    """

    if type(text) != str:
        text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numeric characters
    text = re.sub(r'\d+', '', text)

    # Tokenize text
    tokens = word_tokenize(text)

    tokens = [t for t in tokens if t not in string.punctuation]

    # Remove the tokens smaller than 3
    tokens = [t for t in tokens if len(t) > 2]

    # Join tokens back into a single string for easier analysis later
    cleaned_text = ' '.join(tokens)

    return cleaned_text

def clean_text(text):

    if type(text) != str:
        text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numeric characters
    text = re.sub(r'\d+', '', text)

    # Tokenize text
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Remove the tokens smaller than 3
    tokens = [t for t in tokens if len(t) > 2]

    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word.lower(), pos = 'v') for word in tokens]
    tokens = [t for t in tokens if t not in string.punctuation]

    # Join tokens back into a single string for easier analysis later
    cleaned_text = ' '.join(tokens)

    return cleaned_text


def drop_nulls(df):
    # drop nulls
    df = df[~df['review/summary'].isnull()].reset_index(drop = True)
    print("dropped nulls")
    return df


def add_index(df):
    df['book_id'] = df.index
    print("created id column")
    return df


def clean_title(df):
    df['title'] = df['title'].apply(clean_text_simple)
    print("cleaned title text")
    return df


def create_title_fk(df):
    df['title_fk'] = df['title'].apply(clean_text)
    return df


def clean_review_helpfulness(df):
    df['reviews_helpful'] = df['review/helpfulness'].apply(lambda x: x.split("/")[0]).astype(float)
    df['total_reviews']  = df['review/helpfulness'].apply(lambda x: x.split("/")[1]).astype(float)
    df['reviews_not_helpful'] = df['total_reviews'] - df['reviews_helpful']
    df['pct_reviews_helpful'] = df['reviews_helpful'] / df['total_reviews']
    df['pct_reviews_helpful'] = df['pct_reviews_helpful'].fillna(0)
    #print("parsed review/helpfulness")
    return df


def clean_review_summary(df):
    df['review_summary'] = df['review/summary'].apply(clean_text_simple)
    print("cleaned review/summary")
    return df


def clean_review_text(df):
    df['review_text'] = df['review/text'].apply(clean_text_simple)
    df['review_sentiment'] = df['review_text'].apply(get_sentiment)
    df['review_sentiment_type'] = df['review_sentiment'].apply(get_sentiment_category)
    df['review_text'] = df['review_text'].apply(clean_text)
    #print("cleaned review/text, added review sentiment")
    df = df.drop('review/text', axis = 1)
    return df

    
def clean_review_summary_text(df):
    df['review_summary'] = df['review/summary'].apply(clean_text_simple)
    df['review_summary_sentiment'] = df['review_summary'].apply(get_sentiment)
    df['review_summary_sentiment_type'] = df['review_summary_sentiment'].apply(get_sentiment_category)
    df['review_summary'] = df['review_summary'].apply(clean_text)
    #print("cleaned review/text, added review sentiment")
    df = df.drop('review/summary', axis = 1)
    return df


def clean_description_text(df):
    df['description'] = df['description'].apply(clean_text_simple)
    df['description_sentiment'] = df['description'].apply(get_sentiment)
    df['desc_sentiment_type'] = df['description_sentiment'].apply(get_sentiment_category)
    df['description'] = df['description'].apply(clean_text)
    #print("cleaned description, added description sentiment")
    return df


def clean_categories(df, map_genres):
    # Categories
    df['categories'] = df['categories'].str.replace("'", "").apply(lambda x: x.lower())
    df['categories'] = df['categories'].str.replace('"',"")
    df['remap'] = df['categories'].map(map_genres)
    print(f"{df[df['remap'].isnull()]['categories'].unique().shape[0]} remapped genres missing")

    df['genre'] = df['remap'].apply(lambda x: x[0])
    df['genre'] = df['genre'].apply(lambda x: x.lower())

    df['sub_genre'] = df['remap'].apply(lambda x: x[1])
    df['sub_genre'] = df['sub_genre'].apply(lambda x: x.lower() if x != None else None)
    df = df.drop(['remap', 'categories'], axis = 1).reset_index(drop = True)
    print("remapped categories")
    return df


def clean_popularity(df):
    # popularity
    df['popularity'] = df['popularity'].map(m.map_popular)
    return df


def get_sentiment(text):
    if type(text) != str:
        str(text)

    tb = TextBlob(text)
    return tb.sentiment[0]


def get_sentiment_category(polarity):
    
    if polarity < -0.2:
        return 'Negative'
    elif polarity < 0.2:
        return 'Neutral'
    else:
        return 'Positive'


# Function to remove content between parentheses
def remove_parentheses_content(text):
    return re.sub(r'\([^)]*\)', '', text)


def clean_title(df):
    df['title'] = df['title'].apply(remove_parentheses_content)
    df['title'] = df['title'].str.strip()
    df['book_title_length'] = df['title'].apply(len)
    return df
