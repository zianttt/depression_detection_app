from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
import contractions
import emoji
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin
import keras
import string
import nltk
from nltk import download, ngrams
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
# download('stopwords')
# download('wordnet')
# download('punkt')
# download('averaged_perceptron_tagger')
STOPWORDS = stopwords.words("english") + ['rt']


class TweetModel:

    def __init__(self) -> None:
        self.model = keras.models.load_model('text_model.hdf5')
        self.ct = CleanText()
        self.classes = ['Not depressed', 'Depressed']
        self.MAXWORD = 5000
        self.MAXLEN = 130
        self.tokenizer = Tokenizer(num_words=self.MAXWORD)

    def predict(self, preprocessed_tweet):
        self.tokenizer.fit_on_texts(preprocessed_tweet)
        sequences = self.tokenizer.texts_to_sequences(preprocessed_tweet)
        preprocessed_tweet = pad_sequences(sequences, maxlen=self.MAXLEN)
        prediction = self.classes[np.around(self.model.predict(
            preprocessed_tweet), decimals=0).argmax(axis=1)[0]]

        return prediction

    def predict_batch(self, tweets):
        results = []
        for tweet in tweets:
            results.append(self.predict(tweet))
        return results

    def preprocess_data(self, tweet):
        tweet = self.ct.fit(tweet).transform(tweet)


class CleanText(BaseEstimator, TransformerMixin):

    # DIGITS
    def remove_digits(self, input_text):
        """Removes any digits from the text."""
        return re.sub(r'\d+', ' ', input_text)

    # MENTIONS
    def remove_mentions(self, input_text):
        """Removes mentions from the text by finding them through @"""
        return re.sub(r'@\w+', ' ', input_text)

    # PUNCTUATION
    def remove_punctuation(self, input_text):
        """Replaces any punctuation symbol by a space."""
        punct = string.punctuation + '…“”￼＆—●'
        trantab = str.maketrans(punct, len(punct)*' ')
        return input_text.translate(trantab)

    # STOPWORDS
    def remove_stopwords(self, input_text):
        """Removes english stopwords"""
        words = input_text.split()
        clean_words = [word for word in words if word not in STOPWORDS]
        return " ".join(clean_words)

    # URLS
    def remove_urls(self, input_text):
        """Removes external links from the text by finding them through http(s)"""
        return re.sub(r'http.?://[^\s]+[\s]?', ' ', input_text)

    # STEMMING
    def stemming(self, input_text):
        """ Returns the input text stemmed using Snowball algorithms."""
        stemmer = SnowballStemmer(language="english")
        words = input_text.split()
        stemmed_words = [stemmer.stem(word) for word in words]
        return " ".join(stemmed_words)

    # LEMMATIZATION
    def lemmatize(self, input_text):
        """ Returns the input text lemmatized."""
        lemmatizer = WordNetLemmatizer()
        words = input_text.split()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        return " ".join(lemmatized_words)

    # EMOJI
    def tranform_emoji(self, input_text):
        """ Transforms emoji in the corresponding word."""
        clean_text = emoji.demojize(input_text, delimiters=("_", "_"))
        return re.sub(r'_', ' ', clean_text)

    # LOWER
    def to_lower(self, input_text):
        return input_text.lower()

    # APOSTROPHE
    def apostrophe(self, input_text):
        return re.sub(r'‘', "'", input_text)

    # FORMAT
    def remove_format(self, input_text):
        return format(input_text)

    # CONTRACTIONS - NO VADER
    def expand_contractions(self, input_text):
        return contractions.fix(input_text)

    def expand_contractions_vader(self, input_text):
        """
        Since VADER works well with slang, if mode == 3 (VADER)
        we do not transform the slang in the text (slang = False)
        """
        return contractions.fix(input_text, slang=False)

    # STRIP TEXT
    def strip_text(self, input_text):
        return input_text.strip()

    # HASTAGS
    def hashtags(self, input_text):
        return input_text.replace("#", " ")

    # TRAILING
    def trailing(self, input_text):
        return input_text.replace('\n', " ").replace("\r", " ")

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        clean_X = X.apply(self.remove_format)\
            .apply(self.to_lower)\
            .apply(self.remove_mentions)\
            .apply(self.remove_urls)\
            .apply(self.tranform_emoji)\
            .apply(self.remove_digits)\
            .apply(self.remove_punctuation)\
            .apply(self.apostrophe)\
            .apply(self.expand_contractions)\
            .apply(self.stemming)\
            .apply(self.remove_stopwords)

        return clean_X
