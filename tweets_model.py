from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
import numpy as np
import keras


class TweetModel:

    def __init__(self) -> None:
        self.model = keras.models.load_model('text_model_02.hdf5')
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
