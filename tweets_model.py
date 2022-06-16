from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
import numpy as np
import keras


class TweetModel:

    def __init__(self) -> None:
        self.model = keras.models.load_model('text_model_bi.hdf5')
        self.classes = ['Not depressed', 'Depressed']
        self.MAXWORD = 5000
        self.MAXLEN = 130
        self.tokenizer = Tokenizer(num_words=self.MAXWORD)

    def predict_class(self, tweet):
        self.tokenizer.fit_on_texts(tweet)
        sequences = self.tokenizer.texts_to_sequences(tweet)
        preprocessed_tweet = pad_sequences(sequences, maxlen=self.MAXLEN)
        pred = self.model.predict(preprocessed_tweet)
        prediction_bin = 0
        if (pred[0][1] >= 0.4):
            prediction_bin = 1
        #prediction_bin = np.around(pred, decimals=0).argmax(axis=1)[0]
        prediction = self.classes[prediction_bin]

        return prediction_bin, prediction

    def predict_batch(self, tweets):
        results = []
        results_bin = []
        for t in tweets:
            prediction_bin, prediction = self.predict_class(t)
            results_bin.append(prediction_bin)
            results.append([t, prediction])
        depressed_percent = (results_bin.count(1) / len(tweets)) * 100
        return results, depressed_percent
