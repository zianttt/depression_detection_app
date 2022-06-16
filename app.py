from flask import Flask, redirect, url_for, render_template, request
from tweets import get_tweets_by_username
from tweets_model import TweetModel
import numpy as np

app = Flask(__name__)

tweet_prediction_model = TweetModel()


@app.route("/")
def dashboard():
    presiction_results = []
    depressed_percent = 0
    username = request.args.get('username', '')
    tweets_content = []
    if username:
        tweets_content = get_tweets_by_username(
            username, "2021-06-10", "2022-06-18")
        if len(tweets_content) == 0:
            error_msg = [['No records found', 'hello']]
            return render_template('index.html', res=error_msg, pred=depressed_percent)
        presiction_results, depressed_percent = tweet_prediction_model.predict_batch(
            tweets_content)
        return render_template('index.html', res=presiction_results, pred=depressed_percent)
    return render_template('index.html', res=presiction_results, pred=depressed_percent)


@app.route('/results', methods=['POST', 'GET'])
def results_by_username():
    username = request.form['username']
    return redirect(url_for('dashboard', username=username))


@app.route('/test')
def test():
    result = request.args.get('result', '')
    return render_template('test.html', data=result)


if __name__ == '__main__':
    app.run()
