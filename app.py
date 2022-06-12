from flask import Flask, redirect, url_for, render_template, request
from tweets import get_tweets_by_username
from tweets_model import TweetModel
import numpy as np

app = Flask(__name__)

tweet_prediction_model = TweetModel()


@app.route("/")
def dashboard():
    default = []
    username = request.args.get('username', '')
    opts = request.args.get('opts', [])
    tweets_content = []
    presiction_results = []
    if username:
        tweets_content = get_tweets_by_username(username)
        if len(tweets_content) == 0:
            error_msg = ['Username not found']
            return render_template('index.html', data=error_msg)
        presiction_results = tweet_prediction_model.predict_batch(
            tweets_content)
        return render_template('index.html', data=presiction_results)
    if opts:
        return render_template('index.html', data=opts)
    return render_template('index.html', data=default)


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
