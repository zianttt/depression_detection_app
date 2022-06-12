from flask import Flask, redirect, url_for, render_template, request
from tweets import get_tweets_by_username
from tweets_model import TweetModel
from questions_model import QuestionModel
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


@app.route('/resultsopt', methods=['POST'])
def results_by_questions():
    q1 = int(request.form['a1'])
    q2 = int(request.form['a2'])
    q3 = int(request.form['a3'])
    q4 = int(request.form['a4'])
    q5 = int(request.form['a5'])
    q6 = int(request.form['a6'])
    q7 = int(request.form['a7'])
    q8 = int(request.form['a8'])
    q9 = int(request.form['a9'])
    sleep_hours = float(request.form['sleep_hours'])
    gender = request.form['sleep_hours']
    if gender.strip() == 'f':
        new_gender = 0
    else:
        new_gender = 1
    height = float(request.form['height'])
    weight = float(request.form['weight'])
    bmi = weight / (np.square(height))

    values = [q1, q2, q3, q4, q5, q6, q7, q8, q9, sleep_hours, new_gender, bmi]
    model = QuestionModel()
    prediction = model.predict(np.array([values]))
    return redirect(url_for('test', result=prediction))


@app.route('/test')
def test():
    result = request.args.get('result', '')
    return render_template('test.html', data=result)


if __name__ == '__main__':
    app.run()
