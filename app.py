from flask import Flask, redirect, url_for, render_template, request
from tweets import get_tweets_by_username
from tweets_model import TweetModel
from questions_model import QuestionModel

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

    values = [q1, q2, q3, q4, q5, q6, q7, q8, q9]
    model = QuestionModel()
    classifier = model.svm_classifier()
    prediction = classifier.predict([values])
    result = 'Default'
    if prediction[0] == 0:
        result = 'Your Depression test result : No Depression'
    if prediction[0] == 1:
        result = 'Your Depression test result : Mild Depression'
    if prediction[0] == 2:
        result = 'Your Depression test result : Moderate Depression'
    if prediction[0] == 3:
        result = 'Your Depression test result : Moderately severe Depression'
    if prediction[0] == 4:
        result = 'Your Depression test result : Severe Depression'
    return redirect(url_for('test', result=result))


@app.route('/test')
def test():
    result = request.args.get('result', '')
    return render_template('test.html', data=result)


if __name__ == '__main__':
    app.run()
