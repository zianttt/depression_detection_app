import snscrape.modules.twitter as sntwitter


def get_tweets_by_username(username):

    query = f"(from:{username}) until:2020-01-01 since:2010-01-01"
    tweets_content = []
    limit = 10

    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        if len(tweets_content) == limit:
            break
        else:
            tweets_content.append(tweet.content)
    return tweets_content
