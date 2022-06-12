import snscrape.modules.twitter as sntwitter


def get_tweets_by_username(username, start, end):

    # until:2022-06-10 since:2021-06-10
    query = f"(from:{username}) until:{end} since:{start}"
    tweets_content = []
    limit = 10

    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        if len(tweets_content) == limit:
            break
        else:
            tweets_content.append(tweet.content)
    return tweets_content
