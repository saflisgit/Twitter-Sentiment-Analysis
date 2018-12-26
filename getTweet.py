import tweepy
import readData as rd

auth = tweepy.OAuthHandler('cuFTETVkZHxz8WXA5kX5cfvzD', '8Asz9qQby2RE1sCWMn94vA2Gg2MExtj8RKQgGZWZqrlipk1nHP')
auth.set_access_token('850098837624688644-BTKOuEbqzGgnbEgKHcs8xSAy79XHfUC', '0SWl26cslynakivgMB2JprtnJzsVz3bi8Dyax9Mj7Nng9')

api = tweepy.API(auth)

# public_tweets = api.home_timeline()
# for tweet in public_tweets:
#     print (tweet.text)

tweets = list()

# def process_status(sta):
#     print (sta.text )


def getTweet(userID):
    tweets.clear()
    for status in tweepy.Cursor(api.user_timeline, id=userID).items(10):
        #text = rd.read_and_clean_sentence(status.text)
        tweets.append(status.text)
        #print(status.text)
    print(tweets)
    return tweets



