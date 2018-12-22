from nltk.tokenize import word_tokenize
import readData as rd


class NaiveBayes(object):

    def __init__(self, traindata):
        self.tweets, self.labels = traindata['tweet'], traindata['Sentiment']

        self.pos_tweets, self.neg_tweets = self.labels.value_counts()[4], self.labels.value_counts()[0]
        self.total_tweets = self.pos_tweets + self.neg_tweets

        self.PBA_pos = dict()
        self.PBA_neg = dict()

        self.PA_pos = self.pos_tweets / self.total_tweets
        self.PA_neg = self.neg_tweets / self.total_tweets

# ========================================================================================

    def train(self):
        print('POSITIVE: ', self.pos_tweets, ' NEGATIVE: ', self.neg_tweets)

        self.calc_PBA()

# ========================================================================================

    def calc_PBA(self):
        for i in range(self.total_tweets):
            words = word_tokenize(self.tweets[i])

            for word in words:
                if self.labels[i] == 4:
                    self.PBA_pos[word] = self.PBA_pos.get(word, 0) + 1
                else:
                    self.PBA_pos[word] = self.PBA_neg.get(word, 0) + 1

        for word in self.PBA_pos:
            self.PBA_pos[word] = self.PBA_pos[word] / self.pos_tweets

        for word in self.PBA_neg:
            self.PBA_neg[word] = self.PBA_neg[word] / self.neg_tweets

# ========================================================================================

    def classify(self, tweet):
        p_pos = 1
        p_neg = 1

        words = word_tokenize(tweet)

        for word in words:
            p_pos *= self.pxc_pos.get(word, 1) * self.pc_pos
            p_neg *= self.pxc_neg.get(word, 1) * self.pc_neg

        print('Tweet:', tweet, '\t--->', (p_pos >= p_neg))
        return p_pos >= p_neg



filename = 'train.csv'

data = rd.read_and_clean_data(filename)

naive_data = NaiveBayes(data)
naive_data.train()