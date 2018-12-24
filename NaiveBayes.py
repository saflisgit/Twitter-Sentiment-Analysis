from nltk.tokenize import word_tokenize
import readData as rd
import numpy as np


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
                    self.PBA_pos[word] = self.PBA_pos.get(word, 1) + 1
                else:
                    self.PBA_neg[word] = self.PBA_neg.get(word, 1) + 1

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
            p_pos *= self.PBA_pos.get(word, 1 / self.pos_tweets)
            p_neg *= self.PBA_neg.get(word, 1 / self.neg_tweets)

        p_pos *= self.PA_pos
        p_neg *= self.PA_neg

        print('Tweet:', tweet, '====> pos/neg: ', p_pos/p_neg, '\t--->', (p_pos >= p_neg))
        if p_pos >= p_neg:
            return 4
        else:
            return 0

# ========================================================================================

    def predict(self, test_data):
        result = dict()

        for (i, tweet) in enumerate(test_data):
            result[i] = int(self.classify(tweet))

        return result

# ========================================================================================

def metrics(labels, predictions):
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0

    for i in range(len(labels)):
        true_pos += int(labels[i] == 4 and predictions[i] == 4)
        true_neg += int(labels[i] == 0 and predictions[i] == 0)
        false_pos += int(labels[i] == 0 and predictions[i] == 4)
        false_neg += int(labels[i] == 4 and predictions[i] == 0)

    """
    print(true_pos)
    print(true_neg)
    print(false_pos)
    print(false_neg)
    """

    try:
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        print('Precall')
        print(precision, ',', recall)
        fscore = 2 * precision * recall / (precision + recall)
        accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)

    except ZeroDivisionError:
        print('Zero Division Error')

    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F-score: ", fscore)
    print("Accuracy: ", accuracy)



filename = 'train.csv'

data = rd.read_and_clean_data(filename)

total_tweets = data.shape[0]
trainIndex, testIndex = list(), list()
for i in range(total_tweets):
    if np.random.uniform(0, 1) < 0.75:
        trainIndex += [i]
    else:
        testIndex += [i]
trainData = data.loc[trainIndex]
testData = data.loc[testIndex]

trainData.reset_index(inplace=True)
trainData.drop(['index'], axis=1, inplace=True)

testData.reset_index(inplace=True)
testData.drop(['index'], axis=1, inplace=True)


naive_data = NaiveBayes(trainData)
naive_data.train()
preds = naive_data.predict(testData['tweet'])
#print(testData['Sentiment'])
#print(preds)
metrics(testData['Sentiment'], preds)
