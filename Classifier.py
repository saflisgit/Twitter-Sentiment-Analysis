from math import log
from nltk.tokenize import word_tokenize


class Classifier(object):

    def __init__(self, traindata, method='tf-idf'):
        self.tweets, self.labels = traindata['SentimentText'], traindata['Sentiment']
        self.method = method

    def train(self):
        self.calc_TF_and_IDF()
        if self.method == 'tf-idf':
            self.calc_TF_IDF()
        else:
            self.calc_prob()

    def calc_prob(self):
        self.prob_pos = dict()
        self.prob_neg = dict()
        for word in self.tf_pos:
            self.prob_pos[word] = (self.tf_pos[word] + 1) / (self.pos_words +len(list(self.tf_pos.keys())))

        for word in self.tf_neg:
            self.prob_neg[word] = (self.tf_neg[word] + 1) / (self.neg_words + len(list(self.tf_neg.keys())))

        self.prob_pos_tweet, self.prob_neg_tweet = \
            self.pos_tweets / self.total_tweets, self.neg_tweets / self.total_tweets


    def calc_TF_IDF(self):
        self.prob_pos = dict()
        self.prob_neg = dict()
        self.sum_tf_idf_pos = 0
        self.sum_tf_idf_neg = 0
        for word in self.tf_pos:
            self.prob_pos[word] = (self.tf_pos[word]) * log(
                (self.pos_tweets + self.neg_tweets) / (self.idf_pos[word] + self.idf_neg.get(word, 0)))
            self.sum_tf_idf_pos += self.prob_pos[word]

        for word in self.tf_pos:
            self.prob_pos[word] =\
                (self.prob_pos[word] + 1) / (self.sum_tf_idf_pos + len(list(self.prob_pos.keys())))

        for word in self.tf_neg:
            self.prob_neg[word] = (self.tf_neg[word]) * log(
                (self.neg_tweets + self.pos_tweets) / (self.idf_neg[word] + self.idf_pos.get(word, 0)))
            self.sum_tf_idf_neg += self.prob_neg[word]

        for word in self.tf_neg:
            self.prob_neg[word] =\
                (self.prob_neg[word] + 1) / (self.sum_tf_idf_neg + len(list(self.prob_neg.keys())))

        self.prob_pos_tweet, self.prob_neg_tweet =\
            self.pos_tweets / self.total_tweets, self.neg_tweets * self.total_tweets

    def calc_TF_and_IDF(self):

        no_of_tweets = self.tweets.shape[0]
        self.pos_tweets, self.neg_tweets = self.labels.value_counts()[1], self.labels.value_counts()[0]
        self.total_tweets = self.pos_tweets + self.neg_tweets
        self.pos_words = 0
        self.neg_words = 0
        self.tf_pos = dict()
        self.tf_neg = dict()
        self.idf_pos = dict()
        self.idf_neg = dict()
        for i in range(no_of_tweets):
            tweet = word_tokenize(self.tweets[i])
            count = list() #To keep track of whether the word has ocured in the message or not.

            for word in tweet:
                if self.labels[i]:
                    self.tf_pos[word] = self.tf_pos.get(word, 0) + 1
                    self.pos_words += 1
                else:
                    self.tf_neg[word] = self.tf_neg.get(word, 0) + 1
                    self.neg_words += 1
                if word not in count:
                    count += [word]

            for word in count:
                if self.labels[i]:
                    self.idf_pos[word] = self.idf_pos.get(word, 0) + 1
                else:
                    self.idf_neg[word] = self.idf_neg.get(word, 0) + 1

    def classify(self, tweet):
        p_pos, p_neg = 0, 0

        for word in tweet:
            if word in self.prob_pos:
                p_pos += log(self.prob_pos[word])
            else:
                if self.method == 'tf-idf':
                    p_pos -= log(self.sum_tf_idf_pos + len(list(self.prob_pos.keys())))
                else:
                    p_pos -= log(self.pos_words + len(list(self.prob_pos.keys())))

            if word in self.prob_neg:
                p_neg += log(self.prob_neg[word])
            else:
                if self.method == 'tf-idf':
                    p_neg -= log(self.sum_tf_idf_neg + len(list(self.prob_neg.keys())))
                else:
                    p_neg -= log(self.neg_words + len(list(self.prob_neg.keys())))

            p_pos += log(self.prob_pos_tweet)
            p_neg += log(self.prob_neg_tweet)

        print('Tweet:', tweet, '\t--->', (p_pos >= p_neg))
        return p_pos >= p_neg


    def predict(self, testData):
        result = dict()
        for(i, tweet) in enumerate(testData):
            words = word_tokenize(tweet)
            result[i] = int(self.classify(words))
        return result

# =================================================================================================

def metrics(labels, predictions):
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
    for i in range(len(labels)):
        true_pos += int(labels[i] == 1 and predictions[i] == 1)
        true_neg += int(labels[i] == 0 and predictions[i] == 0)
        false_pos += int(labels[i] == 0 and predictions[i] == 1)
        false_neg += int(labels[i] == 1 and predictions[i] == 0)

    try:
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        print('Precall')
        print(precision, ',', recall)
        fscore = 2 * precision * recall / (precision + recall)
        accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)

        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F-score: ", fscore)
        print("Accuracy: ", accuracy)
    except ZeroDivisionError:
        print('Zero Division Error')

# ================================================================================================


import read_data as rd
import matplotlib.pyplot as plt
import numpy as np

from wordcloud import WordCloud


filename = 'train.csv'

data = rd.read_and_clean_data(filename)

rd.print_head_and_tail(data)

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


sc_tf_idf = classifier(trainData, 'tf-idf')
sc_tf_idf.train()
preds_tf_idf = sc_tf_idf.predict(testData['SentimentText'])
metrics(testData['Sentiment'], preds_tf_idf)

