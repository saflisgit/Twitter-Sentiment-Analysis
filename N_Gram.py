from nltk.tokenize import word_tokenize
import readData as rd
import numpy as np
import csv


class N_Gram(object):

    def __init__(self, traindata, obj_ui):
        self.tweets, self.labels = traindata['tweet'], traindata['Sentiment']

        self.pos_tweets, self.neg_tweets = self.labels.value_counts()[4], self.labels.value_counts()[0]
        self.total_tweets = self.pos_tweets + self.neg_tweets

        self.PBA_pos = dict()
        self.PBA_neg = dict()

        self.PA_pos = self.pos_tweets / self.total_tweets
        self.PA_neg = self.neg_tweets / self.total_tweets

        self.ui = obj_ui

# ========================================================================================

    def train(self):
        print('POSITIVE: ', self.pos_tweets, ' NEGATIVE: ', self.neg_tweets)

        self.calc_PBA()

        return self.PBA_pos, self.PBA_neg, self.pos_tweets, self.neg_tweets

# ========================================================================================

    def calc_PBA(self):
        for i in range(self.total_tweets):
            words = word_tokenize(self.tweets[i])
            n2gram = list()

            for j in range(len(words) - 1):
                n2gram.append((words[j] + words[j + 1]))



            for word in n2gram:
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
        n2gram = list()

        for i in range(len(words) - 1):
            n2gram.append((words[i] + words[i + 1]))

        for word in n2gram:
            p_pos *= self.PBA_pos.get(word, 1 / self.pos_tweets)
            p_neg *= self.PBA_neg.get(word, 1 / self.neg_tweets)

        p_pos *= self.PA_pos
        p_neg *= self.PA_neg

        # messsage = 'Tweet:' + tweet + '====> pos/neg: ' + str(p_pos / p_neg) + '\t--->' + str(p_pos >= p_neg)
        # self.ui.insert_msg_box(messsage)

        if p_pos >= p_neg:
            messsage = 'Tweet:' + tweet + '====> pos/neg: ' + str(p_pos / p_neg) + '\t--->' + 'Positive'
            self.ui.insert_msg_box(messsage)
            return 4
        else:
            messsage = 'Tweet:' + tweet + '====> pos/neg: ' + str(p_pos / p_neg) + '\t--->' + 'Negative'
            self.ui.insert_msg_box(messsage)
            return 0

# ========================================================================================

    def predict(self, test_data):
        result = dict()


        for (i, tweet) in enumerate(test_data):
            result[i] = int(self.classify(tweet))

        return result

# ========================================================================================

def metrics(labels, predictions, obj_ui):
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0

    for i in range(len(labels)):
        true_pos += int(labels[i] == 4 and predictions[i] == 4)
        true_neg += int(labels[i] == 0 and predictions[i] == 0)
        false_pos += int(labels[i] == 0 and predictions[i] == 4)
        false_neg += int(labels[i] == 4 and predictions[i] == 0)

    print(true_pos)
    print(true_neg)
    print(false_pos)
    print(false_neg)

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

    obj_ui.clean_msg_box()
    obj_ui.insert_msg_box('\nPrecision: ' + str(precision))
    obj_ui.insert_msg_box("\nRecall: " + str(recall))
    obj_ui.insert_msg_box("\nF-Score: " + str(fscore))
    obj_ui.insert_msg_box("\nAccuracy: " + str(accuracy))

def train_ngram(filename, outputfilename, obj_ui):
    filename = filename + '.csv'
    count_txt = outputfilename + '.txt'
    model_csv = outputfilename + '.csv'
    data = rd.read_and_clean_data(filename)

    total_tweets = data.shape[0]
    trainIndex, testIndex = list(), list()
    for i in range(total_tweets):
        if np.random.uniform(0, 1) < 0.9:
            trainIndex += [i]
        else:
            testIndex += [i]
    trainData = data.loc[trainIndex]
    testData = data.loc[testIndex]

    trainData.reset_index(inplace=True)
    trainData.drop(['index'], axis=1, inplace=True)

    testData.reset_index(inplace=True)
    testData.drop(['index'], axis=1, inplace=True)

    ngram = N_Gram(data, obj_ui)
    PBA_pos, PBA_neg, pos_tweets, neg_tweets = ngram.train()

    tweets_file = open(count_txt, 'w')
    tweets_file.write((str(pos_tweets) + '\n'))
    tweets_file.write((str(neg_tweets) + '\n'))

    with open(model_csv, 'w') as PBA_file:
        w = csv.writer(PBA_file, lineterminator='\n')
        w.writerows(PBA_pos.items())
        w.writerow('$')
        w.writerows(PBA_neg.items())

    preds = ngram.predict(testData['tweet'])
    metrics(testData['Sentiment'], preds, obj_ui)
