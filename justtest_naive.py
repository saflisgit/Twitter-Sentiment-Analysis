import csv
from nltk.tokenize import word_tokenize
import readData as rd


class justtest_naive(object):

    def __init__(self, count_filename, PBA_filename, obj_ui):

        count_file = open(count_filename, 'r')

        self.pos_tweets = count_file.readline()
        self.neg_tweets = count_file.readline()

        self.pos_tweets = int(self.pos_tweets)
        self.neg_tweets = int(self.neg_tweets)
        self.total_tweets = self.pos_tweets + self.neg_tweets

        self.PA_pos = self.pos_tweets / self.total_tweets
        self.PA_neg = self.neg_tweets / self.total_tweets

        self.PBA_pos = dict()
        self.PBA_neg = dict()

        self.ui = obj_ui

        readPos = True

        reader = csv.reader(open(PBA_filename, 'r'))
        for row in reader:
            if row == ['$']:
                readPos = False
            elif readPos:
                k, v = row
                self.PBA_pos[k] = v
            else:
                k, v = row
                self.PBA_neg[k] = v

    def classify(self, tweet):
        p_pos = 1
        p_neg = 1

        words = word_tokenize(tweet)

        for word in words:
            p_pos *= float(self.PBA_pos.get(word, 1 / self.pos_tweets))
            p_neg *= float(self.PBA_neg.get(word, 1 / self.neg_tweets))

        p_pos *= self.PA_pos
        p_neg *= self.PA_neg

        #print('Tweet:', tweet, '====> pos/neg: ', p_pos/p_neg, '\t--->', (p_pos >= p_neg))

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

    """
    print(true_pos)
    print(true_neg)
    print(false_pos)
    print(false_neg)
    """

    try:
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
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


def test(model, test_data, obj_ui):
    counts_txt = model + '.txt'
    model_csv = model + '.csv'
    test_data = test_data + '.csv'

    testData = rd.read_and_clean_data(test_data)

    new_test = justtest_naive(counts_txt, model_csv, obj_ui)

    preds = new_test.predict(testData['tweet'])
    metrics(testData['Sentiment'], preds, obj_ui)


def predict_sentence(counts, model, sentence, obj_ui):
    new_test = justtest_naive(counts, model, obj_ui)

    pred = new_test.classify(sentence)

    if pred == 4:
        return 'POSITIVE'
    else:
        return 'NEGATIVE'