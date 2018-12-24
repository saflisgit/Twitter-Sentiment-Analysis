import pandas as pd
import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def read_and_clean_data(filename):

    data = read_from_csv(filename)
    data = removeMentions(data)
    data = removeNonLetters(data)
    data = remove_stop_words(data)
    data = stem(data)
    data = removeNonLetters(data)

    return data

# ==================================================================================================

def read_from_csv(filename):

    # READING THE DATA FROM CSV FILE

    data = pd.read_csv(filename, encoding='latin-1', usecols=['tweet', 'Sentiment'])

    data = data[['tweet', 'Sentiment']]

    return data

# ==================================================================================================

def print_head_and_tail(data):
    print('===========Head and tail of data============')
    print(data.head())
    print('---------------------------------------------')
    print(data.tail())


# ==================================================================================================


def remove_stop_words(data):
    # STOP WORD REMOVAL

    stop = stopwords.words('english')

    data['tweet'] = data['tweet'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in stop]))

    return data


# ==================================================================================================

def stem(data):
    # STEMMING

    ps = PorterStemmer()

    data['tweet'] = data['tweet'].apply(
        lambda x: ' '.join([ps.stem(word) for word in x.split()]))

    return data

def removeMentions(data):
    data['tweet'] = [re.sub(r'@[A-Za-z0-9_;&,.#]+', '', tweet) for tweet in data['tweet']]
    data['tweet'] = [re.sub(r'@+', '', tweet) for tweet in data['tweet']]
    data['tweet'] = [re.sub('https?://[A-Za-z0-9./]+', '', tweet) for tweet in data['tweet']]
    data['tweet'] = [re.sub('www?[A-Za-z0-9./]+', '', tweet) for tweet in data['tweet']]

    return data

def removeNonLetters(data):
    data['tweet'] = [re.sub("[^a-zA-Z]", " ", tweet) for tweet in data['tweet']]
    data['tweet'] = [re.sub(' +', ' ', tweet) for tweet in data['tweet']]

    return data