import readData as rd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

clf = GaussianNB()

data = rd.read_and_clean_data('train_short.csv')

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

train_corpus = trainData['tweet'].tolist()
test_corpus = trainData['tweet'].tolist()

#print(train_corpus)

train_corpus_target = trainData['Sentiment'].tolist()
testLabels = testData['Sentiment'].tolist()
print(testLabels)

vectorizer = TfidfVectorizer(ngram_range=(2,2), min_df=1, use_idf=True, smooth_idf=True)

# Vectorize the training data
X_train = vectorizer.fit_transform(train_corpus)

# Vectorize the testing data
X_test = vectorizer.transform(test_corpus)

#print(X_test)

# Train the SVM, optimized by Stochastic Gradient Descent
clf.fit(X_train.toarray(), train_corpus_target) # train_corpus_target is the correct values for each training data.

# Make predictions
pred = clf.predict(X_test.toarray())
print(pred)

print('accuracy : ', accuracy_score(testLabels, pred))



