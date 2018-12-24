import readData as rd
from sklearn.feature_extraction.text import CountVectorizer

data = rd.read_and_clean_data('train_short.csv')
#data = rd.removespaces(data)

#rd.print_head_and_tail(data)

sample = [data['tweet'][3], data['tweet'][5]]

print(sample)

gram = list()

vectorizer = CountVectorizer(ngram_range=(1, 4))

x = vectorizer.fit_transform(sample)

print(vectorizer.get_feature_names())

print(x.toarray())


