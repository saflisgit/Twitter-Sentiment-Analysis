import NaiveBayes as nb


filename = input('Enter filename to train')

output = input('Enter filename to save model')

nb.train_naive(filename, output)


