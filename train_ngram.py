import N_Gram as ng


filename = input('Enter filename to train')

output = input('Enter filename to save model')

ng.train_from_csv(filename, output)




