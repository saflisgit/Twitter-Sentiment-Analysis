import readData as rd

data = rd.read_and_clean_data('train_short.csv')

rd.print_head_and_tail(data)

print(data['tweet'][3])