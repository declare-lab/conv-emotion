import pickle

speakers, labels, sentences, train_ids, test_ids, valid_ids = pickle.load(open('iemocap/iemocap.pkl', 'rb'))

def write_to_file(ids, split):
    f1 = open('iemocap/' + split + '.input0', 'w')
    f2 = open('iemocap/' + split + '.label', 'w')
    
    for item in ids:
        sent = sentences[item]
        label = labels[item]
        
        for s, l in zip(sent, label):
            f1.write(s + '\n')
            f2.write(str(l) + '\n')
    
    f1.close()
    f2.close()
              
write_to_file(train_ids, 'train')
write_to_file(valid_ids, 'valid')
write_to_file(test_ids, 'test')