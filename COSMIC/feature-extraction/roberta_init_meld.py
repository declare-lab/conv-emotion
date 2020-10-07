import pickle

speakers, emotion_labels, sentiment_labels, sentences, train_ids, test_ids, valid_ids = pickle.load(open('meld/meld.pkl', 'rb'))

def write_to_file(ids, split):
    f1 = open('meld/' + split + '.input0', 'w')
    f2 = open('meld/' + split + '.label', 'w')
    
    for item in ids:
        sent = sentences[item]
        label = emotion_labels[item]
        
        for s, l in zip(sent, label):
            x = s.encode('ascii', errors='ignore').decode("utf-8")
            f1.write(x + '\n')
            f2.write(str(l) + '\n')
    
    f1.close()
    f2.close()

write_to_file(train_ids, 'train')
write_to_file(valid_ids, 'valid')
write_to_file(test_ids, 'test')