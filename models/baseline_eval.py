import csv
import pylab as pl
import cPickle as pickle
import itertools
import numpy as np
from scipy.spatial.distance import cdist
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

words,defs = pickle.load(open("train_data_filtered.pkl", "rb" ))
word_embs = pickle.load(open("D_cbow_pdw_8B.pkl", "rb" ))
MODEL_PATH = '128_ada_weights.best.hdf5'
OUPUT_SHORTLIST_FILE = 'output_shortlist.txt'
TEST_SEEN_DATA = 'WN_seen_correct.txt'
training_words = words
training_defs = defs

#Create vocab
def create_vocab(words,defs):
    word_vocab = {}
    #Make sure index starts with 1, since 0 is mask on
    idx = 1
    for word in words:
        if word not in word_vocab:
            word_vocab[word] = idx
            idx += 1

    for word_def in defs:
        for word in word_def:
            if word not in word_vocab:
                word_vocab[word] = idx
                idx += 1
    print('Unique words found in vocab ',idx)
    return word_vocab

training_vocab = create_vocab(training_words,training_defs)
print('Created training vocab')

output_words = [line.strip() for line in open(OUPUT_SHORTLIST_FILE)]
op_sl_vocab = {}
op_sl_rev_dict = {}
idx = 0
for word in output_words:
    if word not in op_sl_vocab:
        op_sl_vocab[word] = idx
        op_sl_rev_dict[idx] = word
        idx += 1

print('Created output vocab and output reverse dict')
'''
Prepare test data from files
Only include words present in the output_vocab
'''
def create_test_data(file_name, vocab_obj):
    X_test = []
    Y_test = []
    with open(file_name) as f:
        for line in f:
            test_wd = line.strip().split('\t')
            if(test_wd[0] in vocab_obj):
                Y_test.append(test_wd[0])
                X_test.append(test_wd[1])
    return X_test,Y_test

x,y = create_test_data(TEST_SEEN_DATA, op_sl_vocab)
print('Created test data instances')

# points should have embeddings of all output words in a list
#Normalize output word embeddings to unit vectors
def normalize_vocab(voc):
    points = [None]*(len(voc))
    for w, idx in voc.iteritems():
        points[idx] = word_embs[w]/np.linalg.norm(word_embs[w])
    points = np.array(points)
    return points

'''
Returns true if label in top k closest for pred in labels
pred - 500d emb
label - 500d emb

Normalize pred to unit vector
Take cosine distance to find closest neighbors
Return rank and compute all metrics given in the table
'''
def get_pred_rank (pred, label, voc):
    label_idx = voc[label]
    dist_list = cdist(points, np.array([pred]), 'cosine').flatten() #Convert 2d into 1d array
    #result_indices = np.argpartition(dist_list, k)[:k]
    #if label_idx in result_indices:
    #    return True
    #return False
    result_indices = list(np.argsort(dist_list))
    return result_indices

def evaluate(test_def, test_label):
    test_sent = []
    word_seq = [training_vocab[t] for t in test_def if t in training_vocab]
    test_sent.append(word_seq)
    test_sent = pad_sequences(test_sent, maxlen=100, dtype='int32', padding='post', truncating='post')
    pred = model1.predict(test_sent, batch_size=1, verbose=1)
    pred = pred/np.linalg.norm(pred)
    ops = get_pred_rank(pred[0], test_label, op_sl_vocab)
    c = 0
    sl = []
    for op_w in ops:
        sl.append(op_sl_rev_dict[op_w])
        c += 1
        if(c > 101):
            break
    if test_label in sl:
        return sl.index(test_label)
    else:
        return -1

model1 = load_model(MODEL_PATH)
points = normalize_vocab(op_sl_vocab)
len(points)
t10 = 0
t100 = 0
for i in range(len(x)):
    test_def = x[i]
    test_label = y[i]
    r = evaluate(test_def, test_label)
    if(i % 10 == 0):
        print(str(t10)+' '+str(t100))
    if(r >= 0 and r < 11):
        t10 += 1
    if(r >= 0 and r < 101):
        t100 += 1
print('Done')
print('Top 10 accuracy '+str(t10*100.0/len(x)))
print('Top 100 accuracy '+str(t100*100.0/len(x)))