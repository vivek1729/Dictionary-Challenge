import cPickle as pickle
import itertools
import numpy as np
from scipy.spatial.distance import cdist
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Flatten,LSTM
from keras.layers.embeddings import Embedding
from keras import optimizers
from keras.callbacks import ModelCheckpoint,CSVLogger
from keras.models import load_model
from keras import backend as K

words,defs = pickle.load(open("train_data_filtered.pkl", "rb" ))
print('Loaded words and definitions')
word_embs = pickle.load(open("D_cbow_pdw_8B.pkl", "rb" ))
print('Loaded word embeddings')
training_vocab = pickle.load(open("training_dict.pkl", "rb" ))
print('Loaded training dictionary')
print(len(training_vocab))

#Hyper params definition (from paper)
vocab_len = len(training_vocab) #To take care of masking
max_len = 32 #Number of timesteps
input_emb_dim = 500 #Dimension of learned input embeddings
output_emb_dim = 500 #Standard dim of output provided
lstm_units = 512
batch_size = 64
epochs = 10
learning_rate = 0.001
CSV_LOG_PATH = 'logs/exp3_bl.log'
MODEL_CHECKPT = 'checkpoints/exp3_bl.hdf5'
OUPUT_SHORTLIST_FILE = 'output_shortlist.txt'
TEST_SEEN_DATA = 'WN_seen_correct.txt'

def normalize(v):
    return v / np.sqrt(np.sum(v**2))

embedding_matrix = np.zeros((vocab_len, output_emb_dim))
for word, i in training_vocab.items():
    if i > 0:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = normalize(word_embs[word])
    else:
        print('Embedding not found')
        v = np.random.normal(scale=0.6, size=(output_emb_dim, ))
        embedding_matrix[i] = normalize(v)
print("Embedding matrix created!")

training_x = []
training_y = []
for w,d in itertools.izip(words,defs):
    word_seq = [training_vocab[t] for t in d if t in training_vocab]
    training_x.append(word_seq)
    #Normalize word embs
    training_y.append(normalize(word_embs[w]))

training_x = pad_sequences(training_x, maxlen=max_len, dtype='int32', padding='post', truncating='post')
training_y = np.array(training_y)
print('Created training data set')
print(training_x.shape)
print(training_y.shape)

# Free memory
words = None
defs = None
#word_embs = None
#training_vocab = None

print('Cleared memory')

model = Sequential()
model.add(Embedding(vocab_len, input_emb_dim, mask_zero=True, weights=[embedding_matrix], input_length=max_len, trainable=False))
model.add(LSTM(lstm_units))
model.add(Dense(lstm_units, activation='tanh'))
model.add(Dense(output_emb_dim, activation='linear'))

def cos_distance(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return 1 - K.sum((y_true * y_pred), axis=-1)


csv_logger = CSVLogger(CSV_LOG_PATH)
filepath= MODEL_CHECKPT
checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True, monitor='loss')

rmsprop = optimizers.RMSprop(lr=learning_rate)
model.compile(optimizer=rmsprop,loss=cos_distance,metrics=['accuracy'])
#model = load_model(MODEL_CHECKPT, custom_objects={'cos_distance': cos_distance})
print(model.summary())

print('Starting training..')
model.fit(training_x,training_y, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=1, validation_split=0.05, callbacks=[])

print('Training complete')

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

print('Loading skipgram vectors...')
wv_vectors = np.zeros((len(op_sl_vocab.keys()), word_embs.values()[0].shape[0]))

for ii, (kk, vv) in enumerate(op_sl_vocab.iteritems()):
    wv_vectors[vv,:] = word_embs[kk]

wv_vectors_normed = wv_vectors / (np.sqrt((wv_vectors ** 2).sum(axis=1))[:,None])
print 'Done'

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

points = 0
t10 = 0
t100 = 0
for i in range(len(x)):
    test_def = x[i]
    test_label = y[i]
    test_sent = []
    word_seq = [training_vocab[t] for t in test_def if t in training_vocab]
    test_sent.append(word_seq)
    test_sent = pad_sequences(test_sent, maxlen=max_len, dtype='int32', padding='post', truncating='post')
    vec = model.predict(test_sent, batch_size=1, verbose=1)
    vec = normalize(vec[0])
    sims_rnn = (wv_vectors_normed * vec).sum(1)
    sorted_idx_rnn = sims_rnn.argsort()[::-1]
    if(test_label in op_sl_vocab):
        points += 1
        test_idx = op_sl_vocab[test_label]
        for j in range(101):
            if(sorted_idx_rnn[j] == test_idx):
                if(j < 11):
                    t10 += 1
                else:
                    t100 += 1

print('Evaluation complete')
print('Top 10 accuracy '+str(t10*100.0/points))
print('Top 100 accuracy '+str(t100*100.0/points))