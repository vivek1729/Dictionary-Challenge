
# coding: utf-8

# In[21]:

from __future__ import print_function
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math, copy, time
import csv
import datetime

CTX_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_GPU = torch.cuda.is_available()
RESUME_TRAINING = False
WORD_DIM = 500

# In[4]:
print('Using device '+str(CTX_DEVICE))


# In[4]:

INP_DIR = "/home/vivekp/DictionaryChallenge"
MODEL_PATH = INP_DIR+"/checkpoints/exp23.tar"
CSV_LOG_FILE = INP_DIR+"/logs/exp23.csv"


# In[5]:

words,defs = pickle.load(open(INP_DIR+"/train_data_filtered.pkl", "rb" ),encoding='latin1')
if(WORD_DIM == 500):
    word_embs = pickle.load(open(INP_DIR+"/D_cbow_pdw_8B.pkl", "rb" ),encoding='latin1')
elif(WORD_DIM == 300):
    word_embs = pickle.load(open(INP_DIR+"/glove_300_6B.pkl", "rb" ),encoding='latin1')
X_vocab = pickle.load(open(INP_DIR+"/full_data_set_vocab.pkl", "rb" ),encoding='latin1')
#Based on https://stackoverflow.com/questions/32957708/python-pickle-error-unicodedecodeerror

# In[5]:

#Selecting subset of data
training_words = words
training_defs = defs

# In[7]:

#Hyper params definition (from paper)
num_blocks = 6 #Number of transformer blocks
num_attn_heads = 4 #Number of multi attn heads
vocab_len = len(X_vocab) #Restrict to some arbitrary top k?
max_len = 32 #Number of timesteps
input_emb_dim = WORD_DIM #Dimension of learned input embeddings
output_emb_dim = WORD_DIM #Standard dim of output provided
batch_size = 32
epochs = 6
train_val_split = 0.95
global_dropout = 0.1
opt_mf = 1
opt_ws = 8000
loss_freq = 100 #Print loss after every n batches in training
epsilon = 1e-07 #Small value added to avoid Nans
lambda1 = 1
lambda2 = 0.2

# In[8]:

def remove_unk(v):
    return [[1 if w >= vocab_len else w for w in sen] for sen in v]


# In[9]:

training_x = []
training_y = [] # legacy code
#training_y = training_words
#Only include examples for which you find an embedding
for i in range(len(training_words)):
    if(training_words[i] in word_embs):
        training_y.append(word_embs[training_words[i]])
        training_x.append([X_vocab[w] if w in X_vocab else 1 for w in training_defs[i]])
training_x = remove_unk(training_x)
#Pad sequences, prepare training data
X = np.ones((len(training_x), max_len)) * 0
for i in range(len(training_x)):
    seq = training_x[i]
    seq_len = min(len(seq),max_len)
    X[i,0:seq_len] = seq[0:seq_len]
Y = np.array(training_y) # legacy code
#Y = training_y
data_set = []
for i in range(len(training_x)):
    if(len(training_x[i]) > 0):
        data_set.append((X[i],Y[i]))


# In[10]:

#Split dataset into training and validation set
np.random.shuffle(data_set)
val_idx = math.floor(train_val_split*len(data_set))
print(val_idx)
train_set = data_set #[:val_idx]
val_set = data_set[val_idx:]
print('Training and validation sets created')


# In[11]:

#Create embedding matrix
embedding_matrix = np.zeros((vocab_len, output_emb_dim))
for word, i in X_vocab.items():
    if(i == 1):
        print(word)
        embedding_matrix[i] = word_embs['unk']
    elif(i > 1):
        embedding_vector = word_embs.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

        else:
            print('Embedding not found for '+word)
            embedding_matrix[i] = word_embs['unk']
print("Embedding matrix created!")


# In[12]:

def get_mask(data_inst):
    mask = np.ones(max_len)
    zero_idx = np.where(data_inst[0]==0)
    mask[zero_idx] = 0
    return mask


# In[13]:

class BatchIterator:

    """Iterator that returns batches of fixed sizes for specified epochs"""

    def __init__(self, data_set, batch_size, epochs):
        self.data_set = data_set
        self.start_idx = 0
        self.epochs = epochs
        self.batch_size = batch_size
        self.len_data_set = len(self.data_set)
        self.current_epoch = 1
        self.current_batch = 1       
        self.batches_per_epoch = math.ceil(self.len_data_set/self.batch_size)
        #print(self.len_data_set)
        #print(self.batches_per_epoch)
        np.random.shuffle(self.data_set)
    def has_ended(self):
        return self.current_epoch > self.epochs
    
    def __iter__(self):
        return self

    def __next__(self):
        #print('Current start index '+str(self.start_idx))
        #print('Current epoch, current batch '+str(self.current_epoch)+' , '+str(self.current_batch))
        batch = []
        masks = []
        i = self.start_idx
        end = min(self.len_data_set-1,i+self.batch_size-1)
        #print(end)
        while(i <= end):
            batch.append(self.data_set[i])
            masks.append(get_mask(self.data_set[i]))
            i += 1
        #If can't find enough data points till the end of list, start from zero again
        if(len(batch) < self.batch_size):
            #print('Over shot the length of the list')
            i = 0
            while(len(batch) != self.batch_size):
                batch.append(self.data_set[i]) 
                masks.append(get_mask(self.data_set[i]))
                i += 1
        self.start_idx = i
        
        #Increment epoch, reset variables, reshuffle the dataset
        if(self.current_batch == self.batches_per_epoch):
            self.current_epoch += 1            
            self.current_batch = 1
            #Shuffle the list, reset start idx
            np.random.shuffle(self.data_set)
            self.start_idx = 0
        else:
            self.current_batch += 1
        return batch, masks


# In[22]:

out_embs = []
inp_embs = []
enc_out = []
attn_arr = []
total_loss = 0


# In[23]:

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# In[24]:

class LayerNorm(nn.Module):
    "Construct a layernorm module"
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features, device=CTX_DEVICE))
        self.b_2 = nn.Parameter(torch.zeros(features, device=CTX_DEVICE))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# In[25]:

class OutputAttentionLayer(nn.Module):
    def __init__(self, attn_size, batch_size):
        super(OutputAttentionLayer, self).__init__()
        self.batch_size = batch_size
        self.attn_size = attn_size
        self.attn_keys = nn.Linear(self.attn_size, self.attn_size)
        self.attn_values = nn.Linear(self.attn_size, self.attn_size)
        self.attn_query = nn.Parameter(torch.Tensor(self.attn_size,1, device=CTX_DEVICE),requires_grad=True)
        nn.init.xavier_uniform_(self.attn_query.data)
        self.counter = 0
        
    def forward(self, key, value, masks):
        key_mat = self.attn_keys(key) #b * max_len * d
        value_mat = self.attn_values(value) #b * max_len * d
        query_mat = self.attn_query.unsqueeze(0).repeat(self.batch_size,1,1) #b * d * 1
        #soft_out = F.softmax(torch.bmm(key_mat,query_mat),dim=1) #b * max_len * 1
        kv = torch.bmm(key_mat,query_mat)
        #Subtract max to stabilize softmax
        m = torch.max(kv,dim = 1,keepdim=True)[0]
        kv = kv - m
        soft_out = F.softmax(kv,dim=1)
               
        #masks is of dimensions b * max_len
        attn_scores = soft_out.squeeze(2)
        #attn_arr.append(attn_scores)
        masked_scores = attn_scores * masks
        den_sum = masked_scores.sum(dim=1, keepdim=True)
        attn_scores = masked_scores.div(den_sum)
        ret_attn_scores = attn_scores
        attn_scores = attn_scores.unsqueeze(2) # b * max_len * 1
        self.counter += 1
        
        '''
        if(self.counter % 1 == 0):
            sf = soft_out.squeeze(2)
            print('keys')
            print(torch.min(key))
            print(torch.max(key))
            print(torch.mean(key))
            print('Attn scores')
            print(torch.min(sf))
            print(torch.max(sf))
            print('Masks......')
            print(masks.sum(dim=1))
            print('************************')
        '''
        ctx_vecs = torch.bmm(attn_scores.transpose(1,2),value_mat) #b * 1 * d
        return ctx_vecs, ret_attn_scores.clone().detach().requires_grad_(False)


# In[26]:

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model, dtype=torch.float32, device=CTX_DEVICE)
        position = torch.arange(0, max_len, dtype=torch.float32, device=CTX_DEVICE).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=CTX_DEVICE) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)


# In[27]:

class DefinitionEncoder(nn.Module):
    '''
    Main module that encodes a definition and spits out an embedding for the word
    '''
    def __init__(self, num_layers, weight_matrix, embedding_dim, hidden_dim, 
                 output_emb_dim, seq_len, dropout, num_heads, batch_size):
        super(DefinitionEncoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_emb_dim = output_emb_dim
        self.seq_len = seq_len
        self.dropout = dropout
        self.batch_size = batch_size
        
        params =(hidden_dim, 
                 num_heads,
                 dropout)
        
        self.word_embeddings = nn.Embedding.from_pretrained(weight_matrix)
        #Trainable false
        self.word_embeddings.weight.requires_grad = False
        #self.timing_signal = _gen_timing_signal(self.seq_len, self.hidden_dim)
        self.pe = PositionalEncoding(embedding_dim, 0, seq_len)
        #self.layers = nn.Sequential(*[EncoderLayer() for l in range(N)])
        self.layers = clones(EncoderLayer(hidden_dim, num_heads, dropout), self.num_layers)
        self.norm = LayerNorm(self.hidden_dim) #hidden size, not sure what it is, we will find out!
        #self.input_dropout = nn.Dropout(self.dropout)
        self.output_attn = OutputAttentionLayer(self.hidden_dim, self.batch_size)
        self.output_projection_layer = nn.Linear(self.hidden_dim, self.output_emb_dim)
        #Do I need normalization, 
        #may be I will have to pass this output through another linear layer and spit out the embedding
    
    def init_emb_layer(self, weight_matrix):
        self.word_embeddings = nn.Embedding.from_pretrained(weight_matrix)
        #Trainable false
        self.word_embeddings.weight.requires_grad = False
        
    def forward(self, inputs, mask):
        x = self.word_embeddings(inputs)
        #x = self.input_dropout(embeds) 
        # Add timing signal
        #x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
        x += self.pe(x)
        #inp_embs.append(x)
        #x = self.layers(x, mask)
        for layer in self.layers:
            x = layer(x, mask.unsqueeze(2))
        x = self.norm(x)
        #enc_out.append(x)
        #print('Going into attn with shape')
        #print(x.shape)
        #print(mask.shape)
        ctx_vecs, attn_scores = self.output_attn(x, x, mask)
        tanh_out = torch.tanh(ctx_vecs.squeeze(1)) #b * d
        #X = tanh_out.contiguous()
        #X = X.view(-1, X.shape[2])
        emb_out = self.output_projection_layer(tanh_out)
        return emb_out, attn_scores


# In[28]:

class EncoderLayer(nn.Module):
    '''
    One layer of transformer containing Multi-head attention and Position Feed Forward modules
    '''
    def __init__(self, hidden_dim, num_heads, dropout):
        super(EncoderLayer, self).__init__()
        #self.multi_head_attention = MultiHeadAttentionLayer()
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads #Has to be a factor of the hidden size
        self.multi_head_attn = MultiHeadedAttention(self.num_heads, self.hidden_dim)
        self.positionwise_feed_forward = PositionwiseFeedForward(hidden_dim, dropout)
        self.layer_dropout = nn.Dropout(self.dropout)
        self.layer_norm_mha = LayerNorm(self.hidden_dim)
        self.layer_norm_ffn = LayerNorm(self.hidden_dim)
        self.counter = 1
        #Can add more layers
        
    def forward(self, x, mask):       
        #print(mask)       
        # Layer Normalization
        x_norm = self.layer_norm_mha(x)
        
        # Multi-head attention
        y = self.multi_head_attn(x_norm, x_norm, x_norm, mask)
        #if(torch.isnan(soft_out).any()):
        '''
        if(self.counter % 1 == 0):
            print('MHA output')
            print(torch.min(y))
            print(torch.max(y))
            print('==========')
        '''
        # Dropout and residual
        x = self.layer_dropout(x + y)
        
        # Layer Normalization
        x_norm = self.layer_norm_ffn(x)
        
        # Positionwise Feedforward
        y = self.positionwise_feed_forward(x_norm)
        
        # Dropout and residual, in https://www.youtube.com/watch?v=OYygPG4d9H0, they drop a residual connection
        # Check again, you might be able to greatly simplify your model
        y = self.layer_dropout(x + y)
        
        return y


# In[29]:

class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.dropout = 0.1
        self.d_ff = hidden_dim*2
        self.d_model = hidden_dim
        self.w_1 = nn.Linear(self.d_model, self.d_ff)
        self.w_2 = nn.Linear(self.d_ff, self.d_model)
        self.dropout = nn.Dropout(self.dropout)
        
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# In[30]:

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(d_k)
    #print(scores.shape)
    #print(key.shape)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    #Stabilize softmax
    m = torch.max(scores,dim=-1,keepdim=True)[0]
    scores = scores - m
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


# In[31]:

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


# In[32]:

def _gen_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """
    Generates a [1, length, channels] timing signal consisting of sinusoids
    Adapted from:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """
    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = (
                    math.log(float(max_timescale) / float(min_timescale)) /
                    (float(num_timescales) - 1))
    inv_timescales = min_timescale * np.exp(
                    np.arange(num_timescales).astype(np.float) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)


    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]], 
                    'constant', constant_values=[0.0, 0.0])
    signal =  signal.reshape([1, length, channels])

    return torch.from_numpy(signal).type(torch.FloatTensor)


# In[33]:

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer, opt_step=0, opt_rate=0):
        self.optimizer = optimizer
        self._step = opt_step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = opt_rate
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        if(self._step == 500):
            print('Current lr of optimizer')
            print(rate)
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor *             (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

# In[43]:

def evaluate(model):
    val_iter = BatchIterator(val_set, batch_size, 1)
    loss_fn = torch.nn.CosineEmbeddingLoss()
    reg_lss = torch.nn.CosineEmbeddingLoss()
    flags = torch.ones(model.batch_size, device=CTX_DEVICE)
    count = 0
    total_loss = 0
    #model.eval()
    with torch.no_grad():
        while(not val_iter.has_ended()):
            batch_data, batch_masks = next(val_iter)
            batch_sen = [sequence[0] for sequence in batch_data]
            state_tensor = torch.tensor(batch_sen, dtype=torch.long, device=CTX_DEVICE)
            mask_tensor = torch.tensor(batch_masks, dtype=torch.float32, device=CTX_DEVICE)
            y_hat = np.array([sequence[1] for sequence in batch_data])
            y_hat_tensor = torch.tensor(y_hat, dtype=torch.float32, device=CTX_DEVICE)

            #Pass through model
            output_embs, attn_scores = model(state_tensor, mask_tensor)
            attn_ids = torch.argmax(attn_scores,dim=1) #Calculate argmax
            attn_ids = state_tensor.gather(1, attn_ids.view(-1,1)).squeeze_(1).long() #Find word ids
            #print(attn_ids)
            attn_embs = weight_matrix[attn_ids] #Lookup embeds
            if(USE_GPU):
                attn_embs = attn_embs.cuda()
            
            loss = loss_fn(F.normalize(output_embs),F.normalize(y_hat_tensor),flags)
            loss += lambda2*reg_lss(F.normalize(output_embs),F.normalize(attn_embs),flags)
            total_loss += loss.item()
            count += 1
    return total_loss/count

# In[45]:

def train(model,data_iterator):
    #optimizer = optim.RMSprop(model.parameters(),lr=0.00001)
    #Custom optimizer with warmup steps and decay schedule as defined in the paper
    opt = NoamOpt(model.hidden_dim, opt_mf, opt_ws,
                  torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    if(RESUME_TRAINING):
        checkpoint = torch.load(MODEL_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt = NoamOpt(model.hidden_dim, opt_mf, opt_ws,
                      torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
                      checkpoint['opt_step'],
                      checkpoint['opt_rate']
                     )
        opt.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    loss_fn = torch.nn.CosineEmbeddingLoss()
    reg_lss = torch.nn.CosineEmbeddingLoss()
    flags = torch.ones(model.batch_size, device=CTX_DEVICE)
    losses=[]
    min_epoch_loss = 5
    counter = 0
    epoch_batch_count = 0
    epoch_loss = 0
    last_epoch = 1
    while(not data_iterator.has_ended()):
        
        #Flush gradient calculations
        #optimizer.zero_grad()
        opt.optimizer.zero_grad()
        
        batch_data, batch_masks = next(data_iterator)
        batch_sen = [sequence[0] for sequence in batch_data]
        state_tensor = torch.tensor(batch_sen, dtype=torch.long, device=CTX_DEVICE)
        mask_tensor = torch.tensor(batch_masks, dtype=torch.float32, device=CTX_DEVICE)
        y_hat = np.array([sequence[1] for sequence in batch_data])
        y_hat_tensor = torch.tensor(y_hat, dtype=torch.float32, device=CTX_DEVICE)
        
        #Pass through model
        output_embs, attn_scores = model(state_tensor, mask_tensor)
        attn_ids = torch.argmax(attn_scores,dim=1) #Calculate argmax
        attn_ids = state_tensor.gather(1, attn_ids.view(-1,1)).squeeze_(1).long() #Find word ids
        #print(attn_ids)
        attn_embs = weight_matrix[attn_ids] #Lookup embeds
        if(USE_GPU):
            attn_embs = attn_embs.cuda()

        loss = loss_fn(F.normalize(output_embs),F.normalize(y_hat_tensor),flags)
        loss += lambda2*reg_lss(F.normalize(output_embs),F.normalize(attn_embs),flags)

        epoch_loss += loss.item()
        epoch_batch_count += 1
        counter += 1
        if(epoch_batch_count % loss_freq == 0):
            train_loss = epoch_loss/epoch_batch_count
            val_loss = evaluate(model)
            print("\rEpoch : {}, sn: {} train loss: {:.4f}  val loss: {:.4f}".format(
                data_iterator.current_epoch, epoch_batch_count, train_loss, val_loss))
            
            #Write these results to the csv log file            
            log_writer.writerow([data_iterator.current_epoch, counter, train_loss, val_loss])
            
        if(last_epoch != data_iterator.current_epoch):
            #Save the model if loss has decreased
            avg_epoch_loss = epoch_loss/epoch_batch_count
            if(min_epoch_loss > avg_epoch_loss):
                print("Loss improved from {:.4f} to {:.4f}".format(
                    min_epoch_loss, avg_epoch_loss
                ))
                min_epoch_loss = avg_epoch_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.optimizer.state_dict(),
                    'opt_step':opt._step,
                    'opt_rate':opt._rate        
                }, MODEL_PATH)
                print("Last checkpoint at "+str(datetime.datetime.now()))
                
            #Reset epoch vars
            last_epoch = data_iterator.current_epoch
            epoch_batch_count = 0
            epoch_loss = 0


        loss.backward()
        #optimizer.step()
        opt.step()


weight_matrix = torch.FloatTensor(embedding_matrix, device=CTX_DEVICE)

batchIterator = BatchIterator(train_set, batch_size, epochs)


# In[47]:

#num_layers, weight_matrix, embedding_dim, hidden_dim, vocab_size, output_emb_dim, seq_len, dropout, num_heads, batch_size
model = DefinitionEncoder(num_blocks,weight_matrix, input_emb_dim, input_emb_dim,
                          output_emb_dim, max_len, global_dropout, num_attn_heads, batch_size)

if(USE_GPU):
    model = model.cuda()
# In[48]:

#Initialize model params
for p in model.parameters():
    if(p.dim() > 1 and p.shape[0] != vocab_len):
        nn.init.xavier_uniform_(p)

#model.init_emb_layer(weight_matrix)
print('Parameter initialization done')

print('Starting training...')
csv_log_file = open(CSV_LOG_FILE, mode='w')
log_writer = csv.writer(csv_log_file)
train(model,batchIterator)
csv_log_file.close()
print('Training complete')
