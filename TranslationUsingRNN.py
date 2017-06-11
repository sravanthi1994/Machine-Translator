import numpy as np

lines_en = open('eng.txt','r').readlines()
data_en = open('eng.txt','r').read()
data_en = data_en.split()
words_en = list(set(data_en))
data_size_en,vocab_size_en = len(data_en),len(words_en)
print('english file has %d words,%d unique'%(data_size_en,vocab_size_en))
word_to_ix_en = { ch:i for i,ch in enumerate(words_en)}
ix_to_word_en = { i:ch for i,ch in enumerate(words_en)}

#taking the german data and creating mappings for one hot encoding
lines_gr = open('ger.txt','r').readlines()
data_gr = open('ger.txt','r').read()
data_gr = data_gr.split()
words_gr = list(set(data_gr))
data_size_gr,vocab_size_gr = len(data_gr),len(words_gr)
print('german file has %d words,%d unique'%(data_size_gr,vocab_size_gr))
word_to_ix_gr = { ch:i for i,ch in enumerate(words_gr)}
ix_to_word_gr = { i:ch for i,ch in enumerate(words_gr)}

#Number of sentences in training data
num = len(lines_en)
num = len(lines_gr)
#hyperparameters which are same for both encoder and decoder
hidden_size = 100
learning_rate = 1e-1

#encoder weight parameters for english language
wxh_en = np.random.randn(hidden_size,vocab_size_en)*0.01
whh_en = np.random.randn(hidden_size,hidden_size)*0.01
why_en = np.random.randn(vocab_size_en,hidden_size)*0.01
bh_en = np.zeros((hidden_size,1))
by_en = np.zeros((vocab_size_en,1))

#decoder weight parameters for german language
wxh_gr = np.random.randn(hidden_size,vocab_size_gr)*0.01
whh_gr = np.random.randn(hidden_size,hidden_size)*0.01
why_gr = np.random.randn(vocab_size_gr,hidden_size)*0.01
bh_gr = np.zeros((hidden_size,1))
by_gr = np.zeros((vocab_size_gr,1))

def trainencoder(inputs, targets, hprev):
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  # forward pass
  for t in range(len(inputs)):
    xs[t] = np.zeros((vocab_size_en,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    hs[t] = np.tanh(np.dot(wxh_en, xs[t]) + np.dot(whh_en, hs[t-1]) + bh_en) # hidden state
    ys[t] = np.dot(why_en, hs[t]) + by_en # unnormalized log probabilities for next words
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next words

  dWxh, dWhh, dWhy = np.zeros_like(wxh_en), np.zeros_like(whh_en), np.zeros_like(why_en)
  dbh, dby = np.zeros_like(bh_en), np.zeros_like(by_en)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(range(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(why_en.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(whh_en.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def traindecoder(inputs, targets, hprev):
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  # forward pass
  for t in range(len(inputs)):
    xs[t] = np.zeros((vocab_size_gr,1)) # encode in 1-of-k representation
    if(inputs[t]!=-1):
        xs[t][inputs[t]] = 1
    hs[t] = np.tanh(np.dot(wxh_gr, xs[t]) + np.dot(whh_gr, hs[t-1]) + bh_gr) # hidden state
    ys[t] = np.dot(why_gr, hs[t]) + by_gr # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars

  dWxh, dWhh, dWhy = np.zeros_like(wxh_gr), np.zeros_like(whh_gr), np.zeros_like(why_gr)
  dbh, dby = np.zeros_like(bh_gr), np.zeros_like(by_gr)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(range(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(why_gr.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(whh_gr.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def test(inputs, targets):
  xs, hs, ys = {}, {}, {}
  hs[-1] = np.zeros((hidden_size,1))
  # forward pass
  for t in range(len(inputs)):
    xs[t] = np.zeros((vocab_size_en,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    hs[t] = np.tanh(np.dot(wxh_en, xs[t]) + np.dot(whh_en, hs[t-1]) + bh_en) # hidden state
    ys[t] = np.dot(why_en, hs[t]) + by_en 
  hprev = hs[len(inputs)-1]
  tem = ""
  ans = ""
  k = 0
  t = -1
  while k<10:
      x = np.zeros((vocab_size_gr,1))
      if(t!=-1):
          x[t] = 1
      hprev = np.tanh(np.dot(wxh_gr, x) + np.dot(whh_gr, hprev) + bh_gr)
      y = np.dot(why_gr, hprev) + by_gr # unnormalized log probabilities for next chars
      pr = np.exp(y) / np.sum(np.exp(y))
      maxi = pr[0][0]
      for i in range(len(words_gr)):
          if pr[i][0]>=maxi:
              maxi = pr[i][0]
              t = i
      tem = ix_to_word_gr[t]
      k = k + 1
      ans = ans + " " +tem
  return ans;
    
n,p = 1,0
mwxh_en,mwhh_en,mwhy_en,mbh_en,mby_en = np.zeros_like(wxh_en),np.zeros_like(whh_en),np.zeros_like(why_en),np.zeros_like(bh_en),np.zeros_like(by_en)
mwxh_gr,mwhh_gr,mwhy_gr,mbh_gr,mby_gr = np.zeros_like(wxh_gr),np.zeros_like(whh_gr),np.zeros_like(why_gr),np.zeros_like(bh_gr),np.zeros_like(by_gr)


while n!=2000:
    curr_en = lines_en[p].split()
    inputs_en = [word_to_ix_en[ch] for ch in curr_en[0:len(curr_en)-1]]
    targets_en = [word_to_ix_en[ch] for ch in curr_en[1:len(curr_en)]]
    
    curr_gr = lines_gr[p].split()
    inputs_gr=[-1]
    temp = [word_to_ix_gr[ch] for ch in curr_gr[0:len(curr_gr)-1]]
    inputs_gr.extend(temp)
    targets_gr = [word_to_ix_gr[ch] for ch in curr_gr[0:len(curr_gr)]]

    hprev = np.zeros((hidden_size,1))
    dwxh_en,dwhh_en,dwhy_en,dbh_en,dby_en,hprev = trainencoder(inputs_en,targets_en,hprev)
    dwxh_gr,dwhh_gr,dwhy_gr,dbh_gr,dby_gr,hprev = traindecoder(inputs_gr,targets_gr,hprev)
    
    p += 1 # move sentence pointer
    if p >= num:
        p = 0
        print('training...iteration:%d'%(n))
        #input_english = input('english:')
        input_english = "iron cement is a ready for use paste which is laid as a fillet by putty knife or finger in the mould edges ( corners ) of the steel ingot mould ."
        curr_en = input_english.split()
        inputs_en = [word_to_ix_en[ch] for ch in curr_en[0:len(curr_en)-1]]
        targets_en = [word_to_ix_en[ch] for ch in curr_en[1:len(curr_en)]]
        output_words = test(inputs_en,targets_en)
        print(output_words)  
        n = n + 1
    for param_en, dparam_en, mem_en in zip([wxh_en, whh_en, why_en, bh_en, by_en], 
                                [dwxh_en, dwhh_en, dwhy_en, dbh_en, dby_en], 
                                [mwxh_en, mwhh_en, mwhy_en, mbh_en, mby_en]):
      mem_en += dparam_en * dparam_en
      param_en += -learning_rate * dparam_en / np.sqrt(mem_en + 1e-8) # adagrad update
    





    for param_gr, dparam_gr, mem_gr in zip([wxh_gr, whh_gr, why_gr, bh_gr, by_gr], 
                                [dwxh_gr, dwhh_gr, dwhy_gr, dbh_gr, dby_gr], 
                                [mwxh_gr, mwhh_gr, mwhy_gr, mbh_gr, mby_gr]):
      mem_gr += dparam_gr * dparam_gr
      param_gr += -learning_rate * dparam_gr / np.sqrt(mem_gr + 1e-8) # adagrad update
 # iteration counter 



                 
    

