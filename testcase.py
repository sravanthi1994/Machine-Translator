#Testing En- Fr

import numpy as np
import gensim.models as word2vec
model_sp = word2vec.Word2Vec.load('spanishwords')
model_en = word2vec.Word2Vec.load('englishwords')

data_sp = open('/Users/Sravanthi/Downloads/ML project/es-en/test.es','r').read()
data_sp = data_sp.split()

words_sp = list(set(data_sp))
data_size_sp,vocab_size_sp = len(data_sp),len(model_sp.wv['the'])
print('spanish file has %d words,%d features'%(data_size_sp,vocab_size_sp))

word_to_ix_sp = {}
ix_to_word_sp = {}


for w in words_sp:
    word_to_ix_sp[w] = model_sp.wv[w]
    ix_to_word_sp[tuple(model_sp.wv[w])] = w
print('Word2Vec Spanish unique words = %d, word2vector features = %d'%(len(word_to_ix_sp),len(model_sp.wv['the'])))



lines_en = open('/Users/Sravanthi/Downloads/ML project/es-en/test.en','r').readlines()
data_en = open('/Users/Sravanthi/Downloads/ML project/es-en/test.en','r').read()
data_en = data_en.split()
words_en = list(set(data_en))
data_size_en,vocab_size_en = len(data_en),len(model_sp.wv['la'])
print('English file has %d words,%d features'%(data_size_en,vocab_size_en))
word_to_ix_en = {}
ix_to_word_en = {}


for w in words_en:
    word_to_ix_en[w] = model_en.wv[w]
    ix_to_word_en[tuple(model_en.wv[w])] = w
print('Word2Vec English unique words =',len(word_to_ix_en))

print("Loading Weights:");

#hyperparameters which are same for both encoder and decoder
hidden_size = 100
learning_rate = 1e-1


#encoder weight parameters for english language
wxh_sp = np.random.randn(hidden_size,vocab_size_sp)*0.01
whh_sp = np.random.randn(hidden_size,hidden_size)*0.01
why_sp = np.random.randn(vocab_size_sp,hidden_size)*0.01
bh_sp = np.zeros((hidden_size,1))
by_sp = np.zeros((vocab_size_sp,1))


#decoder weight parameters for french language
wxh_en = np.random.randn(hidden_size,vocab_size_en)*0.01
whh_en = np.random.randn(hidden_size,hidden_size)*0.01
why_en = np.random.randn(vocab_size_en,hidden_size)*0.01
bh_en = np.zeros((hidden_size,1))
by_en = np.zeros((vocab_size_en,1))


## Loading Weights
data = np.load('weights.sp.npz')
wxh_sp = data['wxh_sp']
print(wxh_sp)
whh_sp = data['whh_sp']
why_sp = data['why_sp']
bh_sp = data['bh_sp']
by_sp = data['by_sp']

data = np.load('weights.en.npz')
wxh_en = data['wxh_en']
whh_en = data['whh_en']
why_en = data['why_en']
bh_en = data['bh_en']
by_en = data['by_en']

## End Load Weights
def cosine_dist(ix,y):
	return np.dot(np.asarray(ix).T,y)/(np.linalg.norm(np.asarray(ix))*np.linalg.norm(y))
def test(inputs, targets):
  xs, hs, ys = {}, {}, {}
  hs[-1] = np.zeros((hidden_size,1))
  # forward pass
  for t in range(len(inputs)):
    a = np.reshape(inputs[t],(300,1))
    xs[t] = np.zeros((vocab_size_sp,1)) # encode in 1-of-k representation
    xs[t] = np.copy(a)
    hs[t] = np.tanh(np.dot(wxh_sp, xs[t]) + np.dot(whh_sp, hs[t-1]) + bh_sp) # hidden state
    ys[t] = np.dot(why_sp, hs[t]) + by_sp 
  hprev = hs[len(inputs)-1]
  tem = ""
  ans = ""
  k = 0
  t = -1
  while k<=len(targets):
      x = np.zeros((vocab_size_en,1))
      if(t!=-1):
          x[t] = 1
      hprev = np.tanh(np.dot(wxh_en, x) + np.dot(whh_en, hprev) + bh_en)
      y = np.dot(why_en, hprev) + by_en # unnormalized log probabilities for next chars
      keys = np.zeros((len(ix_to_word_en)))
      mind = 100000.0
      min_key = tuple
      for ix in ix_to_word_en.keys():
          cosdist = (cosine_dist(ix,y))
          if(cosdist<mind):
                print(ix_to_word_sp[tuple(targets[k-1])],cosdist,ix_to_word_en[ix])
                mind = cosdist	
                minkey = ix
      tem = ix_to_word_en[minkey]
      k = k + 1
      ans = ans + " " +tem
  return ans

input_spglish = "Es "
curr_sp = input_spglish.split()
inputs_sp = [word_to_ix_sp[w] for w in curr_sp[0:len(curr_sp)-1]]
targets_sp = [word_to_ix_sp[w] for w in curr_sp[1:len(curr_sp)]]
output_words = test(inputs_sp,targets_sp)
print(output_words)
#print(cosine_dist(model_en.wv['i'],model_en.wv[output_words.split()[0]]))