from gensim.models import word2vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

temp_sp = open('/Users/Sravanthi/Downloads/ML project/es-en/europarl-v7.es-en.es','r').readlines()
sentences_sp = []
for i in range(len(temp_sp)):
    words_sp = temp_sp[i].split()
    sentences_sp.append(words_sp)
# Set values for various parameters
num_features_sp = 300    # Word vector dimensionality                      
min_word_count_sp = 1   # Minimum word count                        
num_workers_sp = 4       # Number of threads to run in parallel
context_sp = 10          # Context window size                                                                                    
downsampling_sp = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
print( "Training model...")
model_sp = word2vec.Word2Vec(sentences_sp, workers=num_workers_sp, \
            size=num_features_sp, min_count = min_word_count_sp, \
            window = context_sp, sample = downsampling_sp)

# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model_sp.init_sims(replace=True)

# It can be helpful to create a meaningful model name and 
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "spanishwords"
model_sp.save(model_name)


#english words word2vec
temp_en = open('/Users/Sravanthi/Downloads/ML project/es-en/europarl-v7.es-en.en','r').readlines()
sentences_en = []
for i in range(len(temp_en)):
    words_en = temp_en[i].split()
    sentences_en.append(words_en)
# Set values for various parameters
num_features_en = 300    # Word vector dimensionality                      
min_word_count_en = 1   # Minimum word count                        
num_workers_en = 4       # Number of threads to run in parallel
context_en = 10          # Context window size                                                                                    
downsampling_en = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
print( "Training model...")
model_en = word2vec.Word2Vec(sentences_en, workers=num_workers_en, \
            size=num_features_en, min_count = min_word_count_en, \
            window = context_en, sample = downsampling_en)

# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model_en.init_sims(replace=True)

# It can be helpful to create a meaningful model name and 
# save the model for later use. You can load it later using Word2Vec.load()
model_name1 = "englishwords"
model_en.save(model_name1)

