## importing libraries

import keras
from keras.preprocessing import sequence
from keras.preprocessing.text import one_hot
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras import optimizers

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words= 5000)

words2idx= imdb.get_word_index()

index_n = 3
words2idx = {w:(ind + index_n) for w, ind in words2idx.items()}

words2idx["<PAD>"] = 0
words2idx["<START>"] = 1
words2idx["<UNK>"] = 2

idx2words = {index:word for word, index in words2idx.items()}

## padding missing words with 0
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen = max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen = max_review_length)

## Building the model

model = Sequential()
model.add(Embedding(5000, 32, input_length = 500)) # inputs for Embedding
                                                    # max_words, 32 and 
                                                    # max_review_length
                                     
model.add(LSTM(100))
model.add(Dense(1, activation = "sigmoid"))
optimizer = optimizers.Adam(lr = 0.01)

model.summary()

"""
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 500, 32)           160000    
_________________________________________________________________
lstm_1 (LSTM)                (None, 100)               53200     
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 101       
=================================================================
Total params: 213,301
Trainable params: 213,301
Non-trainable params: 0
_________________________________________________________________
"""

model.compile(optimizer = optimizer,
              loss = "binary_crossentropy",
              metrics = ["accuracy"])

model.fit(X_train, y_train,
          batch_size = 64,
          epochs = 2,
          validation_data = (X_test, y_test))


# Evaluating the model
acc = model.evaluate(X_test, y_test, verbose = 1)
print("Accuracy: {:.3f}".format(acc[1]))

## Accuracy: 0.862

# predicting raw text

bad = "it seemed ok at first but then how i feel about the film changed completely"
good = "i really liked the movie and had fun"

# convert srings to list

bad_list = [w for w in bad.split()]
good_list = [w for w in good.split()]

bad_idx = [words2idx[w] for w in bad_list]
good_idx= [words2idx[w] for w in good_list]

bad_padded = sequence.pad_sequences([bad_idx], 500)
good_padded = sequence.pad_sequences([good_idx], 500)

score_bad = model.predict(bad_padded)
print("Review:{}  Sentiment:{:.3f}".format(bad, score_bad[0][0]))

score_good = model.predict(good_padded)
print("Review:{}  Sentiment:{:.3f}".format(good, score_good[0][0]))

"""
Review:it seemed ok at first but then how i feel about the film changed completely  Sentiment:0.403
Review:i really liked the movie and had fun  Sentiment:0.970
"""


# Example with one_hot

docs = ["As previously mentioned, the provided scripts are used",
        "to train a LSTM recurrent neural network on the Large Movie Review Dataset dataset",
        "While the dataset is public, in this tutorial we provide a copy of",
        "the dataset that has previously been preprocessed according to the",
        "needs of this LSTM implementation. Running the code provided in this",
        "tutorial will automatically download the data to the local directory",
        "in order to use your own data, please use a (preprocessing script)",
        "provided as a part of this tutorial.",
        "Once the model is trained, you can test it with your own corpus using",
        " the word-index dictionary (imdb.dict.pkl.gz) provided as a part of",
        " this tutorial"]

vocab_size = 200

docs_onehotted = [one_hot(doc, vocab_size) for doc in docs]
max_length = 150
docs_padded = sequence.pad_sequences(docs_onehotted, max_length)

"""
array([[  0,   0,   0, ..., 121,  94,   9],
       [  0,   0,   0, ...,  13, 163, 163],
       [  0,   0,   0, ...,  11,  38,  10],
       ...,
       [  0,   0,   0, ..., 169,  58,  93],
       [  0,   0,   0, ...,  11,  23,  10],
       [  0,   0,   0, ...,   0,  19, 122]])
"""
## in one_hot capital letters, punctuation are not problem. No need for 
## preprocessing. Includes basic punctuation, new lines and tabs

## keras.preprocessing.text.one_hot(text, n, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~  
##  ', lower=True, split=' ')

## better than one_hot is Tokenizer but it is not helpful here
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(docs)

tokenizer.word_counts  # 
tokenizer.document_count # how many documents in docs
tokenizer.word_index # dictionary words2idx
tokenizer.word_docs # how many occurences of each word in docs

tokenizer.texts_to_matrix(docs, mode = "count") # similar to CountVectorizer



