# -*- coding: utf-8 -*-
"""IMDBMovies_Review.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/10Z8t6_ZQgaR9F6y6NiBopGKFbYqdK-va

# **End to End Deep Learning Project using Simple RNN**
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,SimpleRNN,Embedding

#load imdb dataset
max_features=10000 #vocabulary size
(X_train,y_train),(X_test,y_test)=imdb.load_data(num_words=max_features)
print(f'Training data shape:{X_train.shape},Training labels shape:{y_train.shape}')
print(f'Testing data shape:{X_test.shape},Testing labels shape:{y_test.shape}')

X_train[0]

#inspect sample review and itd label
sample_review=X_train[0]
sample_labels=y_train[0]
print(sample_review,sample_labels)

#mapping of words to index to back to word
word_index=imdb.get_word_index()
word_index

reverse_wordindex={value: key for key,value in word_index.items()}
reverse_wordindex

decoded_review=' '.join([reverse_wordindex.get(i-3,'?')for i in sample_review])
decoded_review

max_len=500
X_train=sequence.pad_sequences(X_train,maxlen=max_len)
X_test=sequence.pad_sequences(X_test,maxlen=max_len)

X_train[0]

#train simple rnn
model=Sequential()
model.add(Embedding(max_features,128,input_length=max_len)) # embedding layers
model.add(SimpleRNN(128,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.summary()

from tensorflow.keras.callbacks import EarlyStopping
earlystopping=EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)
earlystopping

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history=model.fit(
    X_train,y_train,epochs=10,batch_size=32,
    validation_split=0.2,
    callbacks=[earlystopping]
)

#saving model
model.save('simple_rnn_imdb.h5')