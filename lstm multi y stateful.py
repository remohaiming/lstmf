# -*- coding: utf-8 -*-
"""
LSTM predicting many y (number of y predicted = predl)
this is stateful, enabling the network to learn structures longer than the sequence length.

@author: Remo Tacchi
"""
import os
import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import CuDNNLSTM
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler


from keras.models import Sequential
import pandas as pd
from keras import optimizers
from sklearn.preprocessing import StandardScaler
from keras.layers import BatchNormalization


    
nepochs=2000
seq_len=50
predl=30
batchsi=512



f = pd.read_csv('detser.csv')
f=np.array(f)[:,1]#[newaxis].T500
rw=len(f)-seq_len-predl
x,y= np.indices((rw,seq_len+predl))
result=f[x+y]
############################################### removed for b atch normalization
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
result= scaler.fit_transform(result)
########################### differences from previous


row = round(0.9 * result.shape[0]/batchsi)*batchsi
splitt=2.0/row*batchsi
train = result[:int(row), :]
#np.random.shuffle(train)

x_train = train[:, :-predl]
y_train = train[:, -predl:]
x_test = result[int(row):, :-predl]
y_test = result[int(row):, -predl:]

dr=round(len(x_test)/batchsi)*batchsi
x_test=x_test[:int(dr)]
y_test=y_test[:int(dr)]



x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1)) 

global_start_time=time.time()


print('>Loadingdata...')

#X_train,y_train,X_test,y_test=tsmultiinp.load_data('detser.csv',seq_len,True)

print('>DataLoaded.Compiling...')
model=Sequential()
xshape = (x_train.shape[1],1 )
#model.add(LSTM(input_round(len(predictions))-1shape=(xshape),output_dim=300,return_sequences=Tmodel=Sequential()')rue))

model.add(CuDNNLSTM(20, return_sequences=True, batch_input_shape= (batchsi, x_train.shape[1],1), stateful=True))
#model.add(Dropout(0.2))

#model.add(CuDNNLSTM(20,return_sequences=True, stateful=True, batch_input_shape= (batchsi, x_train.shape[1],1)))

model.add(CuDNNLSTM(20,return_sequences=False, stateful=True, batch_input_shape= (batchsi, x_train.shape[1],1)))

model.add(Dense(output_dim=predl))

model.add(Activation("elu"))

optimizers.Nadam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-08,schedule_decay=0.0002)
start=time.time()
model.compile(loss="mse",optimizer="Adam")
print(">CompilationTime:",time.time()-start)





#model.fit(x_train, y_train, batch_size=batchsi, epochs=nepochs, validation_split=splitt, verbose=1, shuffle=False)
checkpointer = ModelCheckpoint(filepath='/home/total/LSTMtimeser/weights2.hdf5', verbose=1, save_best_only=True)


for i in range(nepochs):
    model.fit(x_train, y_train, batch_size=batchsi, epochs=1, validation_split=splitt, verbose=2, shuffle=False, callbacks=[checkpointer])
    model.reset_states()
    print "Step=" , i

model.load_weights("/home/total/LSTMtimeser/weights2.hdf5")

predictions = model.predict(x_test, batch_size=batchsi)

stps=round(len(predictions)/predl)-1

for i in np.arange(0,len(predictions), predl):
    plt.plot(np.arange(i,i+predl),predictions[i])

for i in np.arange(0,len(predictions), predl):
    plt.plot(np.arange(i,i+predl),y_test[i])

"""



model.add(CuDNNLSTM(100, return_sequences=True, input_shape=xshape))
#model.add(Dropout(0.015))
#model.add(CuDNNLSTM(70,return_sequences=True))
#model.add(Dropout(0.03))
#model.add(Dropout(0.2))
#model.add(CuDNNLSTM(100,return_sequences=True))
#model.add(Dropout(0.03))model=Sequential()')
#model.add(Dropout(0.1))
#model.add(CuDNNLSTM(70,return_sequences=True))
#model.add(Dropout(0.03))
#model.add(Dropout(0.1))
model.add(CuDNNLSTM(70,return_sequences=True))
model.add(CuDNNLSTM(30,return_sequences=False))

#model.add(Dropout(0.03))
#model.add(Dropout(0.1))
#model.add(Dropout(0.2))
model.add(Dense(output_dim=predl))
#model.add(BatchNormalization())
#model.add(Dropout(0.2))
model.add(Activation("elu"))
#model.add(Activation("tanh"))

optimizers.Nadam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-08,schedule_decay=0.0002)
start=time.time()
model.compile(loss="mse",optimizer="Adam")
print(">CompilationTime:",time.time()-start)

checkpointer = ModelCheckpoint(filepath='/home/total/LSTMtimeser/weights.hdf5', verbose=1, save_best_only=True)


model.fit(x_train, y_train, batch_size=512, epochs=epochs, validation_split=0.05, verbose=2, callbacks=[checkpointer])

model.load_weights("/home/total/LSTMtimeser/weights.hdf5")

predictions = model.predict(x_test)
#plot_results(predictions, y_test)

stps=round(len(predictions)/predl)-1

for i in np.arange(0,len(predictions), predl):
    plt.plot(np.arange(i,i+predl),predictions[i])

for i in np.arange(0,len(predictions), predl):
    plt.plot(np.arange(i,i+predl),y_test[i])

"""
