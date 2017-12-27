# -*- coding: utf-8 -*-
"""
This model has 3 parameters as input
- stateful = true 
the network is not shuffling the data, and is keeping the state of the last batch instead of resetting it.
this help the LSTM to learn oscillations which are longer than its sequence length

This setup requires the input of a batch input shape. the data fed need to be a multiple of the batch size
I therefore added a resizing of the data accomodate this requirement.

plot is the prediction of 1 step ahead


@author: Remo Tacchi
"""
import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt

from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import CuDNNLSTM
import matplotlib.pyplot as plt
from keras.models import Sequential
import pandas as pd
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()



def predict_sequence_full(model, data, window_size):
    #Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

nepochs=150
seq_len=45   
batchsi=512

f = pd.read_csv('detser2.csv')
f=np.array(f)[:,1]#[newaxis].T
rw=len(f)-seq_len-1
x,y= np.indices((rw,seq_len+1))
result=f[x+y]
############################################### removedplt.figure(2) for b atch normalization
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
result= scaler.fit_transform(result)
########################### differences from previous
resultp = []
resulti=np.zeros(len(result))
resulti[1:]=result[:-1,0]
resulti=resulti[:,np.newaxis]
resultp=np.hstack((resulti,result))

win1= resultp[:,:-1]
win2= resultp[:,1:]
result2=(win2-win1)￼
result4=np.cumsum(result2,axis=0)
scaler2 = MinMaxScaler(copy=True, feature_range=(0, 1))
result2= scaler2.fit_transform(result2)

########################### pct diff

#result3=(win2/win1 -1)
result3=(win2-win1)/(0.00000001+win1)
scaler3 = MinMaxScaler(copy=True, feature_range=(0, 1))
result3= scaler3.fit_transform(result3)
############################ diff cumulplt.figure(2)

#result4=np.cumsum(resubatch_input_shapelt2,axis=0)plt.figure(2)   
scaler4 = MinMaxScaler(copy=True, feature_range=(0, 1))
result4= scaler4.fit_transform(result4)

resall=np.stack((result ,result2,result4) ,axis=2)


"""
False
row = round(0.9 * resall.shape[1])
train = resall[:,:int(row), :]
np.random.shuffle(train)￼
x_train = train[:, :, :-1]
y_train = train[0, :, -1]plt.figure(2)   
plt.plot(yvc)
plt.plot(y_test)    

x_test = resall[:, int(row):, :-1]
y_test = resall[0,int(row):, -1]

x_train = np.reshape(x_train, (x_train.shape[1], x_train.shape[2], x_train.shape[0]))
x_test  = np.reshape(x_test , (x_test .shape[1], x_test .shape[2], x_test .shape[0]))
y_train = np.reshape(y_train , (y_train.shape[0], 1))
y_test = np.reshape(y_test , (y_test.shape[0], 1))

"""
row = round(0.9 * resall.shape[0]/batchsi)*batchsi
splitt=2.0/row*batchsi
train = resall[:int(row), :,:]
#np.random.shuffle(train)
x_train = train[:, :-1,:]
y_train = train[:, -1,0]
x_test = resall[int(row):, :-1,:]
y_test = resall[int(row):, -1,0]

dr=round(len(x_test)/batchsi)*batchsi
x_test=x_test[:int(dr)]
y_test=y_test[:int(dr)]

global_start_time=time.time()


print('>Loadingdata...')

#X_train,y_train,X_test,y_test=tsmultiinp.load_data('detser.csv',seq_len,True)

print('>DataLoaded.Compiling...')total
model=Sequential()
xshape = (x_train.shape[1],x_train.shape[2] )
#model.add(LSTM(input_shape=(xshape),output_dim=300,return_sequences=True))
model.add(CuDNNLSTM(100, return_sequences=True, batch_input_shape= (batchsi, x_train.shape[1],x_train.shape[2]), stateful=True))
#model.add(CuDNNLSTM(200, 150, 100,70,38, return_sequences=True, input_shape=xshape, batch_input_shape= (batchsi, x_train.shape[1],x_train.shape[2])))

#model.add(Dropout(0.02))

model.add(CuDNNLSTM(70,return_sequences=True, stateful=True, batch_input_shape= (batchsi, x_train.shape[1],x_train.shape[2])))

#model.add(Dropout(0.03))
#model.add(Dropout(0.2))
#model.add(CuDNNLSTM(45,return_sequences=True, stateful=True, batch_input_shape= (batchsi, x_train.shape[1],x_train.shape[2])))
#model.add(Dropout(0.03))batch_input_shape= (seq_len, x_train.shape[1],x_train.shape[2])
#model.add(Dropout(0.1))
#model.add(CuDNNLSTM(45,return_sequences=True, stateful=True, batch_input_shape= (batchsi, x_train.shape[1],x_train.shape[2])))
#model.add(Dropout(0.03))
#model.add(Dropout(0.1))
model.add(CuDNNLSTM(30,return_sequences=False, stateful=True, batch_input_shape= (batchsi, x_train.shape[1],x_train.shape[2])))
#model.add(Dropout(0.03))
#model.add(Dropout(0.1))
#model.add(Dropout(0.2))
model.add(Dense(output_dim=1))￼
#model.add(BatchNormalization())
#model.add(Dropout(0.03))
model.add(Activation("elu"))
#model.add(Activation("tanh"))

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
plot_results(predictions, y_test)





