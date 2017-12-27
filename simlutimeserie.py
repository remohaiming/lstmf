# -*- coding: utf-8 -*-
"""
this creates the time series used in the LSTM test.

it gives the possibility to test data without noise (sim=s6), or with noise (sim=s7)

the file created is called by all the other models. make sure to change the path accordingly in your computer


@author: Remo Tacchi
"""

import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
sln=25000

ind=np.arange(sln, dtype='float')

a=np.repeat([0,1],20)
b=np.resize(np.repeat([0,1],20), sln)

#sim= np.sin(ind/8)*1.5 + np.sin(0.25+ ind/21)*5 + np.sin(0.6+ind/235)*7 + ind/250 +np.resize(np.repeat([0,1],30), sln)*-8
sim= np.sin(ind/8)*1.5 + np.sin(0.25+ ind/21)*5+ np.sin(0.6+ind/235)*7 + ind/250 +np.resize(np.repeat([0,1],85), sln)*-8
#pl.plot(sim[:3500])
#pl.plot(sim)

sim =pd.DataFrame(sim)
sim.columns = ['atari']


#sim.to_csv('/home/total/LSTMtimeser/detser.csv')
steps= 5000
f, dta=pl.subplots(6,sharex=True)

s1= np.sin(ind/8)*1.5
dta[0].plot(ind[:steps], s1[:steps])

s2= np.sin(0.25+ ind/21)*5
dta[1].plot(ind[:steps], s2[:steps])

s3= np.sin(0.6+ind/235)*7
dta[2].plot(ind[:steps], s3[:steps])

s4= ind/250
dta[3].plot(ind[:steps], s4[:steps])

s5= np.resize(np.repeat([0,1],85), sln)*-8
dta[4].plot(ind[:steps], s5[:steps])

s6= np.sin(ind/8)*1.5 + np.sin(0.25+ ind/21)*5+ np.sin(0.6+ind/235)*7 + ind/250 +np.resize(np.repeat([0,1],85), sln)*-8
dta[5].plot(ind[:steps], s6[:steps])


"""
steps= 5000
f, dta=pl.subplots(5,sharex=True)

s1= np.sin(ind/8)*1.5
dta[0].plot(ind[:steps], s1[:steps])

s2= np.sin(0.25+ ind/21)*5
dta[1].plot(ind[:steps], s2[:steps])

s3= np.sin(0.6+ind/235)*7
dta[2].plot(ind[:steps], s3[:steps])

s4= ind/250
dta[3].plot(ind[:steps], s4[:steps])

s5= np.resize(np.repeat([0,1],85), sln)*-8
dta[4].plot(ind[:steps], s5[:steps])
"""
#s7= (np.sin(ind/8)*1.5 + np.sin(0.25+ ind/21)*5+ np.sin(0.6+ind/235)*7 + ind/250 +np.resize(np.repeat([0,1],85), sln)*-8)* (1+  (np.random.random()-0.5)/5)

#dta[5].plot(ind[:steps], s7[:steps])

####################### add random noise


sdiff=np.zeros(sln)
sdiff[1:]= s6[1:]-s6[:-1]
pl.figure(4)
noise= (np.random.rand(len(ind)) - 0.5)*np.std(sdiff)*2

#s6= np.sin(ind/8)*1.5 + np.sin(0.25+ ind/21)*5+ np.sin(0.6+ind/235)*7 + ind/250 +np.resize(np.repeat([0,1],85), sln)*-8
pl.plot(ind[:steps], s6[:steps])
#pl.figure(5)
s7= np.add(s6, noise)
pl.plot(ind[:steps], s7[:steps])

sim= s7
#pl.plot(sim[:3500])
pl.figure(6)
pl.plot(sim)

sim =pd.DataFrame(sim)
sim.columns = ['atari']


sim.to_csv('/home/total/LSTMtimeser/detser.csv')


