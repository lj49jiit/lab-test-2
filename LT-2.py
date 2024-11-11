#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as mpy
import requests from bs4 import BeautifulSoup
import scipy as sp
import mongoclient from pymongo
import sklearn.model_selection import train_data
import minmaxscaler from skikit_learn
import recall_score,f1_score from sklearn.metrics

approval_data=pd.read_csv("C:\Users\22103328\Downloads\New Institutes - New Institutes - New Institutes - New Institutes.csv")

new_data=approval_data(column[0])

xy=pd.Dataframe(approval_data.data,columns=approval_data.features_names)

train_test_split= X_train,Y_train,X_test,Y_test



def missing:
    miss=0
    states_listt=approval.data
    for i in range(state_listt)
    if(states_list[i] != states_list[i+1]){
        miss=miss+1
    }
    


def states:
    count=0
    states_list=approval_data.states
    for i in range(state_list):
    if(states_list[i] != states_list[i+1]){
        count=count+1
    }
    print(state_list,count)

svm_model=fit(X_train,Y_train)
print(svm_model)
print(recall_score)
print(f1_score)
    
plt.plot(xy["target"])
plt.xlabel("region")
plt.ylabel("program")
plt.title("region vs program")
plt.show()


# In[ ]:




