#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import joblib


# In[13]:


def ordinal_encoder(input_val, feats): 
    feat_val = list(np.arange(len(feats)))
    feat_key = feats
    feat_dict = dict(zip(feat_key, feat_val))
    value = feat_dict[input_val]
    return value


# In[14]:


def get_prediction(data,model):
    return model.predict(data)

