from xgboost import XGBClassifier
import numpy as np

import texts
import functions
a = functions.nlp_preproc(texts.TIME_INPUTS)
b = functions.nlp_preproc(texts.ABLE_INPUTS)
c = functions.nlp_preproc(texts.PROGRAMMING_INPUTS)
d = functions.nlp_preproc(texts.ML_INPUTS)
e = functions.nlp_preproc(texts.WEATHER_INPUTS)
res = a+b+c+d+e
y = np.concatenate([np.zeros(len(a)),
               np.ones(len(b)),
               2*np.ones(len(c)),
               3*np.ones(len(d)),
                   4*np.ones(len(e))])
x = np.zeros(len(texts.word_index)).reshape(1,len(texts.word_index))

for i in range(len(res)):
    x = np.vstack([x,functions.tokenize_and_to_matrix([res[i]])])
    
x = x[1:]

shuffler = np.random.permutation(x.shape[0])
x = x[shuffler]
y = y[shuffler]

xgb = XGBClassifier(objective='multi:softmax',booster='gblinear',n_estimators=10)
xgb.fit(x,y)