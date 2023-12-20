import numpy as np
import pandas as pd


df = pd.read_csv('gender_classification_train.csv')
df = df.sample(frac=1.0)

n = df.shape[0]
split_point = int(n*0.7)
training_set = df.iloc[:split_point,:]
eval_set = df.iloc[split_point:,:]

training_set.to_csv('gender_classification_training_set.csv',index=False)
eval_set.to_csv('gender_classification_eval_set.csv',index=False)
