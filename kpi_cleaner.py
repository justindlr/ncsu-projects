import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

wk_dir = os.path.abspath('..')
df = pd.read_csv(wk_dir+'\diningData\KPI 2019-20 Oct 18, 2019 - Temp 2019-20.csv', header=2)
#df = df.drop(['3', 'Director','GM /Manager', 'Asst Manager'], axis =1)
df = df[['Location', 'Unnamed: 27', 'Unnamed: 28']]
df = df.drop([df.index[1], df.index[2], df.index[3], df.index[4], df.index[5], df.index[6], df.index[7],
              df.index[8], df.index[10], df.index[11], df.index[12], df.index[14], 
              df.index[24], df.index[27], df.index[28], df.index[29], df.index[35], df.index[40], 
              df.index[43], df.index[49], df.index[52], df.index[53], df.index[54], df.index[55], 
              df.index[57], df.index[62], df.index[63], df.index[64], df.index[65]])
df = df.dropna(how='all')
df = df.rename(columns={'Unnamed: 27':'Total Sales'}, inplace=False)
df = df.rename(columns={'Unnamed: 28':'SPLH'}, inplace=False)
df = df.drop([df.index[0]])

df.to_excel(r'C:\Users\Justin\Documents\NC State\Fall 2019\NC State Dining\oct182019sales.xlsx')