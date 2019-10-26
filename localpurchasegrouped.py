import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

wk_dir = os.path.abspath('..')
df = pd.read_csv(wk_dir+'\\NC State Dining\\Data\\NCSUlocalpurchase.csv', header=1)
df = df.drop(df.index[-1])
df = df.rename(columns={' Sales $ ':'Sales'}, inplace=False)
df['Sales'] = df.Sales.astype(float)

df1 = df.groupby(['Product # (APN)', 'Product Desc # (APN)', 'PIM Brand', 'Location']).sum()

df1 = df1[['Cases', 'Sales']]

df1.to_excel(r'C:\Users\Justin\Documents\NC State\Fall 2019\NC State Dining\analysis.xlsx')