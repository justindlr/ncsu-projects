import ReformatCritData
import numpy as np
from keras.layers import Input, Dense, GaussianDropout, concatenate, Flatten, Add
from keras.models import Model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pandas as pd
import os
from keras.callbacks import EarlyStopping
wk_dir = os.path.abspath('..')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  
import ast

#%% Perform reformat operation on entire dataframe

datavector = []
ithvector = []

attacks = 10
df = pd.read_csv(wk_dir+'\Data\crits.csv')


def crit_maker(attacks, crit_rate):
    reformat_data = ReformatCritData.sequence(attacks, crit_rate)
    for i in range(1, attacks):
          x_list = []
          y_list = []
          for i in range(len(reformat_data) - attacks):
            x = reformat_data[i:i+attacks]
            y = reformat_data[i+attacks]
            x_list.append(x)
            y_list.append(y)
    true_x_list = [item[0] for item in x_list]
    true_y_list = [item[0] for item in y_list]
    data_reformat = np.array(true_x_list)
    ith_reformat = np.array(true_y_list)
    datavector.append(data_reformat)
    ithvector.append(ith_reformat)
    return data_reformat, ith_reformat

for i in range(1, 10):
    iteration = i*10
    crit_rate = 'crit_%d' % (iteration)
    crit_maker(attacks, crit_rate)

df_all = pd.DataFrame.from_records(datavector)
df_all = df_all.T
df_all = df_all.rename(columns={0: 'crit_10', 1: 'crit_20',2: 'crit_30',
                       3: 'crit_40', 4: 'crit_50', 5: 'crit_60', 6: 'crit_70',
                       7: 'crit_80', 8: 'crit_90'})

ith_all = pd.DataFrame.from_records(ithvector)
ith_all = ith_all.T
ith_all = ith_all.rename(columns={0: 'crit_10', 1: 'crit_20',2: 'crit_30',
                       3: 'crit_40', 4: 'crit_50', 5: 'crit_60', 6: 'crit_70',
                       7: 'crit_80', 8: 'crit_90'})
#%% add binary crit rate, makes tuple
df_all_copy = pd.DataFrame.copy(df_all)
for col_index, col_name in enumerate(df_all.columns): 
   array_to_append = [0] * len(df_all.columns) 
   array_to_append[col_index] = 1 
   df_all_copy[col_name] = df_all[col_name].map(lambda x: (x, array_to_append))

#%% Seperate the tuple

tuple_array = df_all_copy.values
tuple_array = tuple_array.flatten()

crit_sequences = df_all_copy.applymap(lambda x:x[0])
crit_binary = df_all_copy.applymap(lambda x:x[1])


crit_array = crit_sequences.values 
temp = crit_array
crit_array = crit_array.flatten()


binary_array = crit_binary.values
binary_array = binary_array.flatten()


ith_array = ith_all.values
ith_array = ith_array.flatten()
#%% Training data
X_train, X_test, y_train, y_test = train_test_split(tuple_array, ith_array, test_size=0.2)                                          
                                                
tuple_seq = np.array([x[0] for x in tuple_array])
tuple_binary = np.array([x[1] for x in tuple_array])
in1_train = np.array([x[0] for x in X_train])
in2_train = np.array([x[1] for x in X_train])
in1_test = np.array([x[0] for x in X_test])
in2_test = np.array([x[1] for x in X_test])

print(in1_train.shape, in2_train.shape, y_train.shape)
print(in1_test.shape, in2_test.shape, y_test.shape)
#%%
def get_model():
    input_layer = Input(shape=(attacks,), name='crit_sequence')
    input_layer2 = Input(shape=(9,), name = 'crit_rate')
    hidden_layer = Dense(10, activation='relu', name='dense1')(input_layer)
    hidden_layer2 = Dense(10, activation='relu', name='dense2')(input_layer2) 
    hidden_layer = GaussianDropout(.2)(hidden_layer)
    #hidden_layer = Dense(25, activation='relu', name='dense3')(hidden_layer)
    #hidden_layer2 = Dense(25, activation='relu', name='dense4')(hidden_layer2)
    #hidden_layer = Dense(50, activation='relu', name='dense4')(hidden_layer)
    #hidden_layer = GaussianDropout(.2)(hidden_layer)
    added_layer = Add()([hidden_layer,hidden_layer2])
    output_layer = Dense(1, activation='relu', name='dense3')(added_layer)
    model = Model(inputs=[input_layer, input_layer2], outputs=output_layer)
    
    return model

model = get_model()
accuracy = model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

es = EarlyStopping(monitor='loss', mode='min', verbose=1)

history = model.fit([in1_train, in2_train],
                y_train,
                batch_size=64,
                epochs=50, 
                validation_data=[[in1_test, in2_test],y_test], verbose=0, callbacks=[es])
_, train_acc = model.evaluate([in1_train, in2_train], y_train, verbose=0)
_, test_acc = model.evaluate([in1_test, in2_test], y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

#%% Plot 
plt.figure(figsize=(10, 10))
plt.title('Model loss')
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(loss, label='loss')
plt.plot(val_loss, color ='r', label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot model accuracy
plt.figure(figsize=(10, 10))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'], color='r')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['acc', 'val_acc'], loc='lower right')
plt.show()
#%% Predictions
model_prediction = model.predict([tuple_seq, tuple_binary])
rounded_prediction = np.matrix.round(model_prediction)

predicted_sequence_of_attacks = np.hstack((tuple_seq, rounded_prediction))
prediction_df = pd.DataFrame.from_records(predicted_sequence_of_attacks)
binary_index = list(tuple_binary)


prediction_df['binary_index'] = pd.Series(binary_index)
prediction_df.index = prediction_df["binary_index"]
prediction_df = prediction_df.drop(columns='binary_index')

#%%
data_df = pd.DataFrame(zipped_data, columns = ['sequence', 'ith'])

unique_df = data_df.groupby(['sequence']).agg(lambda x:x.value_counts().index[0])
# Set unique sequences back to array

sequence = unique_df.index.tolist()
sequence2 = [ast.literal_eval(i) for i in sequence]
true_sequence = np.array([np.array(xi) for xi in sequence2])

ith = unique_df['ith']
#%% Perform reformat operation on entire dataframe

datavector = []
ithvector = []

attacks = 10
df = pd.read_csv(wk_dir+'\Data\crits.csv')


def crit_maker(attacks, crit_rate):
    reformat_data = ReformatCritData.sequence(attacks, crit_rate)
    for i in range(1, attacks):
          x_list = []
          y_list = []
          for i in range(len(reformat_data) - attacks):
            x = reformat_data[i:i+attacks]
            y = reformat_data[i+attacks]
            x_list.append(x)
            y_list.append(y)
    true_x_list = [item[0] for item in x_list]
    true_y_list = [item[0] for item in y_list]
    data_reformat = np.array(true_x_list)
    ith_reformat = np.array(true_y_list)
    datavector.append(data_reformat)
    ithvector.append(ith_reformat)
    return data_reformat, ith_reformat

for i in range(1, 10):
    iteration = i*10
    crit_rate = 'crit_%d' % (iteration)
    crit_maker(attacks, crit_rate)

df_all = pd.DataFrame.from_records(datavector)
df_all = df_all.T
df_all = df_all.rename(columns={0: 'crit_10', 1: 'crit_20',2: 'crit_30',
                       3: 'crit_40', 4: 'crit_50', 5: 'crit_60', 6: 'crit_70',
                       7: 'crit_80', 8: 'crit_90'})

ith_all = pd.DataFrame.from_records(ithvector)
ith_all = ith_all.T
ith_all = ith_all.rename(columns={0: 'crit_10', 1: 'crit_20',2: 'crit_30',
                       3: 'crit_40', 4: 'crit_50', 5: 'crit_60', 6: 'crit_70',
                       7: 'crit_80', 8: 'crit_90'})