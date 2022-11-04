import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
import ast
import re
from collections import Counter
from statsmodels.graphics.tsaplots import plot_acf
from matplotlib import pyplot
os.chdir(r'C:\Users\Justin\Documents\NC State\dlab\Projects\Crit Analysis')
wk_dir =  os.path.abspath('..')
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import ReformatCritData
#import tensorflow as tf
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#%% Full Crit Rules
attacks = 3
#crit_rate = 'crit_70'
all_crit_df = {}

for iterator in range(1, 10):
    crit_rate = 'crit_%d0' % iterator # parse through each of the crit columns
    combo=attacks - 2
    combos = 8* combo 
    
    
    num_crit_rate = re.findall(r'\d+', crit_rate)
    num_crit_rate = ' '.join(map(str, num_crit_rate)) 
    num_crit_rate = float(num_crit_rate) *.01
    reformat_data = ReformatCritData.sequence(attacks, crit_rate)
    df = pd.read_csv(wk_dir+'\Data\crits.csv')
    # Reformat data (random or actual)
    x_list = []
    y_list = []
    for i in range(len(reformat_data) - attacks):
        x = reformat_data[i:i+attacks]
        y = reformat_data[i+attacks]
        x_list.append(x)
        y_list.append(y)
    
    true_x_list = [item[0] for item in x_list]
    true_y_list = [item[0] for item in y_list]
    
    reformatted_data = np.array(true_x_list)
    ith_data = np.array(true_y_list)
    
    # Find the unique sequences
    a = ((reformatted_data))
    a = a.tolist()
    a = list(map(str, a))
    zipped_data = zip(a, ith_data)
    
    data_df = pd.DataFrame(zipped_data, columns = ['sequence', 'ith'])
    
    unique_df = data_df.groupby(['sequence']).agg(lambda x:x.value_counts().index[0])
    
    # Set unique sequences back to array
    sequence = unique_df.index.tolist()
    sequence2 = [ast.literal_eval(i) for i in sequence]
    true_sequence = np.array([np.array(xi) for xi in sequence2])
    
    true_ith = unique_df['ith']
    # Optimal Acc
    merged = pd.merge(data_df, unique_df, right_index=True, left_on='sequence')
    acc = merged['ith_x'] == merged['ith_y']
    true_optimal_acc = sum(acc)/len(merged)
    
    #conduct t-test with mean
    # accuracy refers to the number of times that unique data is correct 
    
    data_unique_df = data_df.groupby(['sequence']).agg(["sum","count"])
    data_unique_df = data_df.groupby(['sequence']).agg(["mean","count"])
    data_unique_df.columns = ["crit_rate","count"]
    data_unique_df["accuracy"] = data_unique_df["crit_rate"].apply(lambda x: x if x > 0.5 else 1-x)
    data_unique_df["correct_preds"] = data_unique_df["accuracy"]*data_unique_df["count"]
    # Plotting sequences of attack data
    counts = map(tuple, reformat_data)
    final_count = Counter(counts)
    keys = list(final_count.keys())
    
    values = list(final_count.values())
    count_df = pd.DataFrame(
        {'Sequence': keys,
         'Value': values})
    
    
    
    # Plot split bar graph by ith_data
    count_df = count_df.sort_values(by=['Sequence'])
    crit_prop_df = pd.crosstab(data_df['sequence'],data_df['ith']).apply(lambda r: r/r.sum(), axis=1)
    
    prop0 = crit_prop_df[0]
    prop1 = crit_prop_df[1]
    
    zeros = count_df[['Value']] * prop0[1]
    ones = count_df[['Value']] * prop1[1]
    
    prop_df = pd.DataFrame(
            {'Sequence': keys,
             'Zeros' : zeros['Value'],
             'Ones' : ones['Value']}
            )
    
    prop_df["Zeros"] = prop_df["Zeros"].astype(int)
    prop_df["Ones"] = prop_df["Ones"].astype(int)
    
    
    #Crit rules
    # count = number of times seq recorded
    # # of Crits is # of crits after the sequence
    data_crit_rule_df = crit_prop_df.copy()
    new_count_df = count_df.set_index('Sequence')
    rule_counts = count_df['Value']
    rule_counts = list(rule_counts)
    data_crit_rule_df['count'] = rule_counts
    data_crit_rule_df['# of Crits'] = data_crit_rule_df['count'] * data_crit_rule_df[1]
    data_crit_rule_df['Crit %'] = data_crit_rule_df['# of Crits'] / data_crit_rule_df['count'] 
    data_crit_rule_df = data_crit_rule_df.drop(columns=[0, 1])
    data_crit_rate = data_crit_rule_df['Crit %']
    data_index =  data_crit_rate.index
    data_counts = data_crit_rule_df['count']
    
    # crit rules with random data
    
    crit_means = []
    crit_counts = []
    for j in range(1, 1001):
        rand_data = np.random.choice([0, 1], size=(2700,), p=[1-num_crit_rate, num_crit_rate])
        rand_data = pd.Series(rand_data) 
        def sequence(attacks, data):
            """
            Loops through all attacks and creates arrays showing all sequences of k crits in a row
                followed by the next crit.
            """
            list_of_attacks = []
            crit_rate = rand_data.tolist()
            for i in range(len(crit_rate)-(attacks-1)):
                hold = crit_rate[i:i+attacks]
                list_of_attacks.append(hold)
            return list_of_attacks
        
        reformat_data = sequence(attacks, rand_data)
        # Reformat data (random or actual)
        x_list = []
        y_list = []
        for i in range(len(reformat_data) - attacks):
            x = reformat_data[i:i+attacks]
            y = reformat_data[i+attacks]
            x_list.append(x)
            y_list.append(y)
        
        true_x_list = [item[0] for item in x_list]
        true_y_list = [item[0] for item in y_list]
        
        reformatted_data = np.array(true_x_list)
        ith_data = np.array(true_y_list)
        
        # Find the unique sequences
        a = ((reformatted_data))
        a = a.tolist()
        a = list(map(str, a))
        zipped_data = zip(a, ith_data)
        
        data_df = pd.DataFrame(zipped_data, columns = ['sequence', 'ith'])
        
        unique_df = data_df.groupby(['sequence']).agg(lambda x:x.value_counts().index[0])
        
        # Set unique sequences back to array
        sequence = unique_df.index.tolist()
        sequence2 = [ast.literal_eval(i) for i in sequence]
        true_sequence = np.array([np.array(xi) for xi in sequence2])
        
        true_ith = unique_df['ith']
        # Optimal Acc
        merged = pd.merge(data_df, unique_df, right_index=True, left_on='sequence')
        acc = merged['ith_x'] == merged['ith_y']
        true_optimal_acc = sum(acc)/len(merged)
        
        #conduct t-test with mean
        # accuracy refers to the number of times that unique data is correct 
        
        data_unique_df = data_df.groupby(['sequence']).agg(["sum","count"])
        data_unique_df = data_df.groupby(['sequence']).agg(["mean","count"])
        data_unique_df.columns = ["crit_rate","count"]
        data_unique_df["accuracy"] = data_unique_df["crit_rate"].apply(lambda x: x if x > 0.5 else 1-x)
        data_unique_df["correct_preds"] = data_unique_df["accuracy"]*data_unique_df["count"]
        #%Plotting sequences of attack data
        counts = map(tuple, reformat_data)
        final_count = Counter(counts)
        keys = list(final_count.keys())
        
        values = list(final_count.values())
        count_df = pd.DataFrame(
            {'Sequence': keys,
             'Value': values})
        
        # Plot split bar graph by ith_data
        count_df = count_df.sort_values(by=['Sequence'])
        crit_prop_df = pd.crosstab(data_df['sequence'],data_df['ith']).apply(lambda r: r/r.sum(), axis=1)
        
        prop0 = crit_prop_df[0]
        prop1 = crit_prop_df[1]
        
        zeros = count_df[['Value']] * prop0[1]
        ones = count_df[['Value']] * prop1[1]
        
        prop_df = pd.DataFrame(
                {'Sequence': keys,
                 'Zeros' : zeros['Value'],
                 'Ones' : ones['Value']}
                )
        
        prop_df["Zeros"] = prop_df["Zeros"].astype(int)
        prop_df["Ones"] = prop_df["Ones"].astype(int)
        
     
        crit_rule_df = crit_prop_df.copy()
        hold_df = crit_rule_df.copy()
        rule_counts = count_df['Value']
        rule_counts = list(rule_counts)
        try:
            hold_df['Rnd Counts'] = rule_counts
            hold_df = data_crit_rule_df.join(hold_df, how='left')
            hold_df = hold_df.drop(['count', '# of Crits', 'Crit %'], axis=1)
        
            crit_rule_df['Rnd Counts'] = hold_df['Rnd Counts']
            crit_rule_df['# of Crits'] = crit_rule_df['Rnd Counts'] * crit_rule_df[1]
            crit_rule_df['Crit %'] = crit_rule_df['# of Crits'] / crit_rule_df['Rnd Counts']
            crit_means.append(crit_rule_df['Crit %'])
            crit_counts.append(crit_rule_df['Rnd Counts'])
        except:
            pass
    
    
    crit_means = [s for s in crit_means if len(s) == len(crit_rule_df)]
    
    arr_crit_means = np.array(crit_means)
    crit_means = pd.DataFrame(crit_means)
    crit_means_df = crit_means.T
    
    placeholder_df = data_crit_rule_df.join(crit_means_df,  how='left', rsuffix='rnd')
    placeholder_df = placeholder_df.drop(['count', '# of Crits', 'Crit %'], axis=1).fillna(0)
                       
    #note: each row is one run that fit the crit_means len(crit_rule_df) criteria
    
    arr_crit_means = placeholder_df.to_numpy()
    arr_crit_counts = np.array(crit_counts)
    rand_crit_means = arr_crit_means.mean(axis=1)
    #rand_crit_means = np.nanmean(arr_crit_means, axis=0)
    #rand_crit_std = arr_crit_means.std(axis=0) 
    rand_crit_counts = arr_crit_counts.mean(axis=0)
    
    #create crit_rules_df with summary statistics
    
    arr_crit_5th = []
    for i in range(len(data_crit_rate)):
        bool_crit_percentile = np.percentile(arr_crit_means[:,i],5 )
        arr_crit_5th.append(bool_crit_percentile)
    arr_crit_95th = []
    for i in range(len(data_crit_rate)):
        bool_crit_percentile = np.percentile(arr_crit_means[:,i], 95)
        arr_crit_95th.append(bool_crit_percentile)
    arr_crit_p_null = []
    for i in range(len(data_crit_rate)):
        bool_crit_percentile = arr_crit_means[:, i] < data_crit_rate[i]
        arr_crit_p_null.append(bool_crit_percentile)
        
    arr_crit_p_null = np.array(arr_crit_p_null)
    arr_crit_5th = np.array(arr_crit_5th)
    arr_crit_95th = np.array(arr_crit_95th)
    #
    
    arr_crit_p_null_means = arr_crit_p_null.mean(axis=1)
    
    alpha=.05
    
    #index = crit_rule_df.index
    
    avg_crit_rule_df = pd.DataFrame({'Sequence':data_index, 'Random Crit %':rand_crit_means, 
                                     'Rnd Counts': rand_crit_counts,
                                     'Data Crit %':data_crit_rate, 'Data Counts': data_counts})
    avg_crit_rule_df = avg_crit_rule_df.set_index('Sequence')
    avg_crit_rule_df['Crit% Diff'] = avg_crit_rule_df['Random Crit %'] - avg_crit_rule_df['Data Crit %']
    
    #avg_crit_rule_df['t_stat'] = avg_crit_rule_df['Crit% Diff'] / (avg_crit_rule_df['Crit Mean Std'] 
    #                / (np.sqrt(avg_crit_rule_df['Counts'])))
    
    #avg_crit_rule_df['deg_free'] = avg_crit_rule_df['Counts'] - 1
    #avg_crit_rule_df['critical t'] =  stats.t.ppf(1-.05,avg_crit_rule_df['Counts'])
    
    #avg_crit_rule_df['p_val'] = (1 - t.cdf(abs(avg_crit_rule_df['t_stat']), avg_crit_rule_df['deg_free']))*2
    #avg_crit_rule_df['stat_sig'] = avg_crit_rule_df['p_val'].apply(lambda x: 1 if x < alpha else 0)
    
    avg_crit_rule_df['PCTL of DataCrit'] = arr_crit_p_null_means
    avg_crit_rule_df['5th PCTL(Rnd)'] = arr_crit_5th
    avg_crit_rule_df['95th PCTL(Rnd)'] = arr_crit_95th
    all_crit_df['crit_{0}0'.format(iterator)] = avg_crit_rule_df
#%% to csv
all_crit_df['crit_10'].to_csv(r'C:\Users\Justin\Documents\NC State\dLab\Projects\Crit Analysis\crit_rules_csv\crit_10.csv')
all_crit_df['crit_20'].to_csv(r'C:\Users\Justin\Documents\NC State\dLab\Projects\Crit Analysis\crit_rules_csv\crit_20.csv')
all_crit_df['crit_30'].to_csv(r'C:\Users\Justin\Documents\NC State\dLab\Projects\Crit Analysis\crit_rules_csv\crit_30.csv')
all_crit_df['crit_40'].to_csv(r'C:\Users\Justin\Documents\NC State\dLab\Projects\Crit Analysis\crit_rules_csv\crit_40.csv')
all_crit_df['crit_50'].to_csv(r'C:\Users\Justin\Documents\NC State\dLab\Projects\Crit Analysis\crit_rules_csv\crit_50.csv')
all_crit_df['crit_60'].to_csv(r'C:\Users\Justin\Documents\NC State\dLab\Projects\Crit Analysis\crit_rules_csv\crit_60.csv')
all_crit_df['crit_70'].to_csv(r'C:\Users\Justin\Documents\NC State\dLab\Projects\Crit Analysis\crit_rules_csv\crit_70.csv')
all_crit_df['crit_80'].to_csv(r'C:\Users\Justin\Documents\NC State\dLab\Projects\Crit Analysis\crit_rules_csv\crit_80.csv')
all_crit_df['crit_90'].to_csv(r'C:\Users\Justin\Documents\NC State\dLab\Projects\Crit Analysis\crit_rules_csv\crit_90.csv')
    
#%% Histogram  Plot
obs_seq = 0 #change number to change sequence

fig= plt.figure(figsize=(20,10))
hist = plt.hist(arr_crit_means[:,obs_seq], color='lightblue') 
plt.axvline(data_crit_rate[obs_seq], color='red', linestyle='dashed', linewidth=3, label='Data Crit Rate')
plt.axvline(rand_crit_means[obs_seq], linestyle='dashed', linewidth=3, color='green', label = 'Mean Random Crit Rate')
plt.axvline((np.percentile(arr_crit_means, 100-(alpha*100))), color='purple', label='95th percentile')
plt.axvline((np.percentile(arr_crit_means, (alpha*100))), color='purple', label='5th percentile')
plt.ylabel('Counts')
plt.xlabel('Crit Rate')
plt.grid()
plt.legend()
plt.title('Distribution of Random Crit Rate at %d%% Crit Rate, Attack Sequence: %s ' 
          % (num_crit_rate*100, data_crit_rate.index[obs_seq]))
#%% Plot the all data points

rand_data = np.random.choice([0, 1], size=(2700,), p=[1-num_crit_rate, num_crit_rate]) #
rand_data = pd.Series(rand_data) 

fig, axs = plt.subplots(2, figsize=(750,15), sharex= True)
plt.margins(0,0)
x = range(len(df))
axs[0].plot(x, df['crit_50'], marker='o', color='r')
axs[0].set_ylabel('Data Crits')
axs[1].plot(x, rand_data, marker = 'o')
axs[1].set_ylabel('Random Crits')

#%% Print autocorrelation for all crit rates
for i in df:
    s = df[i]
    print(s.autocorr(lag=1), i)
   
    

#s = df[crit_rate]
#print(s.autocorr(lag=4))
#print(rand_data.autocorr(lag=3))
#%% Autocorrelation Plots
crit_rate = 'crit_10' # redefine crit_rate to study
fig, ax= plt.subplots(figsize=(15,10))
plot_acf(df[crit_rate], alpha=.05, ax=ax)
plt.title('Autocorrelation of %s Data' % crit_rate)
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
pyplot.show()

#%% Hothand plot (using sequences)
avg_crit_rule_df['Data Crit %'].plot(legend=True, color='r')
avg_crit_rule_df['Random Crit %'].plot(figsize=(10, 7), ylim=(0, 1),
                title = ('Crit Rate After Sequence of Attacks at %.2f Crit Rate' % num_crit_rate), 
                legend = True, color = 'b', 
                label = ('Nominal Crit Rate'))

plt.show()
#%% Hothand plot (similar to the one in the artcle)
hothand = pd.read_csv(wk_dir+'\Data\hothandtable.csv')
hothand = hothand.set_index(hothand.columns[0], inplace=False)
hothand = hothand.multiply(100)

#hothand = hothand.drop(hothand.columns[[0,1, 2, -1, -2, -3]], axis=1) # to drop columns
#hothand = hothand.fillna(0)

ax = hothand.plot(figsize=(10, 10), title=('Critical Strike Rate'))
ax.set_xlabel('Observed Crit Rate')
ax.set_ylabel('Nominal Crit Rate')
#ax.legend(['After 6 non-crits', 'After 5 non-crits', 'After 4 non-crits', 'After 3 non-crits',
#           'After 2 non-crits', 'After 1 non-crit', 'Overall', 'After 1 crit', 'After 2 crits',
#           'After 3 crits', 'After 4 crits', 'After 5 crits', 'After 6 crits'])

#%% attempt at bayes
#p(A) = nominal
#p(B) = observed
hothandA = pd.DataFrame(1, index=hothand.index, columns=hothand.columns)
hothandA = hothandA.mul(hothandA.index, axis=0)
hothandB = hothand
hothandBA = (hothandA * hothandB) / hothandA # this one may be the right one actually
hothandAB = (hothandBA * hothandA) / hothandB
#%% testing
import math
lam = 1/2
answer = []
for i in range(0, 11):
    r_sum = ((math.e**-lam)*(lam**i)) / (math.factorial(i))
    print(r_sum)
    answer.append(r_sum)
    
truth = sum(answer)
