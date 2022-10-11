# import packages
import pandas as pd
from pathlib import Path

#%% read in data
path = r'C:\Users\Justin\OneDrive - North Carolina State University\Documents\NC State\IAA Python\Data\lol_matches' # use your path
files = Path(path).glob('*.csv') 

dfs = list()
for f in files:
    data = pd.read_csv(f)
    data['file'] = f.stem
    dfs.append(data)
    
df = pd.concat(dfs, ignore_index=True)

#%% clean up data (only want to analyze number of games per league per split or year)

lol_games  = df[['league', 'split', 'year']]
lol_games["period"] = lol_games["split"].astype(str) + ' ' +lol_games["year"].astype(str)


lol_games = lol_games.dropna() # get rid of any null split (these are international events)

lol_games = lol_games[['league', 'period']] # clean up dataframe


# redefine EU LCS as LEC
lol_games.loc[(lol_games["league"] == "EU LCS"), "league"] = "LEC"

lol_games.loc[(lol_games["league"] == "NA LCS"), "league"] = "LCS"



# only observe the 4 major regions: NA, EU, Korea, and China

lol_clean = lol_games.loc[(lol_games['league'] == 'LCS') |(lol_games['league'] == 'LEC') | 
                          (lol_games['league'] == 'LCK') | (lol_games['league'] == 'LPL')]

#%% do a groupby of league per year
lol_games_gb = lol_clean.groupby(['league', 'period']).size()
lol_games_gb = lol_games_gb/12 # 12 observations per match (10 players + red team + blue team summary stats)

lol_games_df = lol_games_gb.to_frame()
lol_games_df = lol_games_df.reset_index()
#%% viz, want to plot per split, the number of games split by region

lol_games_df.plot(figsize= (10,10))

#%% export and then look at in tableau
lol_games_df.to_excel('lol_games_df.xlsx')