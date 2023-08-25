#!/usr/bin/env python
# coding: utf-8

# This project was heavily inspired by "Maximizing Precision of Hit Predictions in Baseball" by Jason Clavelli and Joel Gottsegen

# In[265]:


import pandas, numpy
import seaborn as sns
import requests
import lxml
import bs4
#from lxml import html
from urllib import request, response, error, parse
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re, os


# In[2]:


# Scraper from GitHub by Ben Kite, thank you so much

## This is the best place to get started.
## This function simply takes a url and provides the ids
## from the html tables that the code provided here can access.
## Using findTables is great for determining options for the
## pullTable function for the tableID argument.
def findTables(url):
    res = requests.get(url)
    ## The next two lines get around the issue with comments breaking the parsing.
    comm = re.compile("<!--|-->")
    soup = bs4.BeautifulSoup(comm.sub("", res.text), 'html')
    divs = soup.findAll('div', id = "content")
    divs = divs[0].findAll("div", id=re.compile("^all"))
    ids = []
    for div in divs:
        searchme = str(div.findAll("table"))
        x = searchme[searchme.find("id=") + 3: searchme.find(">")]
        x = x.replace("\"", "")
        if len(x) > 0:
            ids.append(x)
    return(ids)
## For example:
## findTables("http://www.baseball-reference.com/teams/KCR/2016.shtml")


## Pulls a single table from a url provided by the user.
## The desired table should be specified by tableID.
## This function is used in all functions that do more complicated pulls.
def pullTable(url, tableID):
    res = requests.get(url)
    ## Work around comments
    comm = re.compile("<!--|-->")
    soup = bs4.BeautifulSoup(comm.sub("", res.text), 'html')
    tables = soup.findAll('table', id = tableID)
    data_rows = tables[0].findAll('tr')
    data_header = tables[0].findAll('thead')
    data_header = data_header[0].findAll("tr")
    data_header = data_header[0].findAll("th")
    game_data = [[td.getText() for td in data_rows[i].findAll(['th','td'])]
        for i in range(len(data_rows))
        ]
    data = pandas.DataFrame(game_data)
    header = []
    for i in range(len(data.columns)):
        header.append(data_header[i].getText())
    data.columns = header
    data = data.loc[data[header[0]] != header[0]]
    data = data.reset_index(drop = True)
    return(data)
## For example:
## url = "http://www.baseball-reference.com/teams/KCR/2016.shtml"
## pullTable(url, "team_batting")



## Pulls game level data for team and year provided.
## The team provided must be a three-character abbreviation:
## 'ATL', 'ARI', 'BAL', 'BOS', 'CHC', 'CHW', 'CIN', 'CLE', 'COL', 'DET',
## 'KCR', 'HOU', 'LAA', 'LAD', 'MIA', 'MIL', 'MIN', 'NYM', 'NYY', 'OAK',
## 'PHI', 'PIT', 'SDP', 'SEA', 'SFG', 'STL', 'TBR', 'TEX', 'TOR', 'WSN'
def pullGameData (team, year):
    url = "http://www.baseball-reference.com/teams/" + team + "/" + str(year) + "-schedule-scores.shtml"
    ## Let's funnel this work into the pullTable function
    dat = pullTable(url, "team_schedule")
    dates = dat["Date"]
    ndates = []
    for d in dates:
        month = d.split(" ")[1]
        day = d.split(" ")[2]
        day = day.zfill(2)
        mapping = {"Mar": "03", "Apr": "04", "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
                   "Sep": "09", "Oct": "10", "Nov":"11"}
        m = mapping[month]
        ndates.append(str(year) + m + day)
    uni, counts = numpy.unique(ndates, return_counts = True)
    ndates = []
    for t in range(len(counts)):
        ux = uni[t]
        cx = counts[t]
        if cx == 1:
            ndates.append(ux + "0")
        else:
            for i in range(int(cx)):
                ii = i + 1
                ndates.append(ux + str(ii))
    dat["Date"] = ndates
    dat.rename(columns = {dat.columns[4] : "Location"}, inplace = True)
    homegame = []
    for g in dat["Location"]:
        homegame.append(g == "")
    dat["HomeGame"] = homegame
    return(dat)


## Pulls data summarizing the season performance of all players on the
## team provided for the given year.
## The table type argument must be one of five possibilities:
## "team_batting"
## "team_pitching"
## "standard_fielding"
## "players_value_batting"
## "players_value_pitching"
def pullPlayerData (team, year, tabletype):
    url = "http://www.baseball-reference.com/teams/" + team + "/" + str(year) + ".shtml"
    data = pullTable(url, tabletype)
    data = data[data.Name.notnull()]
    data = data.reset_index(drop = True)
    names = data.columns
    for c in range(0, len(names)):
        replacement = []
        if type (data.loc[0][c]) == str:
            k = names[c]
            for i in range(0, len(data[k])):
                p = data.loc[i][c]
                xx = re.sub("[#@*&^%$!]", "", p)
                xx = xx.replace("\xa0", "_")
                xx = xx.replace(" ", "_")
                replacement.append(xx)
            data[k] = replacement
    data["Team"] = team
    data["Year"] = year
    return(data)


## This is used later to append integers to games on the same date to
## separate them.
def Quantify (x):
    out = []
    for i in x:
        if len(i) < 1:
            out.append(None)
        else:
            out.append(float(i))
    return(out)


## Pulls box score data from a game provided in the gameInfo input
## This is meant to be run by the pullBoxScores function below.
def gameFinder (gameInfo):
    teamNames = {"KCR":"KCA",
                 "CHW":"CHA",
                 "CHC":"CHN",
                 "LAD":"LAN",
                 "NYM":"NYN",
                 "NYY":"NYA",
                 "SDP":"SDN",
                 "SFG":"SFN",
                 "STL":"SLN",
                 "TBR":"TBA",
                 "WSN":"WAS",
                 "LAA":"ANA"}
    battingNames = {"ATL":"AtlantaBravesbatting",
                    "ARI":"ArizonaDiamondbacksbatting",
                    "BAL":"BaltimoreOriolesbatting",
                    "BOS":"BostonRedSoxbatting",
                    "CHC":"ChicagoCubsbatting",
                    "CHW":"ChicagoWhiteSoxbatting",
                    "CIN":"CincinnatiRedsbatting",
                    "CLE":"ClevelandIndiansbatting",
                    "COL":"ColoradoRockiesbatting",
                    "DET":"DetroitTigersbatting",
                    "KCR":"KansasCityRoyalsbatting",
                    "HOU":"HoustonAstrosbatting",
                    "LAA":"AnaheimAngelsbatting",
                    "LAD":"LosAngelesDodgersbatting",
                    "MIA":"MiamiMarlinsbatting",
                    "MIL":"MilwaukeeBrewersbatting",
                    "MIN":"MinnesotaTwinsbatting",
                    "NYM":"NewYorkMetsbatting",
                    "NYY":"NewYorkYankeesbatting",
                    "OAK":"OaklandAthleticsbatting",
                    "PHI":"PhiladelphiaPhilliesbatting",
                    "PIT":"PittsburghPiratesbatting",
                    "SDP":"SanDiegoPadresbatting",
                    "SEA":"SeattleMarinersbatting",
                    "SFG":"SanFranciscoGiantsbatting",
                    "STL":"StLouisCardinalsbatting",
                    "TBR":"TampaBayRaysbatting",
                    "TEX":"TexasRangersbatting",
                    "TOR":"TorontoBlueJaysbatting",
                    "WSN":"WashingtonNationalsbatting"}
    date = gameInfo["Date"]
    home = gameInfo["HomeGame"]
    if home == False:
        opp = gameInfo["Opp"]
        if opp in teamNames:
            opp = teamNames[opp]
        url = "http://www.baseball-reference.com/boxes/" + opp + "/" + opp + str(date) + ".shtml"
    else:
        team = gameInfo["Tm"]
        if team in teamNames:
            team = teamNames[team]
        url = "http://www.baseball-reference.com/boxes/" + team + "/" + team + str(date) + ".shtml"
    battingInfo = battingNames[gameInfo["Tm"]]
    data = pullTable(url, battingInfo)
    names = []
    for i in data["Batting"]:
        if len(i) > 0:
            xx = (i.split(" ")[0] + "_" + i.split(" ")[1])
            xx = xx.replace("\xa0", "")
            names.append(xx)
        else:
            names.append("NA")
    data["Name"] = names
    data["Date"] = date
    data["HomeGame"] = home
    data = data[data.Name != "NA"]
    for d in data:
        if d not in ["Batting", "Name", "Details", "Date", "HomeGame"]:
            tmp = Quantify(data[d])
            data[d] = tmp
    data = data[data["AB"] > 0]
    return(data)


## Pulls all of the boxscores for a team in a given year.
## The directory argument is used to specify where to save the .csv
## If overwrite is True, an existing file with the same name will be overwritten.
def pullBoxscores (team, year, directory, overwrite = True):
    if not os.path.exists(directory):
        os.makedirs(directory)
    if overwrite == False:
        if os.path.exists(directory + team + ".csv"):
            return("This already exists!")
    dat = pullGameData(team, year)
    DatDict = dict()
    for r in range(len(dat)):
        inputs = dat.loc[r]
        try:
            DatDict[r] = gameFinder(inputs)
        except IndexError:
            pass
    playerGameData = pandas.concat(DatDict)
    playerGameData.reset_index(inplace = True)
    playerGameData = playerGameData.rename(columns = {"level_0": "Game", "level_1": "BatPos"})
    playerGameData.to_csv(directory + team + "_" + str(year) + ".csv")


## This is an internal function to pullPlaybyPlay
def PlayByPlay (gameInfo):
    teamNames = {"KCR":"KCA",
                 "CHW":"CHA",
                 "CHC":"CHN",
                 "LAD":"LAN",
                 "NYM":"NYN",
                 "NYY":"NYA",
                 "SDP":"SDN",
                 "SFG":"SFN",
                 "STL":"SLN",
                 "TBR":"TBA",
                 "WSN":"WAS",
                 "LAA":"ANA"}
    oteam = gameInfo["Tm"]
    date = gameInfo["Date"]
    home = gameInfo["HomeGame"]
    if home == 0:
        team = gameInfo["Opp"]
        opp = gameInfo["Tm"]
        if opp in teamNames:
            opp = teamNames[opp]
    else:
        team = gameInfo["Tm"]
        opp = gameInfo["Opp"]
        if team in teamNames:
            team = teamNames[team]
    url = "http://www.baseball-reference.com/boxes/" + team + "/" + team + str(date) + ".shtml"
    dat = pullTable(url, "play_by_play")
    dat = dat.loc[dat["Batter"].notnull()]
    dat = dat.loc[dat["Play Description"].notnull()]
    dat["Date"] = date
    dat["Hteam"] = team
    dat["Ateam"] = opp
    pteam = []
    pteams = numpy.unique(dat["@Bat"])
    for d in dat["@Bat"]:
        if d == pteams[0]:
            pteam.append(pteams[1])
        else:
            pteam.append(pteams[0])
    dat["Pteam"] = pteam
    if gameInfo["R"] > gameInfo["RA"]:
        winner = oteam
    else:
        winner = gameInfo["Opp"]
    dat["Winner"] = winner
    return(dat)


## Pulls all of the play by play tables for a team for a given year.
## Output is the name of the .csv file you want to save.  I force a
## file to be saved here because the function takes a while to run.
def pullPlaybyPlay (team, year, output, check = False):
    dat = pullGameData(team, year)
    dat = dat[dat.Time == dat.Time] ## Only pull games that have ended
    if check:
        olddat = pandas.read_csv(output)
        dates = numpy.unique(olddat.Date)
        mostrecent = numpy.max(dates)
        dat.Date = dat.Date.astype("int")
        dat = dat.loc[dat.Date > mostrecent]
        dat.reset_index(inplace = True)
        dat = dat.loc[dat.Time == dat.Time]
    DatDict = dict()
    for r in range(len(dat)):
        inputs = dat.loc[r]
        try:
            DatDict[r] = PlayByPlay(inputs)
        except IndexError:
            pass
    if len(DatDict) == 0:
        return("No new games to be added!")
    bdat = pandas.concat(DatDict)
    bdat["Hteam"] = team
    names = []
    for i in bdat["Batter"]:
        if len(i) > 0:
            xx = i
            xx = xx.replace("\xa0", "")
            names.append(xx)
        else:
            names.append("NA")
    bdat["BatterName"] = names
    ## These rules attempt to sort out different play outcomes by
    ## searching the text in the "Play Description" variable.
    bdat["out"] = (bdat["Play Description"].str.contains("out")) | (bdat["Play Description"].str.contains("Play")) | (bdat["Play Description"].str.contains("Flyball")) | (bdat["Play Description"].str.contains("Popfly"))
    bdat["hbp"] = bdat["Play Description"].str.startswith("Hit")
    bdat["walk"] = (bdat["Play Description"].str.contains("Walk"))
    bdat["stolenB"] = bdat["Play Description"].str.contains("Steal")
    bdat["wild"] = bdat["Play Description"].str.startswith("Wild") | bdat["Play Description"].str.contains("Passed")
    bdat["error"] = bdat["Play Description"].str.contains("Reached on")
    bdat["pick"] = bdat["Play Description"].str.contains("Picked")
    bdat["balk"] = bdat["Play Description"].str.contains("Balk")
    bdat["interference"] = bdat["Play Description"].str.contains("Reached on Interference")
    bdat["sacrifice"] = bdat["Play Description"].str.contains("Sacrifice")
    bdat["ab"] = (bdat["walk"] == False) & (bdat["sacrifice"] == False) & (bdat["interference"] == False) & (bdat["stolenB"] == False) & (bdat["wild"] == False) & (bdat["hbp"] == False) & (bdat["pick"] == False) & (bdat["balk"] == False)
    bdat["hit"] =  (bdat["walk"] == False) & (bdat["out"] == False) & (bdat["stolenB"] == False) & (bdat["error"] == False) & (bdat["ab"] == True)
    if check:
        if len(olddat) > 0:
            bdat = olddat.append(bdat)
            bdat.reset_index(inplace = True, drop = True)
    bdat.to_csv(output)
    return(bdat)


## This pulls information about which hand a pitcher throws with.  I
## made this solely to allow pitcher handedness to be used as a
## variable in models.
def pullPitcherData (team, year):
    url = "http://www.baseball-reference.com/teams/" + team + "/" + str(year) + ".shtml"
    data = pullTable(url, "team_pitching")
    data = data[data.Name.notnull()]
    data = data[data.Rk.notnull()]
    data = data[data.G != "162"]
    data = data.reset_index(drop = True)
    data["Team"] = team
    data["Year"] = year
    data["LeftHanded"] = data["Name"].str.contains("\\*")
    names = data.columns
    for c in range(0, len(names)):
        replacement = []
        if type (data.loc[0][c]) == str:
            k = names[c]
            for i in range(0, len(data[k])):
                p = data.loc[i][c]
                xx = re.sub("[#@&*^%$!]", "", p)
                xx = xx.replace("\xa0", "_")
                xx = xx.replace(" ", "_")
                replacement.append(xx)
            data[k] = replacement
    data = data[["Name", "LeftHanded", "Team", "Year"]]
    return(data)


# In[3]:


#pullBoxscores('NYY', 2019, "yakees19") # success


# In[61]:


import pandas as pd
nyy19 = pd.read_csv("yakees19NYY_2019.csv",index_col=0)


# In[62]:


#Cleaning Data
nyy19 = nyy19.fillna("")

#Dropped Unnecessary columns
#nyy19 = nyy19.drop(["Unnamed: 0"], axis=1)

#Names as first and last
nyy19["Name"] = nyy19["Name"].str.replace("_", " ")

#Set Date in correct format
nyy19['Date'] = nyy19["Date"].astype(str)
nyy19['Date'] = nyy19['Date'].map(lambda x: str(x)[:-1])
nyy19["Date"] = pd.to_datetime(nyy19["Date"])

#Set correct data types
nyy19['PO'] = pd.to_numeric(nyy19["PO"], errors='coerce').fillna(0)
nyy19['A'] = pd.to_numeric(nyy19["A"], errors='coerce').fillna(0)


#Deleted rows Team Totals, as we only want player stats
nyy19 = nyy19[nyy19["Batting"] != "Team Totals"]

#Fill NaN values with empty spaces
nyy19


# What we will do is build a **Logistic Regression** model to attempt to predict who will get a hit in the game.
# 
# But first , lets explore our data

# In[63]:


nyy19.dtypes


# In[64]:


nyy19.describe()


# In[65]:


import numpy as np
nyy19['Hit ?'] = np.where(nyy19["H"] > 0,1,0)


# In[66]:


nyy19 = nyy19.fillna("")
nyy19


# In[67]:


nyy19.groupby("Name").mean().index


# In[68]:


def next_day_hit(df,i):
    player = df[df["Name"] == df.groupby("Name").mean().index[i]]
    player.loc[1725] = np.zeros(28,dtype=int)
    new_list = []
    for j in np.arange(len(player)-1):
        new_list.append(player.loc[:,"Hit ?"].to_numpy()[j+1])
    new_list.append(0)
    player["Next Day Hit"] = new_list
    return player     


# In[69]:


res = next_day_hit(nyy19,0).append(next_day_hit(nyy19,1)).append(next_day_hit(nyy19,2)).append(next_day_hit(nyy19,3)).append(next_day_hit(nyy19,4)).append(next_day_hit(nyy19,6)).append(next_day_hit(nyy19,8)).append(next_day_hit(nyy19,9)).append(next_day_hit(nyy19,10)).append(next_day_hit(nyy19,12)).append(next_day_hit(nyy19,13)).append( next_day_hit(nyy19,14)).append(next_day_hit(nyy19,15)).append(next_day_hit(nyy19,16)).append(next_day_hit(nyy19,17)).append(next_day_hit(nyy19,20)).append(next_day_hit(nyy19,21)).append(next_day_hit(nyy19,23)).append(next_day_hit(nyy19,25)).append(next_day_hit(nyy19,26)).append(next_day_hit(nyy19,27)).append(next_day_hit(nyy19,28)).append(next_day_hit(nyy19,29)).append(next_day_hit(nyy19,30))
 


# In[70]:


nyy = res.sort_values(by=['Game', 'BatPos'], ascending=True)


# In[71]:


nyy = nyy[nyy["Name"] != 0]
nyy


# In[72]:


#Partition of the data
from sklearn.model_selection import train_test_split

train, test = train_test_split(nyy, test_size=0.20, random_state=42)

train

X_train = train.drop(["Next Day Hit"], axis=1)
Y_train = train["Next Day Hit"]

X_test = test.drop(["Next Day Hit"], axis=1)
Y_test = test["Next Day Hit"]


# In[89]:


sns.scatterplot(data = train, x = "SLG", y="OPS", hue="Next Day Hit", legend='brief').set(xlim=(0,1), ylim=(0.2,2));


# In[90]:


sns.scatterplot(data = test, x = "OBP", y="BA", hue="Next Day Hit", legend='brief').set(xlim=(0.18, 0.5),ylim=(0.2,0.4));


# In[73]:


X_train = X_train.drop(["Batting", "Details", "Date", "Name"], axis=1)


# In[75]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train,Y_train)

training_accuracy = model.score(X_train,Y_train)
print("Training Accuracy: ", training_accuracy)


# In[76]:


from sklearn.ensemble import RandomForestClassifier as RFC
model2 = RFC()
model2.fit(X_train,Y_train)

training_accuracy2 = model2.score(X_train,Y_train)
print("Training Accuracy: ", training_accuracy2)


# In[81]:


Y_train_hat = model.predict(X_train)
lr_precision = np.sum((Y_train_hat == 1) & (Y_train == 1)) / np.sum(Y_train_hat) # TP /TP + FP
lr_recall = np.sum((Y_train_hat == 1) & (Y_train == 1)) / np.sum(Y_train) # TP / TP + FN
lr_far = np.sum((Y_train_hat == 1) & (Y_train == 0)) / (np.sum((Y_train_hat == 1) & (Y_train == 0)) + np.sum((Y_train_hat == 0) & (Y_train == 0))) # FP / FP +TN

FP =  np.sum((Y_train_hat == 1) & (Y_train == 0))
FN =  np.sum((Y_train_hat == 0) & (Y_train == 1))
tot = len(X_train)

print("Precision: ", lr_precision)
print("Recall: ", lr_recall)
print("False Alarm Rate:", lr_far)
print("False Positive: " , FP/tot)
print("False Negative Rate: ",  FN/tot)


# In[82]:


Y_train_hat2 = model2.predict(X_train)
lr_precision = np.sum((Y_train_hat2 == 1) & (Y_train == 1)) / np.sum(Y_train_hat2) # TP /TP + FP
lr_recall = np.sum((Y_train_hat2 == 1) & (Y_train == 1)) / np.sum(Y_train) # TP / TP + FN
lr_far = np.sum((Y_train_hat2 == 1) & (Y_train == 0)) / (np.sum((Y_train_hat2 == 1) & (Y_train == 0)) + np.sum((Y_train_hat2 == 0) & (Y_train == 0))) # FP / FP +TN

FP =  np.sum((Y_train_hat2 == 1) & (Y_train == 0))
FN =  np.sum((Y_train_hat2 == 0) & (Y_train == 1))
tot = len(X_train)

print("Precision: ", lr_precision)
print("Recall: ", lr_recall)
print("False Alarm Rate:", lr_far)
print("False Positive: " , FP/tot)
print("False Negative Rate: ",  FN/tot)


# In[83]:


def rmse(y, yhat):
    return np.sqrt(np.mean((y - yhat)**2))

print("Training Error for Logistic Regression:", rmse(Y_train, Y_train_hat))
print("Training Error for Random Forest:", rmse(Y_train, Y_train_hat2))


# In[84]:


from sklearn.model_selection import cross_val_score
print(np.mean(cross_val_score(model, X_train, Y_train, scoring='accuracy', cv=10)))
print(np.mean(cross_val_score(model2, X_train, Y_train, scoring='accuracy', cv=10)))


# In[52]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set(style = "whitegrid", 
        color_codes = True,
        font_scale = 1.5)

# Imported to avoid 85 warnings from showing
import warnings
warnings.filterwarnings('ignore') 

# from lab 07
train_error_vs_N = []
range_of_num_features = range(1, X_train.shape[1] + 1)

for N in range_of_num_features:
    X_train_first_N_features = X_train.iloc[:, :N]    
    
    model2.fit(X_train_first_N_features, Y_train)
    train_error = rmse(Y_train, model2.predict(X_train_first_N_features))
    train_error_vs_N.append(train_error)
    
plt.figure(figsize=(10,10))
plt.plot(range_of_num_features, train_error_vs_N)
plt.legend(["training"], loc=1)
plt.xlabel("number of features")
plt.ylabel("RMSE");

# Note: if your plot doesn't appear in the PDF, you should try uncommenting the following line:
# plt.show()


# As we can see, so far as we add more features our rmse keeps decreasing

# In[53]:


X_test = X_test.drop(["Batting", "Details", "Date","Name"], axis=1)
model2.fit(X_test,Y_test)

test_accuracy = model2.score(X_test,Y_test)
print("Test Accuracy: ", test_accuracy)


# In[351]:


import matplotlib.pyplot as plt
from sklearn import tree

fn = list(X_train.columns)
cn = list(train.columns[28])
#fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
tree.plot_tree(model2.estimators_[1],
               feature_names = fn, 
               class_names=cn,
               filled = True);
fig.savefig('rf_individualtree.png')


# Over all, our **logistic regression model** is better than random guessing (50%) but not better than the recommended top hitter (~67%). However, it looks as if the **Random Forest Classifier** did well with this data set. I am unsure of my findings, as I remain skeptical of a model being able to be correct 100% of the time, and I wonder if this was caused by overfitting. One piece of evidence for this thought is that my Cross validation score is at .63, so getting a 100% accuracy does not sit well to me. 
# I will explore this more, but for now, that is all.
# 
# **Thank you.**
