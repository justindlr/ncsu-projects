import pandas as pd
import os
import numpy as np
from math import radians, degrees, sin, cos, asin, acos, sqrt

wk_dir = os.path.abspath('..')
df = pd.read_csv(wk_dir+'\diningData\Centennial Geographical Information.csv')
df1 = pd.read_csv(wk_dir+'\diningData\dininggeodata.csv')

df = df.set_index('Location')
df1 = df1.set_index('Venue Name')



    
def haversine(lon1, lat1, lon2, lat2):
    """
    Takes in two coordinate points and returns the distance
        Args:
            lon1 - longitude1
            lat1 - latitude1
            lon2 - longitude2
            lat2 - latitide2
    Returns distance in miles
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    distance = 2 * 3958.756 * asin(sqrt(a))
    return distance

def distances(location):
    """
    Goal: return the closest dining venue
    Args:
        location - string of the building
        _type - type of venue (1 - food, 2 - drink, 3 - c-store)
    """
    distance_list = []
    building = df.loc[location]
    lon1 = building[1]
    lat1 = building[0]
    for index, row in df1.iterrows():
        lon2 = row[1] 
        lat2 = row[0]
        d = haversine(lon1, lat1, lon2, lat2)
        distance_list.append(d)
    df1['distance_list'] = distance_list
    name = df1.loc[df1['distance_list'].idxmin()] 
    return name

def distances_type(location, _type):
    """
    Goal: return the closest dining venue
    Args:
        location - string of the building
        _type - type of venue (1 - food, 2 - drink, 3 - c-store)
    """
    distance_list = []
    building = df.loc[location]
    lon1 = building[1]
    lat1 = building[0]
    df2 = df1.loc[df1['Type'] == _type]
    for index, row in df2.iterrows():
        lon2 = row[1] 
        lat2 = row[0]
        d = haversine(lon1, lat1, lon2, lat2)
        distance_list.append(d)
    df2['distance_list'] = distance_list
    name = df2.loc[df2['distance_list'].idxmin()] 
    return name

