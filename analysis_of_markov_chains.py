import pandas as pd
import os
import numpy as np

wk_dir = os.path.abspath('..')
df = pd.read_csv(wk_dir+'\Data\crits.csv')

states = ['zero', 'one', 'two', 'three', 'four', 'five'] 
transitionName = [['stay', 'up1'], ['down1', 'stay', 'up1'], ['down1', 'stay', 'up1'],
                  ['down1', 'stay', 'up1'], ['down1', 'stay', 'up1'], ['down1', 'stay']]

def crit_chain(state, attacks, crit_chance):
    """
    Given a number of crits in five basic attacks, output possible states, end state, and
    probability of sequence of states
    - Assumes crit chance is 50% for every basic attack
    
    Args:
        state - string of zero, one, two, three, four, or five that describes how many crits in our list of five
        basic attacks
        attacks - integer descrbing how many basic attacks to run through
        crit_chance - string describing crit chance in the form 'crit_##'
    Prints:
        Starting State, Possible States, End State after n attacks, Probability of Sequence of States
    """
    print("Start state: " + state)
    critList = [state] #stores the sequence of states taken
    if crit_chance == 'crit_10':
        transitionMatrix = [[.9, .1], [.18, .74, .08], [.36, .58, .06], [.54, .42, .04], [.74, .18, .08], [.9, .1]]
    if crit_chance == 'crit_20':
        transitionMatrix = [[.8, .2], [.16, .68, .16], [.32, .56, .12], [.48, .44, .08], [.64, .32, .04], [.8, .2]]
    if crit_chance == 'crit_30':
        transitionMatrix = [[.7, .3], [.14, .62, .24], [.28, .54, .18], [.42, .46, .12], [.56, .38, .06], [.7, .3]]
    if crit_chance == 'crit_40':
        transitionMatrix = [[.6, .4], [.12, .56, .32], [.24, .52, .24], [.36, .48, .16], [.48, .44, .08], [.6, .4]]
    if crit_chance == 'crit_50':
        transitionMatrix = [[.5, .5], [.1, .5, .4], [.2, .5, .3], [.3, .5, .2], [.4, .5, .1], [.5, .5]]
    if crit_chance == 'crit_60':
        transitionMatrix = [[.4, .6], [.08, .44, .48], [.16, .48, .36], [.24, .52, .24], [.32, .56, .12], [.4, .6]]
    if crit_chance == 'crit_70':
        transitionMatrix = [[.3, .7], [.06, .38, .56], [.12, .46, .42], [.18, .54, .28], [.24, .62, .14], [.3, .7]]
    if crit_chance == 'crit_80':
        transitionMatrix = [[.2, .8], [.04, .32, .64], [.08, .44, .48], [.12, .56, .32], [.16, .68, .16], [.2, .8]]
    if crit_chance == 'crit_90':
        transitionMatrix = [[.1, .9], [.02, .26, .72], [.04, .42, .54], [.06, .58, .36], [.08, .74, .18], [.1, .9]]
    i = 0
    prob = 1 # To calculate the probability of the critList
    while i != attacks:
        if state == "zero":
            change = np.random.choice(transitionName[0],replace=True,p=transitionMatrix[0])
            if change == "stay":
                prob = prob * transitionMatrix[0][0]
                state = "zero"
                critList.append("zero")
            elif change == "up1":
                prob = prob * transitionMatrix[0][1]
                state = "one"
                critList.append("one")
        elif state == "one":
            change = np.random.choice(transitionName[1],replace=True,p=transitionMatrix[1])
            if change == "down1":
                prob = prob * transitionMatrix[1][0]
                state = "zero"
                critList.append("zero")
            elif change == "stay":
                prob = prob * transitionMatrix[1][1]
                state = "one"
                critList.append("one")
            else:
                prob = prob * transitionMatrix[1][2]
                state = "two"
                critList.append("two")
        elif state == "two":
            change = np.random.choice(transitionName[2],replace=True,p=transitionMatrix[2])
            if change == "down1":
                prob = prob * transitionMatrix[2][0]
                state = "one"
                critList.append("one")
            elif change == "stay":
                prob = prob * transitionMatrix[2][1]
                state = "two"
                critList.append("two")
            else:
                prob = prob * transitionMatrix[2][2]
                state = "three"
                critList.append("three")
        elif state == "three":
            change = np.random.choice(transitionName[3],replace=True,p=transitionMatrix[3])
            if change == "down1":
                prob = prob * transitionMatrix[3][0]
                state = "two"
                critList.append("two")
            elif change == "stay":
                prob = prob * transitionMatrix[3][1]
                state = "three"
                critList.append("three")
            else:
                prob = prob * transitionMatrix[3][2]
                state = "four"
                critList.append("four")
        elif state == "four":
            change = np.random.choice(transitionName[4],replace=True,p=transitionMatrix[4])
            if change == "down1":
                prob = prob * transitionMatrix[4][0]
                state = "three"
                critList.append("three")
            elif change == "stay":
                prob = prob * transitionMatrix[4][1]
                state = "four"
                critList.append("four")
            else:
                prob = prob * transitionMatrix[4][2]
                state = "five"
                critList.append("five")
        elif state == "five":
            change = np.random.choice(transitionName[5],replace=True,p=transitionMatrix[5])
            if change == "down1":
                prob = prob * transitionMatrix[5][0]
                state = "four"
                critList.append("four")
            elif change == "stay":
                prob = prob * transitionMatrix[5][1]
                state = "five"
                critList.append("five")
        i += 1  
    print("State vists: " + str(critList))
    print("End state after "+ str(attacks) + " attacks: " + state)
    print("Probability of the possible sequence of states: " + str(prob))


# Function that forecasts the possible state for the next n attacks
crit_chain('zero', 10, 'crit_20')
