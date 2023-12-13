#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 10:34:52 2023

@author: victoriamcmillan
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

a=np.array([0,1,0,1,0,1,0,1])
b=np.array([1,1,1,1,0,0,0,0])


model2=[1.43526229, 1.65998116, 4.46500962, 4.71360951, 5.187317,6.16735324,
 2.9646546,  3.51045597, 1.57196921, 5.73845187, 4.9520146,  2.54750922,
 5.80590776, 6.3515247,  6.34531567, 3.02936205, 5.24346006, 5.29408866,
 2.41592282]

modelV2=[1.47357381, 1.94888202, 4.38374318, 4.55772326, 5.28904377, 5.99352488,
 3.47912556, 3.51329438, 1.59207475, 5.68058938, 4.63542268, 2.49270431,
 5.73452864, 6.12149362, 6.11598689, 1.95398861, 5.10066713, 5.13329387,
 2.00033415]

humans =[4.727272727,3.090909091,5.272727273,6.181818182,5.1,6.363636364,5.545454545,4.8,5,
         5.363636364,3.727272727,4.545454545,5.363636364,6.181818182,6.090909091,
         4.090909091,4.909090909,5.272727273,
         5.272727273]

humansBrokenDown =[1,1,6.85714,6.3,5.375,6.4,4.6,5.125,4.8,
         5.2,4.3,4.5,5.363636364,6.3,6.090909091,
         4.5,5.857142857,5.272727273,
         5.25]



disagreeHumans=[4.444444444,2.666666667,5.222222222,3.444444444,4]
agreeHumans=[5,6,5,6.222222222,4.375,4.777777778,5.222222222,4.222222222,5.222222222,6,6,4.666666667,5.111111111,
5.333333333]

agreeModel=[4.04251123,4.33254443,4.88520317,6.02857878,2.9288653,0.66729741,5.52819385,
            1.80542742,5.60689238,6.24344548,6.23620161,4.95070341,5.0097701,1.65190996]
disagreeModel=[0.507806,0.76997802,2.29209704,4.6106837,2.36758906]

disagreeModel1=[]
agreeModel1=[]
for i in range(len(model2)):
    if i in [0,1,6,15,18,8]:
        disagreeModel1.append(model2[i])
    else:
        agreeModel1.append(model2[i])

disagreeHumans=[]
agreeHumans=[]
for i in range(len(humans)):
    if i in [0,1,6,15,18,8]:
        disagreeHumans.append(humans[i])
    else:
        agreeHumans.append(humans[i])


plt.scatter(agreeHumans,agreeModel1,label='Simple')
plt.scatter(disagreeHumans,disagreeModel1,label='Complex')
plt.plot(np.unique(humans), np.poly1d(np.polyfit(humans, model2, 1))(np.unique(humans)))
plt.xlabel('Human Likelihood Rating')
plt.ylabel('Model Probability')
#plt.title('Underconfident Subject')
plt.legend()
plt.ylim(1,7)
plt.xlim(1,7)
plt.title('Human vs Model Certainty Rating')

corr_matrix = np.corrcoef(agreeHumans, agreeModel1)
corr = corr_matrix[0,1]
R_sq = corr**2
print(R_sq)

corr_matrix = np.corrcoef(humans, model2)
corr = corr_matrix[0,1]
R_sq = corr**2
print(R_sq)




#bar chart plot
species = ("Odd #'s", "n*5 and #'s Below 10", "Random #'s")
penguin_means = {
    'Humans': (7, 4.5, 2.66),
    'Model': (4.9520146 , 1.18526578, 0),
}

x = np.arange(len(species))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=2)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Rating')
ax.set_title('Probability Ratings for [1,55,45,3,7]')
ax.set_xticks(x + width, species)
ax.legend(loc='upper right', ncols=2)
ax.set_ylim(1, 7)

plt.show()

fig3=plt.figure()


plt.bar(['Simple','Complex'],[0.9285714286,0.3333333333])
plt.xlabel('Most Likely Hypothesis Type')
plt.ylabel('% Datasets Agreed')
plt.ylim(0,1)
plt.title('Human-Model Agreement Rate')


