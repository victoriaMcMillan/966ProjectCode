#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 20:23:47 2023

@author: victoriamcmillan
"""
import numpy as np
import matplotlib.pyplot as plt

randoms=np.ones(100)
vals=np.arange(100) + 1
odds=np.equal(np.mod(vals, 2), 1).astype(int)
multsOf5=np.equal(np.mod(vals, 5), 0).astype(int)
below10 = np.zeros(100)
below10[0:10] = 1

comb=multsOf5+below10#check allignment
comb[comb>1]=1

hypotheses=np.array([randoms,comb,odds])

fig, ax = plt.subplots(4,1, figsize=(7, 7))
fig.subplots_adjust(top=0.95, bottom=0.05, hspace=0.85,
left=0.05, right=0.95)

predictive = np.dot([.55,.27,.09], hypotheses)

ax[0].bar(np.arange(100)+1.0, predictive, 0.5, color='k')
data=[1,55,45,3,7]
ax[0].set_title('Human predictions given observation(s) 1, 55, 45, 3, 7 in Descending Popularity')
ax[0].set_xlim([-0.5, (100+1)+0.5])
ax[0].set_ylim([-0.05, 1.05])

topN = 3
for i in np.arange(1, 4):
    hypo_index = i-1
    numerical=np.where(hypotheses[hypo_index,:]==1)[0]
    #print(numerical+np.ones(len(numerical)))
    #print(posteriors[hypo_index])
    ax[i].bar(np.arange(100)+1.0, hypotheses[hypo_index,:], 0.5, color='k')
    ax[i].set_xlim([-0.5, (100+1)+0.5])
    ax[i].set_ylim([-0.05, 1.05])
    
ax[1].set_title('Numbers 1-100')
ax[2].set_title('Multiples of 5 and numbers below 10')
ax[3].set_title('Odd Numbers')



interval = np.zeros(100)
interval[70:80+1] = 1
evens=np.equal(np.mod(vals, 2), 0).astype(int)
comb=interval*evens


hypotheses=np.array([comb,evens,interval])

fig2, ax2 = plt.subplots(4,1, figsize=(7, 7))
fig2.subplots_adjust(top=0.95, bottom=0.05, hspace=0.85,
left=0.05, right=0.95)

predictive = np.dot([.63,.18,.18], hypotheses)

ax2[0].bar(np.arange(100)+1.0, predictive, 0.5, color='k')
data=[1,55,45,3,7]
ax2[0].set_title('Human predictions given observation(s) 72, 76, 78 in Descending Popularity')
ax2[0].set_xlim([-0.5, (100+1)+0.5])
ax2[0].set_ylim([-0.05, 1.05])

topN = 3
for i in np.arange(1, 4):
    hypo_index = i-1
    numerical=np.where(hypotheses[hypo_index,:]==1)[0]
    #print(numerical+np.ones(len(numerical)))
    #print(posteriors[hypo_index])
    ax2[i].bar(np.arange(100)+1.0, hypotheses[hypo_index,:], 0.5, color='k')
    ax2[i].set_xlim([-0.5, (100+1)+0.5])
    ax2[i].set_ylim([-0.05, 1.05])
    
ax2[1].set_title('Even Numbers 70-80')
ax2[2].set_title('Even Numbers')
ax2[3].set_title('Numbers 70-80')



