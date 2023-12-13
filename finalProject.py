#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 10:01:33 2023

@author: victoriamcmillan
"""

import numpy as np
import matplotlib.pyplot as plt
from math import log,inf





def number_game_complex_init(N, interval_prior, math_prior, and_prior, or_prior, common_interval_bias): 
    '''
    common_interval_bias is the percentage of the interval prior that will go to the
    common intervals
    '''
    
    if abs((interval_prior + math_prior + and_prior + or_prior) - 1) > 0.05: 
        raise ValueError('Sum of priors should be 1!')
    
    if 0>common_interval_bias or common_interval_bias>1:
        raise ValueError('common_interval_bias should be between 0 and 1!')


	#generate interval concepts of small and medium length
    hypotheses = np.zeros((0, N))
    vals = np.arange(N) + 1
    
    #add in common intervals (starting at multiples of 5)
    for size in [5,10,15,20]:
        for start in np.arange(0,N-size,5):
            end = start + size
            interval = np.zeros(N)
            interval[start:end+1] = 1
            hypotheses = np.vstack([hypotheses, interval])
            
    last_common_interval=hypotheses.shape[0]

    #add in less likely intervals
    for size in np.arange(20)+1:
        for start in np.arange(N-size):
            end = start + size
            interval = np.zeros(N)
            interval[start:end+1] = 1
            if interval.tolist() not in hypotheses.tolist(): #avoid duplicating favored intervals
                hypotheses = np.vstack([hypotheses, interval])
            
    hypotheses=np.vstack([hypotheses,np.ones(100)]) #all numbers 1-100

    last_interval_concept = hypotheses.shape[0] 


	#put in odds
    concept = np.equal(np.mod(vals, 2), 1).astype(int)
    hypotheses = np.vstack([hypotheses, concept])

	#put in multiples of 2 to 12
    for base in np.arange(2,13):
        concept = np.equal(np.mod(vals, base), 0).astype(int)
        hypotheses = np.vstack([hypotheses, concept]) 
        
    last_math_concept=hypotheses.shape[0]
    
    #put in combinations of hypotheses
    favordOrs=[]
    favordAnds=[]
    for i in range(last_common_interval):
        for j in range(last_math_concept-last_interval_concept):
            #hypothesis that include both: (OR)
            hyp=hypotheses[i]+hypotheses[j+last_interval_concept]
            hyp[hyp>1]=1
            if hyp.tolist() not in hypotheses.tolist():
                if len(favordOrs)!=0:
                    if hyp.tolist() not in favordOrs.tolist():
                        favordOrs=np.vstack([favordOrs,hyp])
                else:
                    favordOrs=hyp
            #hypothesis that are conjuct of both: (AND)
            andHyp=hypotheses[i]*hypotheses[j+last_interval_concept]
            if andHyp.tolist() not in hypotheses.tolist():
                if len(favordAnds)!=0:
                    if andHyp.tolist() not in favordAnds.tolist():
                        favordAnds=np.vstack([favordAnds,andHyp])
                else:
                    favordAnds=andHyp
                    
    hypotheses=np.vstack([hypotheses,favordOrs])
    last_common_or_concept=hypotheses.shape[0]
    hypotheses=np.vstack([hypotheses,favordAnds])
    last_common_and_concept=hypotheses.shape[0]
    
    #add with other intervals
    ors=[]
    ands=[]
    for i in range(last_common_interval,last_interval_concept):
        for j in range(last_math_concept-last_interval_concept):
            #hypothesis that include both: (OR)
            hyp=hypotheses[i]+hypotheses[j+last_interval_concept]
            hyp[hyp>1]=1
            if hyp.tolist() not in hypotheses.tolist():
                if len(ors)!=0:
                    if hyp.tolist() not in ors.tolist():
                        ors=np.vstack([ors,hyp])
                else:
                    ors=hyp
            #hypothesis that are conjuct of both: (AND)
            andHyp=hypotheses[i]*hypotheses[j+last_interval_concept]
            if andHyp.tolist() not in hypotheses.tolist():
                if len(ands)!=0:
                    if andHyp.tolist() not in ands.tolist():
                        ands=np.vstack([ands,andHyp])
                else:
                    ands=andHyp
   
    hypotheses=np.vstack([hypotheses,ors])
    last_or_concept=hypotheses.shape[0]
    hypotheses=np.vstack([hypotheses,ands])
    

    last_hypothesis = hypotheses.shape[0]

	#compute prior probabilities
    commonIntervalPrior=interval_prior*common_interval_bias
    regularIntervalPrior=interval_prior-commonIntervalPrior
    priors = np.empty(last_hypothesis)
    priors[:last_common_interval] = commonIntervalPrior/last_common_interval
    priors[last_common_interval:last_interval_concept] = regularIntervalPrior/(last_interval_concept-last_common_interval)
    priors[last_interval_concept:last_math_concept] = math_prior/(last_math_concept-last_interval_concept)
    #combined priors
    commonIntervalAndPrior=and_prior*common_interval_bias
    regularIntervalAndPrior=and_prior-commonIntervalAndPrior
    commonIntervalOrPrior=or_prior*common_interval_bias
    regularIntervalOrPrior=or_prior-commonIntervalOrPrior
    priors[last_math_concept:last_common_or_concept]=commonIntervalOrPrior/(last_common_or_concept-last_math_concept)
    priors[last_common_or_concept:last_common_and_concept]=commonIntervalAndPrior/(last_common_and_concept-last_common_or_concept)
    priors[last_common_and_concept:last_or_concept]=regularIntervalOrPrior/(last_or_concept-last_common_and_concept)
    priors[last_or_concept:]=regularIntervalAndPrior/(last_hypothesis-last_or_concept)

    return hypotheses, priors


def number_game_likelihood(hypothesis, data):
    """
		hypothesis is a logical (0 or 1) vector on N elements, where
		hypothesis[i] = 1 iff i is contained in the extension of the
		concept represented by hypothesis.

		data is, similarly, a logical vector where data[i] = 1 iff
		i is contained in the observed dataset.

		note that length(hypothesis) == length(data) unless the caller
		of this procedure messed up

		TODO: first check if data is consistent with the given hypothesis.

		if it isn't, P(D|H) = 0.

		TODO: under strong sampling WITH REPLACEMENT, every consistent hypothesis
		assigns probability 1/(#options) to each data draw.
    """
    #check if -1 is in the hypothesis
    if -1 in (np.subtract(hypothesis,data)):
        log_likelihood=-inf
    else:
        sizeH=np.count_nonzero(hypothesis==1) 
        sizeD=np.count_nonzero(data==1)
            
        log_likelihood=log((1/sizeH)**sizeD)

    return log_likelihood



def number_game_plot_predictions(hypotheses, priors, data):
    """
		hypotheses = a matrix whose columns are particular hypotheses,
		represented as logical vectors reflecting datapoint membership

		priors = a vector of prior probabilities for each hypothesis

		data = a vector of observed numbers
    """

    def numbers_to_logical(data):
        if np.isscalar(data): data = [data]
        logical_data = np.zeros(N)
        for datum in data:
            logical_data[datum-1] = 1
        return logical_data

    hyps, N = hypotheses.shape
    logical_data = numbers_to_logical(data)

	# compute the posterior for every hypothesis
    posteriors = np.zeros(hyps)

    for h in np.arange(hyps):
        log_joint = np.log(priors[h]) + number_game_likelihood(hypotheses[h,:], logical_data)
        joint = np.exp(log_joint)
        posteriors[h] = joint

    posteriors /= np.sum(posteriors)

	# compute the predictive contribution for each
	# hypothesis and add it in to the predictive

    predictive = np.dot(posteriors, hypotheses) #RATING FOR EACH NUMBER

	# plot it as a bar chart, also plot human data (if available)
	# and the top 6 hypotheses in decreasing order of posterior
	# probability

    fig, ax = plt.subplots(4,1, figsize=(7, 7))
    fig.subplots_adjust(top=0.95, bottom=0.05, hspace=0.85,
		left=0.05, right=0.95)


    ax[0].bar(np.arange(N)+1.0, predictive, 0.5, color='k')
    if np.isscalar(data): data = [data]
    ax[0].set_title('Model Predictions given observation(s) %s'
		% ', '.join(str(d) for d in data))
    ax[0].set_xlim([-0.5, (N+1)+0.5])
    ax[0].set_ylim([-0.05, 1.05])

   
    sort_indices = np.argsort(posteriors)[::-1]

    topN = 3
    for i in np.arange(1, 4):
        hypo_index = sort_indices[(i-1)]
        numerical=np.where(hypotheses[hypo_index,:]==1)[0]
        #print(numerical+np.ones(len(numerical)))
        #print(posteriors[hypo_index])
        ax[i].bar(np.arange(N)+1.0, hypotheses[hypo_index,:], 0.5, color='k')
        ax[i].set_xlim([-0.5, (N+1)+0.5])
        ax[i].set_ylim([-0.05, 1.05])

		# only consider hypotheses with probability greater 0
        if posteriors[hypo_index] == 0.0:
            ax[i].set_visible(False)
            topN -= 1
    ax[1].set_title('Top %u hypotheses in descending order of posterior probability' % topN)
    plt.show()
    
    return posteriors[sort_indices[0]]
    #return [posteriors[sort_indices[0]],posteriors[sort_indices[1]],posteriors[sort_indices[2]]]


data=[[10, 20, 55, 22],[22,19,55,24,70],[56,64,48,12],[11,22],[2,22,44,32,66],[10,40,30,20],[33,36,27,39],[9,39,63],[72,76,78],[15,80,90,25],
      [1,55,45,3,7],[18,84,86],[15,20,75,10],[11,22,44],[11,22,88],[21,49,63,99,97,98],[96,80,72],[96,8,72],[51,54,57]]

hyp,prior=number_game_complex_init(100, .25, .25, .25, .25, .5)
#posteriors=number_game_plot_predictions(hyp, prior, [72,76,78])
#print((np.array(posteriors)*6)+1)

posteriors=[]
for dataset in data:
    posteriors.append(number_game_plot_predictions(hyp, prior, dataset))
    

print((np.array(posteriors)*6)+1)






