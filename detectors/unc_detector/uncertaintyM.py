import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time 
import itertools
from scipy.optimize import minimize
from sklearn.utils import resample
from sklearn import preprocessing
from scipy.optimize import minimize_scalar
from scipy.stats import entropy
import warnings
random.seed(1)

################################################################################################################################################# ent

def sk_ent(probs):
	uncertainties = np.array([entropy(prob) for prob in probs])
	return uncertainties, uncertainties, uncertainties # all three are the same as we only calculate total uncertainty

def uncertainty_ent(probs): # three dimentianl array with d1 as datapoints, (d2) the rows as samples and (d3) the columns as probability for each class
	p = np.array(probs)
	entropy = -p*np.ma.log2(p)
	entropy = entropy.filled(0)
	a = np.sum(entropy, axis=1)
	a = np.sum(a, axis=1) / entropy.shape[1]
	p_m = np.mean(p, axis=1)
	total = -np.sum(p_m*np.ma.log2(p_m), axis=1)
	total = total.filled(0)
	e = total - a
	return total, e, a # now it should be correct

def uncertainty_ent_bays(probs, likelihoods): # three dimentianl array with d1 as datapoints, (d2) the rows as samples and (d3) the columns as probability for each class
	p = np.array(probs)
	entropy = -p*np.ma.log2(p)
	entropy = entropy.filled(0)

	a = np.sum(entropy, axis=2)
	al = a * likelihoods
	a = np.sum(al, axis=1)

	p_m = np.mean(p, axis=1) #* likelihoods
	total = -np.sum(p_m*np.ma.log2(p_m), axis=1)
	total = total.filled(0)
	e = total - a
	return total, e, a # now it should be correct


def uncertainty_ent_levi(probs, credal_size=30): # three dimentianl array with d1 as datapoints, (d2) the rows as samples and (d3) the columns as probability for each class
	p = [] #np.array(probs)
	for data_point in probs:
		d_p = []
		for sampling_seed in range(credal_size):
			d_p.append(resample(data_point, random_state=sampling_seed))
		p.append(np.array(d_p))
	p = np.array(p)
	p = np.mean(p, axis=2)
	entropy = -p*np.ma.log10(p)
	entropy = entropy.filled(0)
	a = np.sum(entropy, axis=1)
	a = np.sum(a, axis=1) / entropy.shape[1]
	p_m = np.mean(p, axis=1)
	total = -np.sum(p_m*np.ma.log10(p_m), axis=1)
	total = total.filled(0)
	e = total - a
	return total, e, a # now it should be correct


def uncertainty_ent_standard(probs): # for tree
	p = np.array(probs)
	entropy = -p*np.ma.log10(p)
	entropy = entropy.filled(0)
	total = np.sum(entropy, axis=1)
	return total, total, total # now it should be correct


################################################################################################################################################# set

def Interval_probability(total_number, positive_number):
    s = 1
    eps = .001
    valLow = (total_number - positive_number + s*eps*.5)/(positive_number+ s*(1-eps*.5))
    valUp = (total_number - positive_number + s*(1-eps*.5))/(positive_number+ s*.5)
    return [1/(1+valUp), 1/(1+valLow)]

def uncertainty_credal_point(total_number, positive_number):
    lower_probability, upper_probability = Interval_probability(total_number, positive_number)
    # due to the observation that in binary classification 
    # lower_probability_0 = 1-upper_probability_1 
    # upper_probability_0 = 1-lower_probability_1
    return -max(lower_probability/(1-lower_probability),(1-upper_probability)/upper_probability) 

def uncertainty_credal_tree(counts):
	credal_t = np.zeros(len(counts))
	for i, count in enumerate(counts):
		# print(f"index {i} count : {count[0][0]}     {count[0][1]}")
		credal_t[i] = uncertainty_credal_point(count[0][0] + count[0][1], count[0][1])
	return credal_t, credal_t, credal_t

def uncertainty_credal_tree_DF(counts):
	print(counts.shape)
	exit()

	credal_t = np.zeros(len(counts))
	for i, count in enumerate(counts):
		# print(f"index {i} count : {count[0][0]}     {count[0][1]}")
		credal_t[i] = uncertainty_credal_point(count[0][0] + count[0][1], count[0][1])
	return credal_t, credal_t, credal_t


def findallsubsets(s):
    res = np.array([])
    for n in range(1,len(s)+1):
        res = np.append(res, list(map(set, itertools.combinations(s, n))))
    return res

def v_q(set_slice):
	# sum
	sum_slice = np.sum(set_slice, axis=2)
	# max
	max_slice = np.min(sum_slice, axis=1)
	return max_slice

def m_q(probs):
	res = np.zeros(probs.shape[0])
	index_set = set(range(probs.shape[2]))
	subsets = findallsubsets(index_set) # this is B in the paper
	set_A = subsets[-1]

	for set_B in subsets:
		set_slice = probs[:,:,list(set_B)]
		set_minus = set_A - set_B
		m_q_set = v_q(set_slice) * ((-1) ** len(set_minus))
		# print(f">>> {set_B}		 {m_q_set}")
		res += m_q_set
	return res

def set_gh(probs):
	res = np.zeros(probs.shape[0])
	index_set = set(range(probs.shape[2]))
	subsets = findallsubsets(index_set) # these subests are A in the paper

	for subset in subsets:
		set_slice = probs[:,:,list(subset)]
		m_q_slice = m_q(set_slice)
		res += m_q_slice * math.log2(len(subset))
	return res

def uncertainty_set14(probs, bootstrap_size=0, sampling_size=0, credal_size=0, log=False):
	if bootstrap_size > 0:
		p = [] #np.array(probs)
		for data_point in probs:
			d_p = []
			for sampling_seed in range(bootstrap_size):
				d_p.append(resample(data_point, random_state=sampling_seed))
			p.append(np.array(d_p))
		p = np.array(p)
		p = np.mean(p, axis=2)
	if sampling_size > 0:
		p = [] 
		for sample_index in range(sampling_size):
			# number_of_samples = int(probs.shape[1] / sampling_size)
			sampled_index = np.random.choice(probs.shape[1], credal_size)
			p.append(probs[:,sampled_index,:])
		p = np.array(p)
		p = np.mean(p, axis=2)
		p = p.transpose([1,0,2])
	else:
		p = probs
		
	if log:
		print("------------------------------------set14 prob after averaging each ensemble")
		print("Set14 p \n" , p)
		print(p.shape)
	# entropy = -p*np.log2(p)
	entropy = -p*np.ma.log2(p)
	entropy = entropy.filled(0)

	entropy_sum = np.sum(entropy, axis=2)
	s_max = np.max(entropy_sum, axis=1)
	s_min = np.min(entropy_sum, axis=1)
	gh    = set_gh(p)
	total = s_max
	e = gh
	a = total - e
	return total, e, a 


def uncertainty_set15(probs, bootstrap_size=0, sampling_size=0, credal_size=0):
	if bootstrap_size > 0:
		p = [] #np.array(probs)
		for data_point in probs:
			d_p = []
			for sampling_seed in range(bootstrap_size):
				d_p.append(resample(data_point, random_state=sampling_seed))
			p.append(np.array(d_p))
		p = np.array(p)
		p = np.mean(p, axis=2)
	if sampling_size > 0:
		p = [] 
		for sample_index in range(sampling_size):
			# number_of_samples = int(probs.shape[1] / sampling_size)
			# print("number_of_samples ", number_of_samples)
			sampled_index = np.random.choice(probs.shape[1], credal_size)
			p.append(probs[:,sampled_index,:])
		p = np.array(p)
		p = np.mean(p, axis=2)
		p = p.transpose([1,0,2])
	else:
		p = probs

	entropy = -p*np.ma.log2(p)
	entropy = entropy.filled(0)
	entropy_sum = np.sum(entropy, axis=2)
	s_min = np.min(entropy_sum, axis=1)
	s_max = np.max(entropy_sum, axis=1)
	total = s_max
	e = s_max - s_min
	a = total - e
	return total, e, a 

def uncertainty_set16(probs, bootstrap_size=0, sampling_size=0, credal_size=0, log=False):
	if bootstrap_size > 0:
		p = [] #np.array(probs)
		for data_point in probs:
			d_p = []
			for sampling_seed in range(bootstrap_size):
				d_p.append(resample(data_point, random_state=sampling_seed))
			p.append(np.array(d_p))
		p = np.array(p)
		p = np.mean(p, axis=2)
	if sampling_size > 0:
		p = [] 
		for sample_index in range(sampling_size):
			# number_of_samples = int(probs.shape[1] / sampling_size)
			sampled_index = np.random.choice(probs.shape[1], credal_size)
			p.append(probs[:,sampled_index,:])
		p = np.array(p)
		p = np.mean(p, axis=2)
		p = p.transpose([1,0,2])
	else:
		p = probs
		
	if log:
		print("------------------------------------set14 prob after averaging each ensemble")
		print("Set14 p \n" , p)
		print(p.shape)
	# entropy = -p*np.log2(p)
	entropy = -p*np.ma.log2(p)
	entropy = entropy.filled(0)

	entropy_sum = np.sum(entropy, axis=2)
	s_min = np.min(entropy_sum, axis=1)
	gh    = set_gh(p)
	e = gh
	a = s_min
	total = a + e

	return total, e, a 

def uncertainty_set17(probs, bootstrap_size=0, sampling_size=0, credal_size=0, log=False):
	if bootstrap_size > 0:
		p = [] #np.array(probs)
		for data_point in probs:
			d_p = []
			for sampling_seed in range(bootstrap_size):
				d_p.append(resample(data_point, random_state=sampling_seed))
			p.append(np.array(d_p))
		p = np.array(p)
		p = np.mean(p, axis=2)
	if sampling_size > 0:
		p = [] 
		for sample_index in range(sampling_size):
			# number_of_samples = int(probs.shape[1] / sampling_size)
			sampled_index = np.random.choice(probs.shape[1], credal_size)
			p.append(probs[:,sampled_index,:])
		p = np.array(p)
		p = np.mean(p, axis=2)
		p = p.transpose([1,0,2])
	else:
		p = probs
		
	if log:
		print("------------------------------------set14 prob after averaging each ensemble")
		print("Set14 p \n" , p)
		print(p.shape)
	# entropy = -p*np.log2(p)
	entropy = -p*np.ma.log2(p)
	entropy = entropy.filled(0)
	p_m = np.mean(p, axis=1)
	total = -np.sum(p_m*np.ma.log2(p_m), axis=1)
	total = total.filled(0)
	entropy_sum = np.sum(entropy, axis=2)
	s_max = np.max(entropy_sum, axis=1)

	gh    = set_gh(p)
	e = gh
	a = s_max
	total = a + e

	return total, e, a 



def uncertainty_setmix(probs, credal_size=30):
	p = [] #np.array(probs)
	for data_point in probs:
		d_p = []
		for sampling_seed in range(credal_size):
			d_p.append(resample(data_point, random_state=sampling_seed))
		p.append(np.array(d_p))
	p = np.array(p)
	p = np.mean(p, axis=2)

	entropy = -p*np.log2(p)
	entropy_sum = np.sum(entropy, axis=2)
	s_min = np.min(entropy_sum, axis=1)
	s_max = np.max(entropy_sum, axis=1)
	e = set_gh(p)
	a = s_max - (s_max - s_min)
	total = e + a
	return total, e, a 


################################################################################################################################################# set convex

def convex_ent_max(alpha, p):
	alpha = np.reshape(alpha,(-1,1))
	p_alpha = alpha * p
	p_alpha_sum = np.sum(p_alpha, axis=0)
	entropy = -p_alpha_sum*np.log2(p_alpha_sum)
	entropy_sum = np.sum(entropy)
	return entropy_sum * -1 # so that we maximize it

def convex_ent_min(alpha, p):
	alpha = np.reshape(alpha,(-1,1))
	p_alpha = alpha * p
	p_alpha_sum = np.sum(p_alpha, axis=0)
	entropy = -p_alpha_sum*np.log2(p_alpha_sum)
	entropy_sum = np.sum(entropy)
	return entropy_sum # so that we maximize it

def constarint(x):
    sum_x = np.sum(x)
    return 1 - sum_x

def uncertainty_set14_convex(probs,bootstrap_size=0):
	if bootstrap_size > 0:
		p = [] #np.array(probs)
		for data_point in probs:
			d_p = []
			for sampling_seed in range(bootstrap_size):
				d_p.append(resample(data_point, random_state=sampling_seed))
			p.append(np.array(d_p))
		p = np.array(p)
		p = np.mean(p, axis=2)
	else:
		p = probs
	cons = ({'type': 'eq', 'fun': constarint})
	b = (0.0, 1.0)
	bnds = [ b for _ in range(probs.shape[1]) ]
	x0 = np.ones((probs.shape[1]))
	x0_sum = np.sum(x0)
	x0 = x0 / x0_sum

	s_max = []
	for data_point_prob in probs:	
		sol_max = minimize(convex_ent_max, x0, args=data_point_prob, method='SLSQP', bounds=bnds, constraints=cons)
		s_max.append(-sol_max.fun)


	s_max = np.array(s_max)
	gh    = set_gh(p)
	total = s_max
	e = gh
	a = total - e
	return total, e, a 

def uncertainty_set15_convex(probs, bootstrap_size=0):
	if bootstrap_size > 0:
		p = [] #np.array(probs)
		for data_point in probs:
			d_p = []
			for sampling_seed in range(bootstrap_size):
				d_p.append(resample(data_point, random_state=sampling_seed))
			p.append(np.array(d_p))
		p = np.array(p)
		p = np.mean(p, axis=2)
	else:
		p = probs
	cons = ({'type': 'eq', 'fun': constarint})
	b = (0.0, 1.0)
	bnds = [ b for _ in range(probs.shape[1]) ]
	x0 = np.random.rand(probs.shape[1])
	x0_sum = np.sum(x0)
	x0 = x0 / x0_sum

	s_max = []
	s_min = []
	for data_point_prob in probs:	
		sol_max = minimize(convex_ent_max, x0, args=data_point_prob, method='SLSQP', bounds=bnds, constraints=cons)
		s_max.append(-sol_max.fun)
		sol_min = minimize(convex_ent_min, x0, args=data_point_prob, method='SLSQP', bounds=bnds, constraints=cons)
		s_min.append(sol_min.fun)
	
	s_max = np.array(s_max)
	s_min = np.array(s_min)
	total = s_max
	e = s_max - s_min
	a = s_min #total - e
	return total, e, a

################################################################################################################################################# GS agent


def uncertainty_gs(probs, likelyhoods, credal_size):
	sorted_index = np.argsort(likelyhoods, kind='stable')
	l = likelyhoods[sorted_index]
	p = probs[:,sorted_index]

	gs_total = []
	gs_epist = []
	gs_ale   = []
	for level in range(credal_size-1):
		p_cut = p[:,0:level+2] # get the level cut probs based on sorted likelyhood
		# computing levi (set14) for level cut p_cut and appeinding to the unc array
		entropy = -p_cut*np.ma.log2(p_cut)
		entropy = entropy.filled(0)
		entropy_sum = np.sum(entropy, axis=2)
		s_max = np.max(entropy_sum, axis=1)
		s_min = np.min(entropy_sum, axis=1)
		gh    = set_gh(p_cut)
		total = s_max
		e = gh
		a = total - e
		gs_total.append(total)
		gs_epist.append(e)
		gs_ale.append(a)

	gs_total = np.mean(np.array(gs_total), axis=0)	
	gs_epist = np.mean(np.array(gs_epist), axis=0)	
	gs_ale   = np.mean(np.array(gs_ale), axis=0)	

	return gs_total, gs_epist, gs_ale



################################################################################################################################################# rl

def unc_rl_prob(train_probs, pool_probs, y_train, log=False):

	log_likelihoods = []
	for prob in train_probs:
		l = y_train * prob
		l = np.sum(l, axis=1)
		l = np.log(l)
		l = np.sum(l)
		log_likelihoods.append(l)
	log_likelihoods = np.array(log_likelihoods)
	
	max_l = np.amax(log_likelihoods)
	normalized_likelihoods = np.exp(log_likelihoods - max_l)
	if log:
		print(">>> debug log_likelihoods \n", log_likelihoods)
		print(">>> debug normalized_likelihoods \n", normalized_likelihoods)
		print(">>> debug train_probs \n", train_probs[:,0,:])
		print("------------------------------------")
	min_pos_nl_list = []
	min_neg_nl_list = []
	for prob, nl in zip(pool_probs, normalized_likelihoods):
		# print(prob)
		pos = prob[:,0] - prob[:,1] # diffefence betwean positive and negetive class
		neg = prob[:,1] - prob[:,0] #-1 * pos # diffefence betwean negetive and positive class
		pos = pos.clip(min=0)
		neg = neg.clip(min=0)
		
		nl_array = np.full(pos.shape, nl) # the constant likelihood vector

		min_pos_nl_list.append(np.minimum(nl_array,pos)) # min for pos support
		min_neg_nl_list.append(np.minimum(nl_array,neg)) # min for neg support
	min_pos_nl_list = np.array(min_pos_nl_list)
	min_neg_nl_list = np.array(min_neg_nl_list)
	pos_suppot = np.amax(min_pos_nl_list, axis=0) # sup for pos support
	neg_suppot = np.amax(min_neg_nl_list, axis=0) # sup for neg support
	epistemic = np.minimum(pos_suppot, neg_suppot)
	aleatoric = 1 - np.maximum(pos_suppot, neg_suppot)
	total = epistemic + aleatoric

	return total, epistemic, aleatoric


def uncertainty_rl_avg(counts):
	# unc = np.zeros((counts.shape[0],counts.shape[1],3))
	support = np.zeros((counts.shape[0],counts.shape[1],2))
	for i,x in enumerate(counts):
		for j,y in enumerate(x):
			# res = relative_likelihood(y[0],y[1])
			# if y[0] >= 1000 or y[1] >= 1000:
			res = degrees_of_support_linh(y[1]+y[0],y[0])
			# else:
			# res = linh_fast(y[0],y[1])
			# res = rl_fast(y[0],y[1])
			support[i][j] = res
			# unc[i][j] = res
	support = np.mean(support, axis=1)
	t, e, a = rl_unc(support)
	# unc = unc / np.linalg.norm(unc, axis=0)		
	# unc = np.mean(unc, axis=1)
	# t = unc[:,0]
	# e = unc[:,1]
	# a = unc[:,2]
	return t,e,a

def unc_avg_sup_rl(counts):
	support = np.zeros((counts.shape[0],counts.shape[1],2))
	for i,x in enumerate(counts):
		for j,y in enumerate(x):
			# res = degree_of_support(y[0],y[1])
			# res = degrees_of_support_linh(y[1]+y[0],y[0])
			res = linh_fast(y[0],y[1])
			# res = sup_fast(y[0],y[1])
			support[i][j] = res
	support = np.mean(support, axis=1)
	t, e, a = rl_unc(support)
	return t,e,a

def unc_rl_score(counts):
	support = np.zeros((counts.shape[0],counts.shape[1],2))
	for i,x in enumerate(counts):
		for j,y in enumerate(x):
			res = linh_fast(y[0],y[1])
			support[i][j] = res
	
	# unc calculation
	epistemic = np.minimum(support[:,:,0], support[:,:,1])
	aleatoric = 1 - np.maximum(support[:,:,0], support[:,:,1])
	total = epistemic + aleatoric

	epistemic = np.mean(epistemic, axis=1)
	aleatoric = np.mean(aleatoric, axis=1)
	total     = np.mean(total,     axis=1)

	# score calculation
	s_score = np.reshape(support,(-1,2))
	i1 = np.arange(s_score.shape[0])
	i2 = s_score.argmin(axis=1)
	s_score[i1, i2] = 0
	s_score = np.reshape(s_score,(-1,counts.shape[1],2))
	s_score = np.mean(s_score, axis=1)
	s_score  = np.abs(s_score[:,0] - s_score[:,1])
	s_score = 1 - s_score

	# final unc * score
	epistemic = s_score * epistemic
	aleatoric = s_score * aleatoric
	total     = s_score * total

	return total,epistemic,aleatoric

def EpiAle_Averaged_Uncertainty_Preferences_Weight(clf, X_train, y_train, X_pool, uncertype, n_trees):
    n_samples = len(y_train)
    length_pool = len(X_pool)
    uncertainties = [0 for ind in range(length_pool)]
    pool_preferencePos = [0 for ind in range(length_pool)]
    pool_preferenceNeg = [0 for ind in range(length_pool)]
    for tree in clf:
        acc = tree.score(X_train, y_train)
        leaves_indices = tree.apply(X_pool, check_input=True).tolist()
        unique_leaves_indices = list(set(leaves_indices))
        leaves_indices_train = tree.apply(X_train, check_input=True).tolist()
        leaves_inices_train_pos = [leaves_indices_train[i] for i in range(n_samples) if y_train[i] == 1]
        leaves_size_neighbours = []
        leaves_uncertainties = []
        leaves_preferencePos = []
        leaves_preferenceNeg = []
        for leaf_index in range(len(unique_leaves_indices)):
            n_total_instance = leaves_indices_train.count(unique_leaves_indices[leaf_index])
            n_positive_instance = leaves_inices_train_pos.count(unique_leaves_indices[leaf_index])
            leaves_size_neighbours.append(n_total_instance)
            posSupPa, negSupPa =  degrees_of_support_linh(n_total_instance, n_positive_instance)
            epistemic = min(posSupPa, negSupPa)
            aleatoric = 1 -max(posSupPa, negSupPa)
            if posSupPa > negSupPa:
                preferencePos = 1- (epistemic+aleatoric)
            elif posSupPa == negSupPa:
                preferencePos = 1- (epistemic+aleatoric)/2
            else:
                preferencePos = 0
            preferenceNeg = 1 - (epistemic+aleatoric+preferencePos)
            leaves_preferencePos.append(preferencePos*acc)
            leaves_preferenceNeg.append(preferenceNeg*acc)
            if uncertype == "e": # Epistemic uncertainty
                leaves_uncertainties.append(epistemic)
            if uncertype == "a": # Aleatoric uncertainty
                leaves_uncertainties.append(aleatoric)
            if uncertype == "t": # Epistemic + Aleatoric uncertainty
               leaves_uncertainties.append(epistemic + aleatoric)       
        for instance_index in range(length_pool):
            uncertainties[instance_index] += leaves_uncertainties[unique_leaves_indices.index(leaves_indices[instance_index])]
#            neighbour_sizes[instance_index] += [leaves_size_neighbours[unique_leaves_indices.index(leaves_indices[instance_index])]]
            pool_preferencePos[instance_index] +=leaves_preferencePos[unique_leaves_indices.index(leaves_indices[instance_index])]
            pool_preferenceNeg[instance_index] +=leaves_preferenceNeg[unique_leaves_indices.index(leaves_indices[instance_index])]
    uncertaintiesEA = []
    for instance_index in range(length_pool):
        AveragePreferencePos = pool_preferencePos[instance_index]/n_trees
        AveragePreferenceNeg = pool_preferenceNeg[instance_index]/n_trees
        score = 1 - abs(AveragePreferencePos-AveragePreferenceNeg)
        uncertaintiesEA.append(score*uncertainties[instance_index]/n_trees)
    return uncertaintiesEA

def EpiAle_Averaged_Uncertainty_Preferences(clf, X_train, y_train, X_pool, uncertype, n_trees):
    n_samples = len(y_train)
    length_pool = len(X_pool)
    positive_preferences = [0 for ind in range(length_pool)]
    negative_preferences = [0 for ind in range(length_pool)]
    uncertainty = [0 for ind in range(length_pool)]
    for tree in clf:
        acc = tree.score(X_train, y_train)
        leaves_indices = tree.apply(X_pool, check_input=True).tolist()
        unique_leaves_indices = list(set(leaves_indices))
        leaves_indices_train = tree.apply(X_train, check_input=True).tolist()
        leaves_inices_train_pos = [leaves_indices_train[i] for i in range(n_samples) if y_train[i] == 1]
        leaves_preferencePos = []
        leaves_preferenceNeg = []
        leaves_uncertainties = []
        for leaf_index in range(len(unique_leaves_indices)):
            n_total_instance = leaves_indices_train.count(unique_leaves_indices[leaf_index])
            n_positive_instance = leaves_inices_train_pos.count(unique_leaves_indices[leaf_index])
            posSupPa, negSupPa =  degrees_of_support_linh(n_total_instance, n_positive_instance)
            epistemic = min(posSupPa, negSupPa)
            aleatoric = 1 -max(posSupPa, negSupPa)
            if posSupPa > negSupPa:
                preferencePos = 1- (epistemic+aleatoric)
            elif posSupPa == negSupPa:
                preferencePos = 1- (epistemic+aleatoric)/2
            else:
                preferencePos = 0
            preferenceNeg = 1 - (epistemic+aleatoric + preferencePos)
            leaves_preferencePos.append(preferencePos*acc)
            leaves_preferenceNeg.append(preferenceNeg*acc)

            if uncertype == "e": # Epistemic uncertainty
                leaves_uncertainties.append(epistemic*acc)
            if uncertype == "a": # Aleatoric uncertainty
                leaves_uncertainties.append(aleatoric*acc) 
            if uncertype == "t": # Epistemic + Aleatoric uncertainty
               leaves_uncertainties.append((epistemic + aleatoric) *acc)
        for instance_index in range(length_pool):
            positive_preferences[instance_index] += leaves_preferencePos[unique_leaves_indices.index(leaves_indices[instance_index])]
            negative_preferences[instance_index] += leaves_preferenceNeg[unique_leaves_indices.index(leaves_indices[instance_index])]
            uncertainty[instance_index] += leaves_uncertainties[unique_leaves_indices.index(leaves_indices[instance_index])]
    uncertaintiesEA = []
    for instance_index in range(length_pool):
        preferencePos = positive_preferences [instance_index]/n_trees
        preferenceNeg = negative_preferences [instance_index]/n_trees
        score = 1 - abs(preferencePos-preferenceNeg)
        uncertaintiesEA.append((uncertainty[instance_index]/n_trees)*score)
    return uncertaintiesEA


def EpiAle_Averaged_Support_Uncertainty(clf, X_train, y_train, X_pool, uncertype, n_trees):
    n_samples = len(y_train)
    length_pool = len(X_pool)
    positive_support = [0 for ind in range(length_pool)]
    negative_support = [0 for ind in range(length_pool)]
    uncertainty = [0 for ind in range(length_pool)]
    for tree in clf:
        leaves_indices = tree.apply(X_pool, check_input=True).tolist()
        unique_leaves_indices = list(set(leaves_indices))
        leaves_indices_train = tree.apply(X_train, check_input=True).tolist()
        leaves_inices_train_pos = [leaves_indices_train[i] for i in range(n_samples) if y_train[i] == 1] 
        leaves_posSupPa = []
        leaves_negSupPa = []
        leaves_uncertainties = []
        for leaf_index in range(len(unique_leaves_indices)): 
            n_total_instance = leaves_indices_train.count(unique_leaves_indices[leaf_index])
            n_positive_instance = leaves_inices_train_pos.count(unique_leaves_indices[leaf_index])
            posSupPa, negSupPa =  degrees_of_support_linh(n_total_instance, n_positive_instance)
            if posSupPa >= negSupPa:
                leaves_posSupPa.append(posSupPa)
                leaves_negSupPa.append(0)
            else:
                leaves_posSupPa.append(0)
                leaves_negSupPa.append(negSupPa)
            if uncertype == "e": # Epistemic uncertainty
                leaves_uncertainties.append(min(posSupPa, negSupPa))
            if uncertype == "a": # Aleatoric uncertainty
                leaves_uncertainties.append(1 -max(posSupPa, negSupPa)) 
            if uncertype == "t": # Epistemic + Aleatoric uncertainty
               leaves_uncertainties.append(min(posSupPa, negSupPa) +1 -max(posSupPa, negSupPa))
        for instance_index in range(length_pool): 
            positive_support[instance_index] += leaves_posSupPa[unique_leaves_indices.index(leaves_indices[instance_index])]
            negative_support[instance_index] += leaves_negSupPa[unique_leaves_indices.index(leaves_indices[instance_index])]
            uncertainty[instance_index] += leaves_uncertainties[unique_leaves_indices.index(leaves_indices[instance_index])]
    uncertaintiesEA = []
    for instance_index in range(length_pool):
        posSupPa = positive_support[instance_index]/n_trees
        negSupPa = negative_support[instance_index]/n_trees
        score = 1 - abs(posSupPa-negSupPa)
        uncertaintiesEA.append((uncertainty[instance_index]/n_trees)*score) 
    return uncertaintiesEA


def uncertainty_rl_ALB(counts):
	unc = np.zeros((counts.shape[0],counts.shape[1],3))
	for i,x in enumerate(counts):
		for j,y in enumerate(x):
			res = relative_likelihood(y[0],y[1])
			unc[i][j] = res
	e_unc = np.mean(unc, axis=1)
	a_unc = np.max(unc, axis=1)
	e = e_unc[:,1]
	a = a_unc[:,2]
	t = a + e

	return t,e,a



def uncertainty_rl_one(counts):
	unc = []
	for class_counts in counts:
		unc.append(relative_likelihood(class_counts[0],class_counts[1]))
	unc = np.array(unc)
	t = unc[:,0]
	e = unc[:,1]
	a = unc[:,2]
	return t,e,a
	

def likelyhood(p,n,teta):
	
	# old
	# a = teta**p
	# b = (1-teta)**n
	# c = (p/(n+p))**p
	# d = (n/(n+p))**n
	# return (a * b) / (c * d)

	if   p == 0:
		return ( ( (1-teta) * (n + p) ) / n ) ** n
	elif n == 0:
		return ( ( teta * (n + p) ) / p ) ** p
	else:
		return ( ( ( teta * (n + p) ) / p ) ** p ) * ( ( ( (1-teta) * (n + p) ) / n ) ** n )



def prob_pos(teta):
	return (2 * teta) - 1

def prob_neg(teta):
	return 1 - (2*teta)

def degree_of_support(pos,neg):
	sup_pos = 0
	sup_neg = 0
	for x in range(1,100):
		x /= 100

		l = likelyhood(pos,neg,x)
		p_pos = prob_pos(x)
		min_pos = min(l,p_pos)

		if min_pos > sup_pos:
			sup_pos = min_pos

		p_neg = prob_neg(x)

		min_neg = min(l,p_neg)
		if min_neg > sup_neg:
			sup_neg = min_neg
	return np.array([sup_pos, sup_neg])


############################################################ Linh
# def targetFunction(alpha, sizeins, posins, classId):
#     if classId == 1:
#        highFunc = max(2*alpha -1,0)
#     else:
#        highFunc = max(1 -2*alpha,0) 
#     necins = sizeins - posins
#     proportion = posins*(1/float(sizeins))
#     numerator = (alpha**posins)*((1-alpha)**necins)
#     denominator = (proportion**posins)*((1-proportion)**necins)
#     supportFunc = numerator*(1/float(denominator))
#     TargetFunc = - min(supportFunc, highFunc)
#     return TargetFunc
def targetFunction(alpha, sizeins, p, classId):
    if classId == 1:
        highFunc = max(2*alpha -1,0)
    else:
        highFunc = max(1 -2*alpha,0)
    n = sizeins - p
    if p == 0:
        supportFunc = (((1-alpha)*(n + p))/n)**n
    elif n == 0:
        supportFunc = ((alpha*(n+p))/p)**p
    else:
        supportFunc = (((alpha*(n+p))/p)**p)*((((1-alpha)*(n+p))/n)**n)
    TargetFunc = - min(supportFunc, highFunc)
    return TargetFunc
 
dictionary_DoS ={}    
def degrees_of_support_linh(sizeins, posins):
    global dictionary_DoS    
    key = "%i_%i"%(sizeins, posins) 
    if (key in dictionary_DoS):
        return dictionary_DoS.get(key)        
    if sizeins == 0:
        return [1,1]
    def Optp(alpha): return targetFunction(alpha, sizeins, posins, 1)
    posSupPa =  minimize_scalar(Optp, bounds=(0, 1), method='bounded')
    def Optn(alpha): return targetFunction(alpha, sizeins, posins, -1)
    negSupPa =  minimize_scalar(Optn, bounds=(0, 1), method='bounded')  
    dictionary_DoS[key] = [-posSupPa.fun, -negSupPa.fun] 
    return [-posSupPa.fun, -negSupPa.fun]
############################################################ Linh End


def rl_unc(support): # rl unc with the degrees of support
	epistemic = np.minimum(support[:,0], support[:,1])
	aleatoric = 1 - np.maximum(support[:,0], support[:,1])
	total = epistemic + aleatoric
	# unc = np.stack((total, epistemic, aleatoric), axis=1)
	return total, epistemic, aleatoric
	
def relative_likelihood(pos,neg):
	sup_pos = 0
	sup_neg = 0
	for x in range(1,1000):
		x /= 1000

		l = likelyhood(pos,neg,x)
		p_pos = prob_pos(x)
		min_pos = min(l,p_pos)

		if min_pos > sup_pos:
			sup_pos = min_pos

		p_neg = prob_neg(x)

		min_neg = min(l,p_neg)
		if min_neg > sup_neg:
			sup_neg = min_neg
	epistemic = min(sup_pos, sup_neg)
	aleatoric = 1 - max(sup_pos, sup_neg)
	total = epistemic + aleatoric

	# if pos ==0 and neg == 84:
	# 	print(f" pos {pos} neg {neg} unc {np.array([total, epistemic, aleatoric])}")


	return np.array([total, epistemic, aleatoric])

# n = 500
# rl_unc_array = np.zeros((n*2,n*2,3))
# rl_sup_array = np.zeros((n*2,n*2,2))
# rl_linh_array = np.zeros((n*2,n*2,2))

# for i in range(n):
# 	for j in range(i,n):
# 		if(i==0 and j==0):
# 			continue
# 		rl = relative_likelihood(i,j)
# 		rl_unc_array[i][j] = rl
# 		rl_unc_array[j][i] = rl

# for i in range(n):
# 	for j in range(n):
# 		if(i==0 and j==0):
# 			continue
# 		spd = degree_of_support(i,j)
# 		rl_sup_array[i][j] = spd

# for i in range(n):
# 	for j in range(n):
# 		if(i==0 and j==0):
# 			continue
# 		spd = degrees_of_support_linh(i + j,i)
# 		rl_linh_array[i][j] = spd

# with open('Data/pr_rl/rl_unc_array.npy', 'rb') as f:
# 	rl_unc_array = np.load(f)
# 	# print("rl_unc_array shape", rl_unc_array.shape)
# with open('Data/pr_rl/rl_sup_array.npy', 'rb') as f:
# 	rl_sup_array = np.load(f)
# 	# print("rl_sup_array shape", rl_sup_array.shape)
# with open('Data/pr_rl/rl_linh_array.npy', 'rb') as f:
# 	rl_linh_array = np.load(f)
# 	# print("rl_linh_array shape", rl_linh_array.shape)
rl_unc_array = 0
rl_sup_array = 0
rl_linh_array = 0
def rl_fast(pos,neg):	
	return rl_unc_array[int(pos)][int(neg)]

def sup_fast(pos,neg):	
	return rl_sup_array[int(pos)][int(neg)]

def linh_fast(pos,neg):	
	return rl_linh_array[int(pos)][int(neg)]




#################################################################################################################################################

def accuracy_rejection(predictions_list, labels_list, uncertainty_list, unc_value=False, log=False): # 2D inputs for average plot -> D1: runs D2: uncertainty data

	accuracy_list = []
	r_accuracy_list = []
	
	steps = np.array(list(range(90)))
	if unc_value:
		steps = uncertainty_list


	for predictions, uncertainty, labels in zip(predictions_list, uncertainty_list, labels_list):

		predictions = np.array(predictions)
		uncertainty = np.array(uncertainty)

		correctness_map = []
		for x, y in zip(predictions, labels):
			if x == y:
				correctness_map.append(1)
			else:
				correctness_map.append(0)

		# uncertainty, correctness_map = zip(*sorted(zip(uncertainty,correctness_map),reverse=False))

		correctness_map = np.array(correctness_map)
		sorted_index = np.argsort(uncertainty, kind='stable')
		uncertainty = uncertainty[sorted_index]
		correctness_map = correctness_map[sorted_index]

		correctness_map = list(correctness_map)
		uncertainty = list(uncertainty)
		data_len = len(correctness_map)
		accuracy = []

		for step_index, x in enumerate(steps):
			if unc_value:
				rejection_index = step_index
			else:
				rejection_index = int(data_len *(len(steps) - x) / len(steps))
			x_correct = correctness_map[:rejection_index].copy()
			x_unc = uncertainty[:rejection_index].copy()
			if log:
				print(f"----------------------------------------------- rejection_index {rejection_index}")
				for c,u in zip(x_correct, x_unc):
					print(f"correctness_map {c} uncertainty {u}")
				# print(f"rejection_index = {rejection_index}\nx_correct {x_correct} \nunc {x_unc}")
			if rejection_index == 0:
				accuracy.append(np.nan) # random.random()
			else:
				accuracy.append(np.sum(x_correct) / rejection_index)
		accuracy_list.append(accuracy)

		# random test plot
		r_accuracy = []
		
		for step_index, x in enumerate(steps):
			random.shuffle(correctness_map)
			if unc_value:
				r_rejection_index = step_index
			else:
				r_rejection_index = int(data_len *(len(steps) - x) / len(steps))

			r_x_correct = correctness_map[:r_rejection_index].copy()
			if r_rejection_index == 0:
				r_accuracy.append(np.nan)
			else:
				r_accuracy.append(np.sum(r_x_correct) / r_rejection_index)

		r_accuracy_list.append(r_accuracy)

	accuracy_list = np.array(accuracy_list)
	r_accuracy_list = np.array(r_accuracy_list)
		
	# print(accuracy_list)
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", category=RuntimeWarning)

		avg_accuracy = np.nanmean(accuracy_list, axis=0)
		avg_r_accuracy = np.nanmean(r_accuracy_list, axis=0)
		std_error = np.std(accuracy_list, axis=0) / math.sqrt(len(uncertainty_list))


	return avg_accuracy, avg_accuracy - std_error, avg_accuracy + std_error, avg_r_accuracy , steps
