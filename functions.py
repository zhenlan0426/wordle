#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 11:04:17 2022

@author: will
"""
import numpy as np
import torch
import pickle
matrix = np.load('/home/will/Desktop/LC/wordle/matrix.npy')
out = []
# %timeit groupby1(matrix,5)
# %timeit groupby2(matrix,5)
# %timeit groupby1(matrix[:,:500],1)
# %timeit groupby2(matrix[:,:500],1)
# %timeit groupby1(matrix[:,:100],1)
# %timeit groupby2(matrix[:,:100],1)

mapping = dict()
policy_map = dict()
# with open('/home/will/Desktop/LC/wordle/mapping.pkl', 'rb') as f:
#     mapping = pickle.load(f)
#l = []
def entropy(p):
    return np.sum(-p*np.log2(p))

# def groupby1(matrix,row_n):
#     row = matrix[row_n]
#     sort_idx = np.argsort(row)
#     row = row[sort_idx]
#     diff = np.where(np.diff(row)!=0)[0] + 1
#     diff = np.insert(diff,0,0)
#     diff = np.insert(diff,diff.shape[0],matrix.shape[1])
#     matrix = matrix[:,sort_idx]
#     return [matrix[:,i:j] for i,j in zip(diff[:-1],diff[1:])]
        
# def groupby2(matrix,row_n):
#     row = matrix[row_n]
#     unq = np.unique(row)
#     return [matrix[:,row==u] for u in unq]


def wordle(matrix,index,top=0.6,count=2309):
    # return min expected guesses
    if count == 1: return 1
    if count == 2: return 1.5
    if tuple(index) in mapping:
        return mapping[tuple(index)]
    best = np.log2(count)
    threshold = best * top
    min_ = 1000
    for row in range(12953):
        tmp = matrix[row]
        unq,counts = np.unique(tmp,return_counts=True)
        ps = counts/count
        entro = entropy(ps)
        prob = ps[np.where(unq==242)[0]]
        prob = prob if prob.size > 0 else 0
        if entro == best:
            threshold = best # wont consider less than best entropy
            guess_row = 2 - prob # 1 + (1-prob) * 1 + prob * 0
            if guess_row < min_:
                min_ = guess_row
        elif entro>threshold:
            #l.append((matrix.shape[1],entro))
            guess_row = 1
            for p,u,c in zip(ps,unq,counts):
                if u == 242: # (G,G,G,G,G)
                    continue # guess_row += p * 0
                tmp2 = tmp == u
                guess_row += p * wordle(matrix[:,tmp2],index[tmp2],count=c)
                if guess_row > min_:
                    break
            if guess_row < min_:
                min_ = guess_row
                if min_ == 1:
                    mapping[tuple(index)] = min_
                    return min_
    
    if min_ == 1000:
        return wordle(matrix,index,top/1.2,count)
    else:
        mapping[tuple(index)] = min_
        return min_

def evaluate(matrix,index,policy,count,**kways):
    # evaluate how many steps does policy take on average
    if count == 1: return 1
    if count == 2: return 1.5

    # best = np.log2(count)
    row = policy(matrix,index,**kways)
    
    tmp = matrix[row]
    unq,counts = np.unique(tmp,return_counts=True)
    ps = counts/count
    # entro = entropy(ps)
    # if entro == best:
    #     return 2
    # can be 1, when (G,G,G,G,G) or 242
    guess_row = 1
    for p,u,c in zip(ps,unq,counts):
        if u == 242: # (G,G,G,G,G)
            continue # guess_row += p * 0
        tmp2 = tmp == u
        guess_row += p * evaluate(matrix[:,tmp2],index[tmp2],policy,c,**kways)
    return guess_row

def evaluate_depth(matrix,index,policy,count,**kways):
    # pass in depth info to policy
    if count == 1: return 1
    if count == 2: return 1.5

    # best = np.log2(count)
    row = policy(matrix,index,**kways)
    
    tmp = matrix[row]
    unq,counts = np.unique(tmp,return_counts=True)
    ps = counts/count
    # entro = entropy(ps)
    # if entro == best:
    #     return 2
    # can be 1, when (G,G,G,G,G) or 242
    guess_row = 1
    kways['depth'] += 1
    for p,u,c in zip(ps,unq,counts):
        if u == 242: # (G,G,G,G,G)
            continue # guess_row += p * 0
        tmp2 = tmp == u
        guess_row += p * evaluate(matrix[:,tmp2],index[tmp2],policy,c,**kways)
    return guess_row

def evaluate_save(matrix,index,policy,count,log_p,**kways):
    # evaluate how many steps does policy take on average
    # save all (index,val,log_prob) for training NN model
    # return val - 1 to account for bug
    if count == 1: return 1
    if count == 2: return 1.5

    # best = np.log2(count)
    row = policy(matrix,index,**kways)
    
    tmp = matrix[row]
    unq,counts = np.unique(tmp,return_counts=True)
    ps = counts/count
    entro = entropy(ps)
    if entro == np.log2(count): 
        prob = ps[np.where(unq==242)[0]]
        prob = prob if prob.size > 0 else 0
        return 2 - prob
    guess_row = 1
    for p,u,c in zip(ps,unq,counts):
        if u == 242: # (G,G,G,G,G)
            continue # guess_row += p * 0
        tmp2 = tmp == u
        guess_row += p * evaluate_save(matrix[:,tmp2],index[tmp2],policy,c,log_p+np.log(p),**kways)
    out.append((index,guess_row-1,log_p))
    return guess_row

def evaluate_search_save(matrix,index,policy,count,**kways):
    # policy return top k results in a list
    if count == 1: return 1
    if count == 2: return 1.5
    if tuple(index) in policy_map:
        return policy_map[tuple(index)][1]
    
    # best = np.log2(count)
    rows = policy(matrix,index,**kways)
    min_ = 1000
    for row in rows:
        tmp = matrix[row]
        unq,counts = np.unique(tmp,return_counts=True)
        ps = counts/count
        guess_row = 1
        for p,u,c in zip(ps,unq,counts):
            if u == 242: # (G,G,G,G,G)
                continue # guess_row += p * 0
            tmp2 = tmp == u
            guess_row += p * evaluate_search_save(matrix[:,tmp2],index[tmp2],policy,c,**kways)
            if guess_row > min_:
                break
        if guess_row < min_:
            min_ = guess_row
            argmin_ = row
    policy_map[tuple(index)] = (argmin_,min_)
    return min_

def evaluate_saveQ(matrix,index,policy,count,log_p,**kways):
    # evaluate how many steps does policy take on average
    # save all (index,val,log_prob) for training NN model
    # return val - 1 to account for bug
    if count == 1: return 1
    if count == 2: return 1.5

    # best = np.log2(count)
    row = policy(matrix,index,**kways)
    
    tmp = matrix[row]
    unq,counts = np.unique(tmp,return_counts=True)
    ps = counts/count
    entro = entropy(ps)
    if entro == np.log2(count): 
        prob = ps[np.where(unq==242)[0]]
        prob = prob if prob.size > 0 else 0
        return 2 - prob
    guess_row = 1
    for p,u,c in zip(ps,unq,counts):
        if u == 242: # (G,G,G,G,G)
            continue # guess_row += p * 0
        tmp2 = tmp == u
        guess_row += p * evaluate_saveQ(matrix[:,tmp2],index[tmp2],policy,c,log_p+np.log(p),**kways)
    out.append((index,guess_row-1,log_p,row,entro))
    return guess_row

# wordle(matrix,np.arange(2309))
# with open('/home/will/Desktop/LC/wordle/mapping.pkl', 'wb') as f:
#     pickle.dump(mapping, f)

        
def policy_entropy_depth(matrix,index,factor,depth):
    # return the best action given the original matrix and current index
    # assume a value function mapping, tuple(index) -> val
    max_,argmax = -np.Inf,-np.Inf
    count = matrix.shape[1]
    for row in range(12953):
        tmp = matrix[row]
        unq,counts = np.unique(tmp,return_counts=True)
        ps = counts/count
        entro = entropy(ps)
        prob = ps[np.where(unq==242)[0]]
        if prob.size > 0:
            entro += prob[0] * factor * depth
        if entro > max_:
            max_ = entro
            argmax = row
    return argmax

def policy_entropy(matrix,index,factor):
    # return the best action given the original matrix and current index
    # assume a value function mapping, tuple(index) -> val
    max_,argmax = -np.Inf,-np.Inf
    count = matrix.shape[1]
    for row in range(12953):
        tmp = matrix[row]
        unq,counts = np.unique(tmp,return_counts=True)
        ps = counts/count
        entro = entropy(ps)
        prob = ps[np.where(unq==242)[0]]
        if prob.size > 0:
            entro += prob[0] * factor
        if entro > max_:
            max_ = entro
            argmax = row
    return argmax

def policy_entropy_best(matrix,index,factor):
    # use optimal end-game policy
    max_,argmax = -np.Inf,-np.Inf
    count = matrix.shape[1]
    best = np.log2(count)
    only_best = False
    for row in range(12953):
        tmp = matrix[row]
        unq,counts = np.unique(tmp,return_counts=True)
        ps = counts/count
        entro = entropy(ps)
        prob = ps[np.where(unq==242)[0]]
        prob = prob[0] if prob.size>0 else 0
        if only_best:
            if entro == best and prob > max_:
                max_ = prob
                argmax = row
        else:
            if entro == best:
                only_best = True
                max_ = prob
                argmax = row
            else:
                entro += prob * factor
                if entro > max_:
                    max_ = entro
                    argmax = row
    return argmax

def policy_entropy_best_topK(matrix,index,k):
    # use optimal end-game policy
    max_,argmax = -np.Inf,-np.Inf
    count = matrix.shape[1]
    best = np.log2(count)
    only_best = False
    entros = []
    for row in range(12953):
        tmp = matrix[row]
        unq,counts = np.unique(tmp,return_counts=True)
        ps = counts/count
        entro = entropy(ps)
        prob = ps[np.where(unq==242)[0]]
        prob = prob[0] if prob.size>0 else 0
        if only_best:
            if entro == best and prob > max_:
                max_ = prob
                argmax = row
        else:
            if entro == best:
                only_best = True
                max_ = prob
                argmax = row
            else:
                entros.append(entro)
    if only_best:
        return [argmax]
    entros = np.array(entros)
    return np.argsort(entros)[:k]
        
def policy_lookup(matrix,index):
    # return the best action given the original matrix and current index
    # assume a value function mapping, tuple(index) -> val
    min_,argmin = np.Inf,np.Inf
    count = matrix.shape[1]
    best = np.log2(count)
    for row in range(12953):
        tmp = matrix[row]
        unq,counts = np.unique(tmp,return_counts=True)
        ps = counts/count
        entro = entropy(ps)
        if entro == best:
            return row
        elif entro > 0:
            guess_row = 1
            finish_ind = True
            for p,u,c in zip(ps,unq,counts):
                tmp2 = tmp == u
                index2 = index[tmp2]
                len_ = len(index2)
                if tuple(index2) in mapping:
                    guess_row += p * mapping[tuple(index2)]
                # short len_ results were not saved in mapping
                elif len_ == 1:
                    continue
                elif len_ == 2:
                    guess_row += p
                else:
                    finish_ind = False
                    break
            if finish_ind and (guess_row < min_):
                min_ = guess_row
                argmin = row
                if min_ == 1:
                    return argmin
    return argmin


def policy_model(matrix,index,model,words_embed,top,eps):
    count = matrix.shape[1]
    best = np.log2(count)
    threshold = best * top
    best_val = 1000
    for row in range(12953):
        tmp = matrix[row]
        unq,counts = np.unique(tmp,return_counts=True)
        ps = counts/count
        entro = entropy(ps)
        prob = ps[np.where(unq==242)[0]]
        prob = prob if prob.size > 0 else 0
        if entro == best:
            if threshold == best:
                value = 2 - prob # 1 + (1-prob) * 1 + prob * 0
                if value < best_val:
                    best_val = value
                    best_action = row
            else:
                threshold = best # wont consider NN model policy
                best_val = 2 - prob
                best_action = row
        elif entro > threshold:
            index_ = []
            p_ = []
            c_ = []
            value = 1
            for p,u,c in zip(ps,unq,counts):
                # only use model when c > 2
                if c == 1:
                    continue
                if c == 2:
                    value += p * 0.5
                else:
                    p_.append(p)
                    c_.append(c)
                    tmp2 = tmp == u
                    index_.append(index[tmp2])
            # call NN model to eval
            if p_:
                length = torch.tensor(c_,dtype=torch.float32,device='cuda')
                word = torch.tensor(words_embed[np.concatenate(index_)],device='cuda').long()
                with torch.no_grad():
                    out = model((word,length))
                out = out.detach().cpu().numpy()
                value += np.dot(np.array(p_),out)
                if eps > 0:
                    value += eps * best * np.random.randn()
            if value < best_val:
                best_val = value
                best_action = row
    if best_val == 1000:
        return policy_model(matrix,index,model,words_embed,top/1.2,eps)
    else:
        return best_action

# policy_model_vector is just as slow and has a bug that needs to be fixed (does not match policy_model)
# def policy_model_vector(matrix,index,model,words_embed,top,eps,k):
#     # model is a wrapped model that takes c_, index_,words_embed as inputs
#     # and returns a numpy array
#     # k is None, then return argmin
#     count = matrix.shape[1]
#     best = np.log2(count)
#     threshold = best * top
#     best_val = 1000
#     index_ = []
#     p_ = []
#     c_ = []
#     values_ = []
#     actions_ = []
#     counter_ = [0]
#     for row in range(12953):
#         tmp = matrix[row]
#         unq,counts = np.unique(tmp,return_counts=True)
#         ps = counts/count
#         entro = entropy(ps)
#         prob = ps[np.where(unq==242)[0]]
#         prob = prob if prob.size > 0 else 0
#         if entro == best:
#             if threshold == best:
#                 value = 2 - prob # 1 + (1-prob) * 1 + prob * 0
#                 if value < best_val:
#                     best_val = value
#                     best_action = row
#             else:
#                 threshold = best # wont consider NN model policy
#                 best_val = 2 - prob
#                 best_action = row
#         elif entro > threshold:
#             value = 1
#             for p,u,c in zip(ps,unq,counts):
#                 # only use model when c > 2
#                 if c == 1:
#                     continue
#                 if c == 2:
#                     value += p * 0.5
#                 else:
#                     p_.append(p)
#                     c_.append(c)
#                     index_.append(index[tmp == u])
#             actions_.append(row)
#             values_.append(value)
#             counter_.append(len(ps))

#     if threshold==best: return best_action if k is None else [best_action]
#     if len(values_)==0:
#         return policy_model_vector(matrix,index,model,words_embed,top/1.2,eps,k)
    
#     n = len(values_)
#     values_ = np.array(values_)
#     actions_ = np.array(actions_)
#     if index_: # call NN model
#         out = model(c_, index_,words_embed)
#         p_ = np.array(p_)
#         counter_ = np.cumsum(np.array(counter_))
#         for i in range(n):
#             values_[i] += np.dot(out[counter_[i]:counter_[i+1]],p_[counter_[i]:counter_[i+1]])    
#         if eps>0:
#             values_ += eps * best * np.random.randn(n)
            
#     return actions_[np.argmin(values_)] if k is None else actions_[np.argsort(values_)][:k]

# def model_wrap(model):
#     def predict(c_, index_,words_embed):
#         length = torch.tensor(c_,dtype=torch.float32,device='cuda')
#         word = torch.tensor(words_embed[np.concatenate(index_)],device='cuda').long()
#         with torch.no_grad():
#             out = model((word,length))
#         out = out.detach().cpu().numpy()
#         return out
#     return predict

def policy_model_topK(matrix,index,model,words_embed,top,k):
    count = matrix.shape[1]
    best = np.log2(count)
    threshold = best * top
    actions = []
    values = []
    best_val = 1000
    for row in range(12953):
        tmp = matrix[row]
        unq,counts = np.unique(tmp,return_counts=True)
        ps = counts/count
        entro = entropy(ps)
        prob = ps[np.where(unq==242)[0]]
        prob = prob if prob.size > 0 else 0
        if entro == best:
            if threshold == best:
                value = 2 - prob # 1 + (1-prob) * 1 + prob * 0
                if value < best_val:
                    best_val = value
                    best_action = row
            else:
                threshold = best # wont consider NN model policy
                best_val = 2 - prob
                best_action = row
        elif entro > threshold:
            index_ = []
            p_ = []
            c_ = []
            value = 1
            for p,u,c in zip(ps,unq,counts):
                # only use model when c > 2
                if c == 1:
                    continue
                if c == 2:
                    value += p * 0.5
                else:
                    p_.append(p)
                    c_.append(c)
                    tmp2 = tmp == u
                    index_.append(index[tmp2])
            # call NN model to eval
            if p_:
                length = torch.tensor(c_,dtype=torch.float32,device='cuda')
                word = torch.tensor(words_embed[np.concatenate(index_)],device='cuda').long()
                with torch.no_grad():
                    out = model((word,length))
                out = out.detach().cpu().numpy()
                value += np.dot(np.array(p_),out)
            actions.append(row)
            values.append(value)
            
    if threshold == best: return [best_action]
    if len(actions)==0:
        return policy_model_topK(matrix,index,model,words_embed,top/1.2,k)
    actions = np.array(actions)
    values = np.array(values)
    argsort = np.argsort(values)
    return actions[argsort][:k]
    
def policy_model_eps(matrix,index,model,words_embed,top,eps):
    # eps greedy to randomly pick action from > threshold
    count = matrix.shape[1]
    best = np.log2(count)
    threshold = best * top
    best_val = 1000
    random = np.random.rand()<eps
    if random: action_list = []   
    for row in range(12953):
        tmp = matrix[row]
        unq,counts = np.unique(tmp,return_counts=True)
        ps = counts/count
        entro = entropy(ps)
        prob = ps[np.where(unq==242)[0]]
        prob = prob if prob.size > 0 else 0
        if entro == best:
            if threshold == best:
                value = 2 - prob # 1 + (1-prob) * 1 + prob * 0
                if value < best_val:
                    best_val = value
                    best_action = row
            else:
                threshold = best # wont consider NN model policy
                best_val = 2 - prob
                best_action = row
        elif entro > threshold:
            if random:
                action_list.append(row)
            else:
                index_ = []
                p_ = []
                c_ = []
                value = 1
                for p,u,c in zip(ps,unq,counts):
                    # only use model when c > 2
                    if c == 1:
                        continue
                    if c == 2:
                        value += p * 0.5
                    else:
                        p_.append(p)
                        c_.append(c)
                        tmp2 = tmp == u
                        index_.append(index[tmp2])
                # call NN model to eval
                if p_:
                    length = torch.tensor(c_,dtype=torch.float32,device='cuda')
                    word = torch.tensor(words_embed[np.concatenate(index_)],device='cuda').long()
                    with torch.no_grad():
                        out = model((word,length))
                    out = out.detach().cpu().numpy()
                    value += np.dot(np.array(p_),out)
                if value < best_val:
                    best_val = value
                    best_action = row
    if threshold == best: return best_action
    if ((not random) and best_val == 1000) or (random and len(action_list)==0):
        return policy_model_eps(matrix,index,model,words_embed,top/1.2,eps)
    return np.random.choice(action_list) if random else best_action
    
def policy_modelQ(matrix,index,model,words_embed,allowed_words_embed,top,eps):
    count = matrix.shape[1]
    best = np.log2(count)
    threshold = best * top
    best_val = 1000
    actions = []
    for row in range(12953):
        tmp = matrix[row]
        unq,counts = np.unique(tmp,return_counts=True)
        ps = counts/count
        entro = entropy(ps)
        prob = ps[np.where(unq==242)[0]]
        prob = prob if prob.size > 0 else 0
        if entro == best:
            threshold = best # wont consider NN model policy
            value = 2 - prob # 1 + (1-prob) * 1 + prob * 0
            if value < best_val:
                best_val = value
                best_action = row
        elif entro > threshold:
            actions.append(row)
    
    if threshold == best: return best_action
    if actions:
        # call NN model to eval
        k = len(index)
        n = len(actions)
        length = k*torch.ones(n,dtype=torch.float32,device='cuda')
        word = torch.tensor(words_embed[index],device='cuda').long().repeat(n,1)
        actions_tch = torch.tensor(allowed_words_embed[np.array(actions)],device='cuda').long()
        with torch.no_grad():
            out = model((word,length,actions_tch))
        out = out.detach().cpu().numpy()
        if eps > 0:
            out = out + eps * best * np.random.randn(n)
        return actions[np.argmin(out)]

    return policy_modelQ(matrix,index,model,words_embed,allowed_words_embed,top/1.2,eps)
