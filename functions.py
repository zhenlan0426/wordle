#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 11:04:17 2022

@author: will
"""
import numpy as np
matrix = np.load('/home/will/Desktop/LC/wordle/matrix.npy')
# %timeit groupby1(matrix,5)
# %timeit groupby2(matrix,5)
# %timeit groupby1(matrix[:,:500],1)
# %timeit groupby2(matrix[:,:500],1)
# %timeit groupby1(matrix[:,:100],1)
# %timeit groupby2(matrix[:,:100],1)

mapping = dict()
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
    if count == 1: return 0
    if count == 2: return 1
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
        if entro == best:
            mapping[tuple(index)] = 1
            return 1
        elif entro>threshold:
            #l.append((matrix.shape[1],entro))
            guess_row = 1
            for p,u,c in zip(ps,unq,counts):
                tmp2 = tmp == u
                guess_row += p * wordle(matrix[:,tmp2],index[tmp2],count=c)
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

wordle(matrix,np.arange(2309))
