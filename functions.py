#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 11:04:17 2022

@author: will
"""
import numpy as np
matrix = np.load('matrix.npy')
mapping = dict()

def entropy(p):
    return np.sum(-p*np.log2(p))
    
def wordle(matrix,index,top=0.5,count=2309):
    # return min expected guesses
    d = matrix.shape[1]
    if d == 1: return 0
    if d == 2: return 1
    if tuple(index) in mapping:
        return mapping[tuple(index)]
    threshold = np.log2(count) * top
    min_ = 1000
    for row in range(12953):
        unq,unqtags,counts = np.unique(matrix[row],return_inverse=True,return_counts=True)
        ps = counts/count
        entro = entropy(ps)
        if entro>threshold:
            guess_row = 1
            for p,u,c in zip(ps,unq,counts):
                tmp = unqtags==u
                guess_row += p * wordle(matrix[:,tmp],index[tmp],count=c)
            if guess_row < min_:
                min_ = guess_row
                if min_ == 1:
                    mapping[tuple(index)] = min_
                    return min_
    mapping[tuple(index)] = min_
    return min_

wordle(matrix,np.arange(2309))