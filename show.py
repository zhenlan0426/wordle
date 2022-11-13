#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 07:51:38 2022

@author: will
"""
import pickle
with open('/home/will/Desktop/LC/wordle/policy_map.pkl', 'rb') as f:
    policy_map = pickle.load(f)
    
with open('/home/will/Desktop/LC/wordle/allowed_words_dict.pkl', 'rb') as f:
    allowed_words_dict = pickle.load(f)

with open('/home/will/Desktop/LC/wordle/possible_words_dict.pkl', 'rb') as f:
    possible_words_dict = pickle.load(f)
    
color_dict = {'b':0,'y':1,'g':2}
def c2num(color):
    sum_,prod_ = 0,1
    for c in color:
        sum_ += color_dict[c] * prod_
        prod_ *= 3
    return sum_


class wordle_game():
    def __init__(self,s,matrix):
        self.s0 = s
        self.s = s
        self.matrix0 = matrix
        self.matrix = matrix
        
    def reset(self):
        self.s = self.s0
        self.matrix = self.matrix0
        
    def get_action_num(self):
        return policy_map[tuple(self.s)][0]
    
    def get_action(self):
        return allowed_words_dict[self.get_action_num()]
    
    def update_color(self,color):
        color = c2num(color)
        tmp = self.matrix[self.get_action_num()]
        tmp2 = tmp == color
        self.s = self.s[tmp2]
        self.matrix = self.matrix[:,tmp2]
        if len(self.s)>2:
            return self.get_action()
        return [possible_words_dict[i] for i in self.s]
    
    
    
    
    
    
    
    
    
    
    