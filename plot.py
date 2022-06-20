# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 16:58:50 2022

@author: lxi12
"""
## Plot heatmap of adjacency matrix and normalized adj
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

A = [[0,1,1,1],[1,0,1,0],[1,1,0,0],[1,0,0,0]]
 
plt.figure(figsize=(4,4))
sns.heatmap(A, linewidth=1, annot=True)
plt.show()

A_t = A + np.identity(4)
D_t = [[4,0,0,0],[0,3,0,0],[0,0,3,0],[0,0,0,2]]

D_v=[[0.5,0,0,0],[0,0.577,0,0],[0,0,0.577,0],[0,0,0,0.707]]

A_h = D_v @ A_t @ D_v

plt.figure(figsize=(4,4))
sns.heatmap(A_h, linewidth=1, annot=True)
plt.show()