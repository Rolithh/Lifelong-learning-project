# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 11:55:27 2021

@author: Laetitia Haye
"""
import matplotlib.pyplot as plt
import numpy as np
from train import train


#QUAND TRAIN MARCHERA:
#scores, num_epochs, _ = train()

#EXEMPLE A LA MAIN EN ATTENDANT:
scores, num_epochs = {"0" : [3,8,12,10,8,6,5,4,4,3,3,2],
                      "1" : [0,0,2,6,10,13,12,9,6,5,4,3],
                      "2" : [0,1,3,1,2,2,6,11,14,11,8,6],
                      "3" : [0,0,2,1,2,1,1,2,2,5,10,13]}, 3

scores, num_epoch = train()

def plot(scores, num_epochs):
    """plots the scores for all tasks as a function of training time"""


    num_tasks = len(scores.keys())

    plt.title("Training curves for different tasks learnt sequentially")
    plt.xlabel("Training time")
    plt.ylabel("Score")

    for key in scores:
        plt.plot(range(1,num_tasks*num_epochs+1),scores[key],label="task "+str(key))

    plt.ylim((0,15.0)) #remettre entre 0 et 1 qd pb train sera reglé
    plt.xticks(np.arange(1, num_tasks*num_epochs+1, 1.0))

    #trace une verticale lorsque la tâche en cours d'apprentissage change
    x_epochs = [i*num_epochs for i in range(1, num_tasks)]
    for x in x_epochs:
        plt.axvline(x, linestyle = "--", c="gray")

    plt.legend()
    plt.show()


plot(scores, num_epochs)
