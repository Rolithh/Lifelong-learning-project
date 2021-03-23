# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 17:20:04 2021

@author: Laetitia Haye
"""

import matplotlib.pyplot as plt
import numpy as np
from LwFtrain import train




model, loss_data = train()

#EXEMPLE A LA MAIN EN ATTENDANT:
#loss_data = {0: [0.4731554388999939, 0.5600252151489258, 0.4650968909263611, 0.47342050075531006], \
#             1: [0.48507168889045715, 0.48503175377845764, 0.48527178168296814, 0.4851912260055542], \
#             2: [0.43074414134025574, 0.5298635363578796, 0.4564273953437805, 0.5675736665725708], \
#             3: [0.3524559736251831, 0.5255559682846069, 0.5425003170967102, 0.5980441570281982], \
#             4: [0.5002911686897278, 0.4616524279117584, 0.5269559025764465, 0.4176923930644989]}


def plot(data):
    """plots the loss as a function of training time"""
    print("................... plotting.......................")

    num_tasks = len(data.keys())
    num_epochs = len(data[0])

    plt.title("Loss during training of different tasks learnt sequentially")
    plt.xlabel("Training time")
    plt.ylabel("Loss")

    y = []
    for key in data:
        y = y + data[key]
        
     
    plt.plot(range(1, len(y)+1, 1), y)

    plt.ylim((0,1)) #remettre entre 0 et 1 qd pb train sera reglé
    plt.xticks(np.arange(1, num_tasks*num_epochs+1, 1.0))

    #trace une verticale lorsque la tâche en cours d'apprentissage change
    x_epochs = [i*num_epochs for i in range(1, num_tasks)]
    for x in x_epochs:
        plt.axvline(x, linestyle = "--", c="gray")

    plt.legend()
    plt.show()


plot(loss_data)
