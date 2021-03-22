# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 11:15:25 2021

@author: Laetitia Haye
"""

from data_loader import ContinualJacquardLoader
from data_loader2 import ContinualJacquardLoader2
from model import load_model, minMSELoss
import torch
import imagecodecs
from sklearn.preprocessing import StandardScaler

data_path = './Samples/'





def train(n_epochs=3):
    print("###############################FONCTION TRAIN###################################")

    train_data_loader = ContinualJacquardLoader(data_path)
    test_data_loader = ContinualJacquardLoader2(data_path)
    model = load_model()
    print("-------------------------------LOAD OK")

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0005)
    print("-------------------------------OPTIMIZER OK")
    for elem in train_data_loader:
        print(elem)
    print(train_data_loader)

    scores = {}

    for task_i, task_data in enumerate(train_data_loader):
        #there are n_task objects in dataloader (one per task)
        #task_data is a torch.utils.data.dataloader.DataLoader object
        #each task_data object contains several batches

        print('\n ----------  training task ', task_i, '  ---------- \n')

        for epoch in range(n_epochs):

            print('\n ------  epoch', epoch, '  ------ \n')


            print('\n training model...  \n')
            for i, batch in enumerate(task_data):

                print('\n ----- batch ', i, ' ----- \n')

                x, y_gt = batch

                optimizer.zero_grad()

                y_pred = model(x)


                loss = minMSELoss(y_pred, y_gt)


                #print(loss)

                loss.mean().backward()
                optimizer.step()



    print('\n computing scores on each task... \n')
    print("=======================================================================================")
    for elem in test_data_loader:
        print(elem)
    print(test_data_loader)

    for task_j, task_test_data in enumerate(test_data_loader):
        print('\n ----------  training task ', task_j, '  ---------- \n')

        for epoch in range(n_epochs):

            print('\n ------  epoch', epoch, '  ------ \n')


            print('\n training model...  \n')
            for i, batch in enumerate(task_test_data):

                print('\n ----- batch ', i, ' ----- \n')

                x_test, y_gt_test = batch
                optimizer.zero_grad()

                y_pred_test = model(x_test)

                loss_test = minMSELoss(y_pred_test, y_gt_test)
                #print(loss)

                loss_test.mean().backward()
                optimizer.step()


        print("================================== LOSS ============================")
        print(loss_test)
        if task_j in scores:
            scores[task_j].append(loss_test)
        else:
            scores.setdefault(task_j ,[loss])

    return scores, n_epochs, model


if __name__ == "__main__":

    scores, num_epochs, _ = train()




######### details


#x is a tensor containing 16 arrays (one per obj in the batch)
#each array contains 4 arrays (one per orientation)
#each orientation has 224 arrays, and each array has 224 elements itself

#y_gt is a tensor containing 16 arrays (one per obj in the batch)
#each array contains a single array, itself containing a \
#random number (usually between 20 and 140) of arrays of length 5

#y_pred contains 16 tensors (one per obj in the batch), each tensor is of length 5


#print('\n --XXXXXXX-- ', len(x), ' --XXXXXXX--- \n')
#print('\n --XXXXXXX-- ', len(x[0]), ' --XXXXXXX--- \n')
#
#print('\n --Y_GT--- ', len(y_gt[0]) , '--Y_GT--- \n')
#print('\n --Y_GT--- ', len(y_gt[0][0]) , '--Y_GT--- \n')
#
#print('\n --Y_PRED--- ', len(y_pred) , '--Y_PRED---\n')
#print('\n --Y_PRED--- ', y_pred[0] , '--Y_PRED--- \n')

