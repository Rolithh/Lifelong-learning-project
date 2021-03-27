# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 11:15:25 2021

@author: Laetitia Haye
"""
from tqdm import tqdm
from data_loader_light2 import ContinualJacquardLoader
from model import load_model, minMSELoss
import torch

data_path = '../Jacquard_light2'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def train(n_epochs=256):
    print("###############################FONCTION TRAIN###################################")

    data_loader = ContinualJacquardLoader(data_path)
    model = load_model().to(device)
    print("-------------------------------LOAD OK")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.001, betas=(0.9,0.999))
    print("-------------------------------OPTIMIZER OK")


    scores = []

    for task_i, task_data in enumerate(data_loader):
        #there are n_task objects in dataloader (one per task)
        #task_data is a torch.utils.data.dataloader.DataLoader object
        #each task_data object contains several batches

        print('\n ----------  training task ', task_i, '  ---------- \n')
        for epoch in tqdm(range(n_epochs)):

           
            for i, batch in enumerate(task_data):


                x, y_gt = batch
                x = x.to(device)
                y_gt = [yi.to(device) for yi in y_gt]

                optimizer.zero_grad()

                y_pred = model(x)


                loss = minMSELoss(y_pred, y_gt, device)


                #print(loss)

                loss.backward()
                optimizer.step()


        torch.save(model.state_dict(), "model{}".format(task_i))
        print('\n computing scores on each task... \n')
        print("=======================================================================================")
    
        score_task = []
        for task_j in range(len(data_loader)):
            task_test_data = data_loader.get_data('test', task_j)
    
    
            loss_test = 0
            for i, batch in enumerate(task_test_data):
    
    
                x_test, y_gt_test = batch
                x_test = x_test.to(device)
                y_gt_test = [yi.to(device) for yi in y_gt_test]
    
                with torch.no_grad():
                    y_pred_test = model(x_test)
    
                    loss_test += minMSELoss(y_pred_test, y_gt_test, device).item()
    
    
    
    
    
            print("================================== LOSS ============================")
            loss_test /= len(task_test_data)
            score_task.append(loss)
            print(loss_test)
            
    

        scores.append(score_task)
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

