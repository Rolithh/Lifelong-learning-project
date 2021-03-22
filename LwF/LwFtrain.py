# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 09:30:11 2021

@author: Laetitia Haye
"""


#from data_loader_light2 import ContinualJacquardLoader
from data_loader import ContinualJacquardLoader

from LwFmodel import load_model, create_new_layer, minMSELoss
import torch
import copy
import torch.utils.data


path = '/home/projet17/Jacquard_light2'
data_path = 'C:/Users/Laetitia Haye/Documents/ECL2/projet info Lifelong learning/jac'


def train(n_epochs=4):
    
#    train_data_loader = ContinualJacquardLoader(path, 1)
    train_data_loader = ContinualJacquardLoader(data_path)

    
    #load a model already trained on a single task
    new_model = load_model()
    
    params_to_optimize = new_model.parameters() 
    optimizer = torch.optim.SGD(params_to_optimize, lr= 0.001,\
                                momentum=0.9, weight_decay=0.0005)
    
    for task_i, task_data in enumerate(train_data_loader):
        #there are n_task objects in dataloader (one per task)
        #task_data is a torch.utils.data.dataloader.DataLoader object
        #each task_data object contains several batches
        
        #copy the model for previous tasks
        original_model = copy.deepcopy(new_model)
        #add some task-specific neurones for the new task
        new_model = create_new_layer(original_model)
        
        nb_old_neurones = original_model.classifier[-1].out_features
        nb_new_neurones = new_model.classifier[-1].out_features
        print('\n -- nb de neurones sur la dernière couche avant et après \
              apprentissage de la tâche en cours : ', nb_old_neurones , ' et ', \
              nb_new_neurones, '--\n')
          
        
        #Training
        print('\n ----------  training task {}/{}  ---------- \n'.format(task_i, len(train_data_loader)))
        
        for epoch in range(n_epochs):
                    
            print('\n ------  epoch {}  (out of {})  ---------- \n'.format(epoch, n_epochs))
            
            
            print('\n training model...  \n')                        
            for i, batch in enumerate(task_data):
                
                print('\n ----- batch ', i, ' ----- \n')
                
                # Normalization vector
                norm_vect = torch.tensor([1024,  1024,  180,  1024, 1024])
                
                # norm_vect repeated number of already trained tasks times
                nb_old_tasks = int(nb_old_neurones / 5)
                long_norm_vect = torch.cat([norm_vect] * nb_old_tasks)
                
                # Get training data and ground truth for the new task
                xn, yn_gt = batch
                
                # Record output of old tasks for new data
                yo_target = original_model(xn)
                yo_target_normalized = yo_target / long_norm_vect
                
                # Compute output for new data with new parameters
                yall_pred = new_model(xn)
                
                #G et output of old tasks (for new data with new parameters)
                yo_pred = yall_pred.narrow(1,0,len(yall_pred[0])-5) #(all elements except last 5)
                yo_pred_normalized = yo_pred / long_norm_vect
                
                # Get output of new tasks (for new data with new parameters)
                yn_pred = yall_pred.narrow(1,len(yall_pred[0])-5, 5) #(only last 5 elements)
                yn_pred_normalized = yn_pred / norm_vect
                               
                print('\n --Yo target--- ', yo_target[0] , '--Yo target--- \n') 
                print('\n --Yo pred--- ', yo_pred[0] , '--Yo pred--- \n') 
                print('\n --Yn pred--- ', yn_pred[0] , '--Yn pred-- \n')
                
                # Set grads to 0
                optimizer.zero_grad()
                                
                # Compute loss 
                    # Apply loss to all old tasks
                old_loss = minMSELoss(yo_pred_normalized, yo_target_normalized)
                    # Compute loss for current task
                for k in range (len(yn_gt)):
                    yn_gt[k] = yn_gt[k] / norm_vect
                new_loss = minMSELoss(yn_pred_normalized, yn_gt)
                
                loss = old_loss + new_loss
                
                print("loss : ", loss)

                # Do backward
                loss.mean().backward()
                
                # Update weights
                optimizer.step()
        
        #save model
        torch.save(new_model, "model{}".format(task_i))
        print("Saved model {}".format(task_i))
        
    return new_model

              

if __name__ == "__main__":
    
    new_model = train()



######### LwF

                
#Given a CNN with shared parameters θs and task-specific parameters θo , 
#our goal is to add task-specific parameters θn for a new task and to learn 
#parameters that work well on old and new tasks, using images and labels
#from only the new task

#First, we record responses yo on each new-task image
#from the original network for outputs on the old tasks
#Yo ← CNN(Xn, θs, θo)
    
#Next, we train the network to minimize loss for all tasks (and regularization R -> weight decay)
#θ∗s , θ∗o , θ∗n ← argmin [θˆs,θˆo,θˆn] (λo Lold(Yo, Yˆo) + Lnew(Yn, Yˆn) + R(ˆθs,ˆθo, θn)
    
#For new tasks, the loss encourages predictions ˆyn to be consistent with the
#ground truth yn. 
#Yˆn ≡ CNN(Xn,ˆθs,ˆθn) and Lnew(Yn, Yˆn)  
    
#For each original task, we want the output probabilities
#for each image to be close to the recorded output from the
#original network.
    
#Each time a new task is added, the responses of all other
#tasks Yo are re-computed, to emulate the situation where
#data for all original tasks are unavailable. Therefore, Yo for
#older tasks changes each time.
#    
    
#### didn't do
#adapt distillation loss to regression (not necessary)
#warm-up step (fine-tuning on θn only to improve efficiency)
# what difference between optimizer.zero_grad() and model.zero_grad()?
#lambda_0 (no value given in the paper)

####useless
#print('\n --Yall pred--- ', yall_pred[0] , '--Yall pred--- \n')
#from data_loader2 import ContinualJacquardLoader2
#path2 = '/home/projet17/jac'
#    test_data_loader = ContinualJacquardLoader2(data_path)
# print('\n computing scores on each task... \n')
    
  



