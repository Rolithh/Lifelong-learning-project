from dataloader import ContinualJacquardLoader
from model import load_model, minMSELoss

data_path = 'C:/Users/alexa/OneDrive/Documents/Cours/Cours 3A/MSO - Informatique/Projet/Jacquard/'


def train():
    
    model = load_model()
    
    data_loader = ContinualJacquardLoader(data_path)
    
    optimizer = torch.optim.SGD(model.parameters, lr=0.0005)
    
    for task_i, task_data in enumerate(data_loader):
        
        for epoch in range(n_epochs):
            
            for i, batch in enumerate(task_data):
                
                x, y_gt = batch
                
                optimizer.zero_grad()
                
                y_pred = model(x)
                
                loss = minMSELoss(y_pred, y_gt)
                
                loss.backward()
                optimizer.step()