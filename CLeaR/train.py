from clear import CLeaR
from data_loader_light2 import ContinualJacquardLoader
from time import sleep



#path = "/content/drive/My Drive/Jacquard_light2/"
path = "../Jacquard_light2/"
dataloader = ContinualJacquardLoader(path, 1)

model = CLeaR()

warmup = True
for task, task_dl in enumerate(dataloader):
    

    print("\n\nTask {}/{}\n".format(task, len(dataloader)))
    model.train_on_task(task_dl, warmup)

    model.save_alexnet("model{}".format(task))
    
    warmup = False