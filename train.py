from models import YOLO,YOLOLoss
from data.coco import YOLOData
from torch.utils.data import DataLoader

import torch

cfg_file="cfg/yolov4.cfg"
coco_dir=r"F:\coco"
data_type="train2017"

import os
img_dir=os.path.join(coco_dir,"images",data_type)
ann_dir=os.path.join(coco_dir,"annotations",data_type)



def train(model,dataset,loss,optimizer,epoches=1,cuda=True):
    model.train()
    if not torch.cuda.is_available():
        cuda=False
    if cuda:
        model.cuda()
    for epoch in range(epoches):
        running_loss=0.0
        for i,(img,truth) in enumerate(dataset):
            optimizer.zero_grad()
            if cuda:
                img,truth=img.cuda(),truth.cuda()
            
            pred=model(img)
            loss=criterion(pred,truth)
            
            loss.backward()
            optimizer.step()

            running_loss+=loss.item()
            print(loss.item())
            if i%20==19:
                print("Running loss:%f"%(running_loss/20))
                running_loss=.0
                
            torch.save(net.state_dict(),"yolov4.ptn")
            

if __name__ == "__main__":
    cocoset=YOLOData(img_dir,ann_dir)
    coco=DataLoader(cocoset,batch_size=1)
    net=YOLO(cfg_file)
    net.load_weights("weights/yolov4.weights")
    criterion=YOLOLoss()
    optimizer=torch.optim.Adam(net.parameters(),lr=1e-3)
    train(net,coco,criterion,optimizer)
    
