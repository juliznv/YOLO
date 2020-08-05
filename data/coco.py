from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as tfs
import os
import torch
from PIL import Image
img_tfs=tfs.Compose([
    tfs.Resize((608,608)),
    tfs.ToTensor()
])

class YOLOData(Dataset):
    def __init__(self,img_dir,ann_dir,transform=img_tfs,target_transform=None,num_max_obj=100):
        super(YOLOData,self).__init__()
        self.num_max_obj=num_max_obj
        self.img_dir=img_dir
        self.ann_dir=ann_dir
        self.transform=transform
        self.target_transform=target_transform
        self.ids=[f.split('.')[0] for f in os.listdir(ann_dir)]
        
    
    def __getitem__(self,index):
        img_path=os.path.join(self.img_dir,self.ids[index]+".jpg")
        target_path=os.path.join(self.ann_dir,self.ids[index]+".csv")
        img=Image.open(img_path).convert("RGB")
        target=torch.zeros(self.num_max_obj,5)
        af=open(target_path,"r")
        anns=af.readlines()
        af.close()
        for i,a in enumerate(anns):
            bbox=[float(x) for x in a.split(",")]
            target[i,:]=torch.FloatTensor(bbox)
        if self.transform:
            img=self.transform(img)
        if self.target_transform:
            target=self.target_transform(target)
            
        return img,target

    def __len__(self):
        return len(self.ids)


# img_dir=r"E:\code\coco\images\train2017"
# ann_dir=r"E:\code\coco\detections"
# coco=YOLOData(img_dir,ann_dir)


def plot_bbox(img,bbox):
    
    import cv2
    import numpy as np
    img=np.copy(img)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    for b in bbox:
        x1=int(b[0]-b[2]/2)
        y1=int(b[1]-b[3]/2)
        x2=int(b[0]+b[2]/2)
        y2=int(b[1]+b[3]/2)
        img=cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),1)
    cv2.imshow("",img)
    cv2.waitKey()
"""   
img,bbox=coco[0]
plot_bbox(img,bbox)

"""



"""
from torchvision.datasets.coco import CocoDetection
from torch.utils.data import DataLoader

import torchvision.transforms as tfs

import os.path as osp
coco_dir="E:\\code\\coco"
data_type="train2017"
img_dir=osp.join(coco_dir,"images",data_type)
ann_file=osp.join(coco_dir,"annotations","instances_{}.json".format(data_type))

img_tfs=tfs.Compose([
    tfs.Resize((608,608)),
    tfs.ToTensor()
])

cocoset=CocoDetection(img_dir,ann_file,transform=img_tfs)
coco=DataLoader(cocoset,batch_size=1)
print(len(coco))
save_dir="E:\\code\\coco\\detections"
for i,(img,target) in enumerate(cocoset):
    if not len(target):
        continue
    img_file=str(target[0]["image_id"])
    img_file=(12-len(img_file))*"0"+img_file+".txt"
    tfile=osp.join(save_dir,img_file)
    with open(tfile,"w") as tf:
        for t in target:
            line=""
            for c in t["bbox"]:
                line+=str(c)+"\t"
            line+=str(t["category_id"])+"\n"
            tf.write(line)
print("All done.")
"""