import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.utils import bbox_iou,bbox_xywh,bbox_xyxy



class YOLOLoss(nn.Module):
    def __init__(self,thresh=0.5,num_classes=80,
            img_size=608,cuda=True):
        super(YOLOLoss,self).__init__()

        self.num_classes=num_classes
        self.thresh=thresh
        self.img_size=img_size
        self.use_cuda=cuda
        self.anchors=[[12,16],[19,36],[40,28],[36,75],
            [76,55],[72,146],[142,110],[192,243],[459,401]]
        self.mask=[[0,1,2],[3,4,5],[6,7,8]]
        self.abbox=torch.zeros(len(self.anchors),4)
        self.abbox[:,2:]=torch.Tensor(self.anchors)

    def forward(self,preds,truths):
        loss,loss_xy,loss_wh,loss_obj,loss_cls=(.0,)*5
        
        for i,pred in enumerate(preds):
            B,_,H,W=pred.size()
            num_obj=5+self.num_classes
            

            pred=pred.permute(0,2,3,1).reshape(B,H,W,3,num_obj)
            pred[...,:2]=torch.sigmoid(pred[...,:2])
            pred[...,4:]=torch.sigmoid(pred[...,4:])
            
            obj_mask,scale_wh,target=self.build_target(pred[...,:4],truths,B,H,W,i)
            if self.use_cuda:
                obj_mask=obj_mask.cuda()
                scale_wh=scale_wh.cuda()
                target=target.cuda()

            pred[...,2:4]*=scale_wh
            pred[...,:4]*=obj_mask
            pred[...,5:]*=obj_mask
            target[...,2:4]*=scale_wh

            loss_xy+=F.binary_cross_entropy(pred[...,:2],target[...,:2],reduction="sum")
            loss_wh+=F.mse_loss(pred[...,2:4],target[...,2:4],reduction="sum")
            loss_obj+=F.binary_cross_entropy(pred[...,4],target[...,4],reduction="sum")
            loss_cls+=F.binary_cross_entropy(pred[...,5:],target[...,5:],reduction="sum")
        loss=loss_xy+loss_wh+loss_obj+loss_cls
            
        return loss

    def build_target(self,pbbox,truth,B,H,W,out_i,num_obj=85):
        truth=truth.cpu()
        obj_mask=torch.zeros((B,H,W,3,1))
        target=torch.zeros(B,H,W,3,num_obj)
        scale_wh=torch.zeros(B,H,W,3,1)

        num_tgt=(truth.sum(dim=2)>0).sum(dim=1)
        stride=self.img_size/H

        for b in range(B):
            tbbox=truth[b,:num_tgt[b],:4]*self.img_size
            tbbox_wh=torch.zeros_like(tbbox)
            tbbox_wh[...,2:]=tbbox[...,2:]
            ious=bbox_iou(tbbox_wh,self.abbox,mode="ciou")
            best_match=ious.argmax(dim=1)
            best_match_mask=((best_match==self.mask[out_i][0])|
                (best_match==self.mask[out_i][1])|
                (best_match==self.mask[out_i][2]))
            if not sum(best_match_mask):
                continue
            for ti in range(best_match.shape[0]):
                if best_match_mask[ti]:
                    cx,cy=(tbbox[ti,:2]/stride).int().numpy()
                    a=best_match[ti]%3
                    obj_mask[b,cy,cx,a]=1

                    target[b,cy,cx,a,0]=tbbox[ti,0]/stride-cx
                    target[b,cy,cx,a,1]=tbbox[ti,1]/stride-cy
                    target[b,cy,cx,a,2]=torch.log(tbbox[ti,2]/self.anchors[best_match[ti]][0])
                    target[b,cy,cx,a,3]=torch.log(tbbox[ti,3]/self.anchors[best_match[ti]][1])

                    target[b,cy,cx,a,4]=1
                    target[b,cy,cx,a,5+truth[b,ti,4].int().numpy()]=1
                    scale_wh[b,cy,cx,a,:]=2-target[b,cy,cx,a,2]*target[b,cy,cx,a,3]
                    
        
        return obj_mask,scale_wh,target

