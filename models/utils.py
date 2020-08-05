# -*- coding:utf-8 -*-

import torch

def bbox_xyxy(bbox):
    xyxy=torch.zeros_like(bbox)
    xyxy[...,:2]=bbox[...,:2]-bbox[...,2:]/2
    xyxy[...,2:]=bbox[...,:2]+bbox[...,2:]/2
    return xyxy

def bbox_xywh(bbox):
    xywh=torch.zeros_like(bbox)
    xywh[...,:2]=(bbox[...,:2]+bbox[...,2:])/2
    xywh[...,2:]=(bbox[...,2:]-bbox[...,:2]).clamp(0)
    return xywh

def bbox_iou(bbox_a,bbox_b,mode="iou"):
    modes=["iou","giou","diou","ciou"]
    if mode not in modes:
        raise ValueError(mode+" not support. Support \" %s \"."%(", ".join(modes)))

    min_i=torch.max(bbox_a[:,None,:2],bbox_b[:,:2])
    max_i=torch.min(bbox_a[:,None,2:],bbox_b[:,2:])
    wh_i=(max_i-min_i).clamp(0)
    area_i=wh_i[...,0]*wh_i[...,1]

    xywh_a=bbox_xywh(bbox_a)
    xywh_b=bbox_xywh(bbox_b)
    area_a=xywh_a[...,2]*xywh_a[...,3]
    area_b=xywh_b[...,2]*xywh_b[...,3]
    
    iou=area_i/(area_a[:,None]+area_b-area_i)
    if mode=="iou":
        return iou
    
    min_c=torch.min(bbox_a[:,None,:2],bbox_b[:,:2])
    max_c=torch.max(bbox_a[:,None,2:],bbox_b[:,2:])
    area_c=torch.prod(max_c-min_c,2)
    
    giou=iou-(area_c-area_a[:,None]-area_b+area_i)/area_c
    if mode=="giou":
        return giou
    
    dd=torch.sum((xywh_a[:,None,:2]-xywh_b[:,:2])**2,dim=-1)
    dc=torch.sum((max_c-min_c)**2,dim=-1)

    diou=iou-dd/dc
    if mode=="diou":
        return diou

    from math import pi
    atan_a=torch.atan2(xywh_a[...,2],xywh_a[...,3])
    atan_b=torch.atan2(xywh_b[...,2],xywh_b[...,3])
    v=4*(atan_b-atan_a[:,None])**2/(pi**2)
    alpha=v/(1-iou+v)

    ciou=diou-alpha*v
    if mode=="ciou":
        return ciou

def nms(bbox,conf,nms_thresh=0.5):
    keep=[]
    cidx=torch.argsort(conf,-1,True)
    while len(cidx):
        keep+=[cidx[0].item()]
        iou=bbox_iou(bbox[cidx[0]][None,:],bbox[cidx[1:]]).squeeze(0)
        mask=(iou<=nms_thresh).float().nonzero().squeeze(1)
        cidx=cidx[mask+1]
    return keep

def post_process(pred,conf_thresh=0.5,nms_thresh=0.4,num_classes=80):
    max_conf,max_index=torch.max(pred[...,4:],-1)
    pred=torch.cat((pred[...,:4],max_conf.unsqueeze(2),max_index.float().unsqueeze(2)),-1)
    bboxes=[]
    for i in range(pred.size(0)):
        box_conf=pred[i][(pred[i][...,4]>conf_thresh).float().nonzero().squeeze(1)]
        keep=nms(bbox_xyxy(box_conf[:,:4]),box_conf[:,4],nms_thresh)
        bboxes+=[box_conf[keep]]
    return bboxes

def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names

def plot_boxes_cv2(img,bboxes,save_path,class_name,color=None):
    import cv2
    height,width,_=img.shape
    for i in range(len(bboxes)):
        rgb=(i/len(bboxes)*255,i/len(bboxes)*255,i/len(bboxes)*255)
        x1=int((bboxes[i,0]-bboxes[i,2]/2.0)*width)
        y1=int((bboxes[i,1]-bboxes[i,3]/2.0)*height)
        x2=int((bboxes[i,0]+bboxes[i,2]/2.0)*width)
        y2=int((bboxes[i,1]+bboxes[i,3]/2.0)*height)
        cidx=int(bboxes[i,5])
        img=cv2.putText(img,class_name[cidx],(x1,y1),\
            cv2.FONT_HERSHEY_SIMPLEX,1.2,rgb,1)
        img=cv2.rectangle(img,(x1,y1),(x2,y2),rgb,1)
    
    cv2.imwrite(save_path,img)
