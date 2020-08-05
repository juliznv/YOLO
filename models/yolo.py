# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import *

# 激活函数，支持Mish、LeakyReLU、Linear
class Activation(nn.Module):
    def __init__(self,atype="mish"):
        super(Activation,self).__init__()
        self.atype=atype

    def forward(self,x):
        if self.atype=="mish":
            return x*(torch.tanh(F.softplus(x)))
        elif self.atype=="relu":
            return F.relu(x,inplace=True)
        elif self.atype=="leaky":
            return F.leaky_relu(x,0.1,inplace=True)
        elif self.atype=="linear":
            return x

class EmptyLayer(nn.Module):
    r"""
    占位层，shortcut、route占位
    """
    def __init__(self):
        super(EmptyLayer,self).__init__()

# YOLO头，训练时计算误差，测试时输出预测结果
class Head(nn.Module):
    r"""
    YOLO
    """
    def __init__(self,anchors,num_classes=80,img_size=608):
        super(Head,self).__init__()
        self.anchors=anchors
        self.num_classes=num_classes
        self.img_size=img_size
        self.thresh=0.5
        self.seen=0

    def forward(self,x):
        if self.training:
            return x
            
        B,_,H,W=x.size()
        stride=self.img_size/H
        anchors=[[a[0]/stride,a[1]/stride] for a in self.anchors]
        num_anchors=len(anchors)
        num_obj=5+self.num_classes
        
        x=x.permute(0,2,3,1).reshape(B,H,W,num_anchors,num_obj).contiguous()
        
        #xy
        x[...,:2]=torch.sigmoid(x[...,:2])
    
        grid_x,grid_y=torch.arange(W),torch.arange(H)
        grid_x,grid_y=torch.meshgrid(grid_x,grid_y)
    
        grid_x=grid_x.reshape(1,H,W,1,1).repeat(B,1,1,num_anchors,1)
        grid_y=grid_y.reshape(1,H,W,1,1).repeat(B,1,1,num_anchors,1)
        cxy=torch.cat((grid_y,grid_x),-1).cuda()
        x[...,:2]+=cxy

        # wh
        pwh=torch.tensor(anchors).reshape(1,1,1,3,2).repeat(B,H,W,1,1).cuda()
        x[...,2:4]=torch.exp(x[...,2:4])*pwh

        # conf
        x[...,4]=torch.sigmoid(x[...,4])
        # cls
        x[...,5:]=torch.sigmoid(x[...,5:])

        x=x.view(B,-1,num_obj)
        x[...,5:]*=x[...,4:5]
        x=torch.cat((x[...,:4],x[...,5:]),-1)
        x[...,:4]/=W
        return x
    

def parse_cfg(cfg_file):
    r"""
    解析配置文件网络配置项
    * cfg_file:str    网络配置文件路径
    - layers:list [layer:dict]    返回网络每层的配置项
    """
    # 按行读取
    with open(cfg_file,"r") as cf:
        lines=cf.readlines()
    layers=[]
    layer={}
    for line in lines:
        # 去除空行和注释行
        line=line.replace(' ','').replace('\n','')
        if not len(line) or line[0]=='#':
            continue

        if line[0]=='[':
            if len(layer):
                # 保存上一层配置项
                layers+=[layer]
                layer={}
            layer["type"]=line[1:-1]
        else:
            # 保存当前层配置项
            line=line.split('=')
            layer[line[0]]=line[1]
    layers+=[layer] # 保存最后一层配置项
    return layers

def create_modules(layers):
    r"""
    根据网络每层配置项，创建对应网络层
    * layers:list[layer:dict]   网络配置项，每个dict元素表示一层配置项
    - modules: ModuleList   返回依序构建的网络层对象
    """
    modules=nn.ModuleList()
    net_info=layers[0]
    layers=layers[1:]
    in_channels=int(net_info["channels"])
    outs=[]
    for i,layer in enumerate(layers):
        module=nn.Sequential()
        # 创建卷积层（卷积、批归一化、激活）
        if layer["type"]=="convolutional":
            out_channels=int(layer["filters"])
            kernel_size=int(layer["size"])
            stride=int(layer["stride"])
            pad=(kernel_size-stride+1)//2 if int(layer["pad"]) else 0
            try:
                bn=int(layer["batch_normalize"])
            except:
                bn=0

            if bn:
                bias=False
                bn=nn.BatchNorm2d(out_channels)
            else:
                bias=True
                bn=0
            conv=nn.Conv2d(in_channels,out_channels,kernel_size,stride,pad,bias=bias)
            module.add_module("conv_%d"%i,conv)
            if bn:
                module.add_module("bn_%d"%i,bn)
            module.add_module(layer["activation"]+"_%d"%i,Activation(layer["activation"]))
            
            in_channels=out_channels
        # 创建上采样层
        elif layer["type"]=="upsample":
            stride=int(layer["stride"])
            upsample=nn.Upsample(scale_factor=stride,mode="nearest")
            module.add_module("upsample_%d"%i,upsample)
        # 创建最大池化层
        elif layer["type"]=="maxpool":
            stride=int(layer["stride"])
            kernel_size=int(layer["size"])
            pad=(kernel_size-stride+1)//2
            maxp=nn.MaxPool2d(kernel_size,stride,pad)
            module.add_module("maxpool_%d"%i,maxp)
        # 创建残差连接层，无计算使用EmptyLayer占位
        elif layer["type"]=="shortcut":
            module.add_module("shortcut_%d"%i,EmptyLayer())
        # 创建路由连接层，无计算使用EmptyLayer占位
        elif layer["type"]=="route":
            # 需要路由连接层相对索引
            lys=layer["layers"].split(',')
            lys=[int(ly) for ly in lys]
            lys=[(ly,ly-i)[ly>0] for ly in lys]
            # 计算路由连接后的输出通道
            out_channels=0
            for ly in lys:
                out_channels+=outs[ly+i]
            module.add_module("route_%d"%i,EmptyLayer())
            in_channels=out_channels
        # Head层，处理结果以适应训练或预测
        elif layer["type"]=="yolo":
            mask=layer["mask"].split(',')
            mask=[int(m) for m in mask]
            anchors=layer["anchors"].split(',')
            anchors=[int(a) for a in anchors]
            anchors=[[anchors[i],anchors[i+1]] for i in range(0,len(anchors),2)]
            anchors=[anchors[m] for m in mask]
            yolo=Head(anchors)
            module.add_module("yolo_%d"%i,yolo)

        modules.append(module)
        outs+=[in_channels]
    return modules

class YOLO(nn.Module):
    r'''
    YOLO网络结构
    __init__() 构造函数
        *cfg_file 网络配置文件路径

    forward() 网络前向网络

    load_weight() 加载网络权值
        *weight_file 网络参数
    '''
    def __init__(self,cfg_file):
        super(YOLO,self).__init__()
        self.layers=parse_cfg(cfg_file)
        self.blocks=create_modules(self.layers)

    def forward(self,x):
        outputs=[]
        dets=[]
        for i,block in enumerate(self.blocks):
            ltype=self.layers[i+1]["type"]
            if ltype in ["convolutional","maxpool","upsample"]:
                x=block(x)
            elif ltype=="shortcut":
                fi=int(self.layers[i+1]["from"])
                x+=outputs[i+fi]
            elif ltype=="route":
                layers=self.layers[i+1]["layers"].split(',')
                layers=[(int(layer),int(layer)-i)[int(layer)>0] for layer in layers]
                x=outputs[layers[0]+i]
                for l in layers[1:]:
                    x=torch.cat((x,outputs[l+i]),1)
            elif ltype=="yolo":
                x=block(x)
                dets+=[x]
            outputs+=[x]
        if not self.training:
            dets=torch.cat(dets,1)
        return dets


    def load_weights(self, weight_file):
        import numpy as np
        with open(weight_file,"rb") as wf:
            header=np.fromfile(wf,dtype=np.int32,count=5)
            self.header=torch.from_numpy(header)
            self.seen=self.header[3]
            weights=np.fromfile(wf,dtype=np.float32)
        ptr=0
        for i,layer in enumerate(self.layers):
            if layer["type"]=="convolutional":
                block=self.blocks[i-1]
                try:
                    bn=int(layer["batch_normalize"])
                except:
                    bn=0
                conv=block[0]

                if bn:
                    bn=block[1]
                    num_bn=bn.bias.numel()

                    bn_bias=torch.from_numpy(weights[ptr:ptr+num_bn])
                    bn.bias.data.copy_(bn_bias)
                    ptr+=num_bn

                    bn_weights=torch.from_numpy(weights[ptr:ptr+num_bn])
                    bn.weight.data.copy_(bn_weights)
                    ptr+=num_bn
                    
                    bn_running_mean=torch.from_numpy(weights[ptr:ptr+num_bn])
                    bn.running_mean.data.copy_(bn_running_mean)
                    ptr+=num_bn

                    bn_running_var=torch.from_numpy(weights[ptr:ptr+num_bn])
                    bn.running_var.data.copy_(bn_running_var)
                    ptr+=num_bn
                else:
                    num_bias=conv.bias.numel()
                    conv_bias=torch.from_numpy(weights[ptr:ptr+num_bias])
                    ptr+=num_bias
                    conv.bias.data.copy_(conv_bias)

                num_weights=conv.weight.numel()
                conv_weights=torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr+=num_weights
                conv.weight.data.copy_(conv_weights.view_as(conv.weight.data))

