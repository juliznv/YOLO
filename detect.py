from models import *
import cv2
import torch
# 配置信息
cfg_file="cfg/yolov4.cfg"
weight_file="weights/yolov4.weights"

# 加载网络
net=YOLO(cfg_file)
net.load_weights(weight_file)
net.cuda()
net.eval()
# print(net)

print("Network and weights loaded.")
oimg=cv2.imread("demo.png")
img=cv2.resize(oimg,(608,608))
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).cuda()
# print(img)
pred=net(img)
bboxes=post_process(pred)
box=bboxes[0].detach().cpu().numpy()
print(box)
class_names=load_class_names("data/coco.names")
plot_boxes_cv2(oimg,box,"predict.png",class_names)