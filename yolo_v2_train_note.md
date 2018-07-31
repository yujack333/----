## train 1:
loss function 按照yolo v1设置，但其中的object confidence一项的标签设置的为1，而不是文章中要求的pre和gt的IOU，
训练参数：lr初始为0.0005，学习率按exponential下降（阶段试的），每5000步下降一次，decay_rate为0.5.optimizer为动量sgd。
loss的权重为：object_scale=5，noobject_scale=1,coordinate_scale=5,class_scale=1. 

mAP:

!()[/picture/mAP_1.png]

