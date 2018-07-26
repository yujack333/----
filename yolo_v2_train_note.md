## train 1:
loss function 按照yolo v1设置，但其中的object confidence一项的标签设置的为1，而不是文章中要求的pre和gt的IOU，
训练参数：lr初始为0.0005，学习率按exponential下降（阶段试的），每5000步下降一次，decay_rate为0.5.optimizer为动量sgd。
loss的权重为：object_scale=5，noobject_scale=1,coordinate_scale=5,class_scale=1. 

mAP:
```python
{'class_0': array([0.27272727]), 'class_1': array([0.09090909]), 'class_2': array([0.09090909]), 'class_3': array([0.09090909]), 'class_4': array([0.18181818]), 'class_5': array([0.09090909]), 'class_6': array([0.]), 'class_7': array([0.27272727]), 'class_8': array([0.09090909]), 'class_9': array([0.09090909]), 'class_10': array([0.18181818]), 'class_11': array([0.]), 'class_12': array([0.18181818]), 'class_13': array([0.27272727]), 'class_14': array([0.09090909]), 'class_15': array([0.09090909]), 'class_16': array([0.18181818]), 'class_17': array([0.09090909]), 'class_18': array([0.09090909]), 'class_19': array([0.09090909]), 'mAP': array([0.12727273])}
```
