## yolo v2

yolo v2 基于 v1做了以下的尝试性的实验：

![](/picture/yolo_v2_1.png)

-  batch_norm,convolutional,new network 可以总结为一点，改变了网络结构，backbone用的是自己设计的darknet19.
-  hi-res classifier：将backbone网络在IMAGENET数据集上训练分类网络时，不再是基于224的分辨率，而是用448.因为在retrain检测网络时，用到的分辨率是这个量级的，这样可以让网络快速的适应这个分辨率。

- anchor box在最后的网络中没有用到，所以就不解释了。
- dimension priors 是指 w h的设置用聚类的方法来得到一个最优的，而不是人为的设定。这里的 w h在后面location prediction要用到。
- location prediction 这里是修正v1的坐标预测的方法。v1是直接预测 W H，v2是有一个预设的W H，是预测一个缩放的程度。

![](/picture/yolo_v2_2.png)

- passthrough 是将低一层的feature map 结合到高层 feature map中，以便提供更多的高分辨率的信息用来做小物体的检测。
示意图如下：

![](/picture/yolo_v2_4.jpg)

代码实现如下：

![](/picture/yolo_v2_3.png)

- multi-scale 多分辨率的训练
- hi-res detector 用高分辨率的输入图片。


## 文章中没写的
文章中有两个部分没有些：

1.true bbox怎么分配给anchor

2.loss function

对于第一个问题：
 分配方法为：true bbox落在哪个cell，则这个cell就负责detect这个true box。进一步，5个anchor与 true box算IOU，IOU最大的anchor负责预测这个true bbox。也就是说一个 true bbox分配到一个anchor。
 
```python
#box_coor_trans为预测box经过转换后的box。shape为[batch,cell_size,cell_size,box_per_cell,4]
# boxes 为label, shape为[batch,cell_size,cell_size,box_per_cell,4]
 
 # iou为预测的框和真实框的IOU，shape为[batch,cell_size,cell_size,box_per_cell,1]
iou = self.calc_iou(box_coor_trans, boxes)
# 找到最优的预测box
best_box = tf.to_float(tf.equal(iou, tf.reduce_max(iou, axis=-1, keep_dims=True)))
# 用response去除没有物体的cell，response的shape为[batch,cell_size,cell_size,box_per_cell,1]表示cell中是否有物体。
obj_mask = tf.expand_dims(best_box * response, axis = 4)
# 最后obj_mask的输出为[batch,cell_size,cell_size,box_per_cell,1]
```
对于第二个问题：

这里先给出loss functuin:

![](/picture/yolo_v2_6.jpg)

- 第一项为背景的loss，背景的判定为：预测框与所有的true bbox的IOU都小于一个阈值。这里的阈值为0.6。预测框不用与所有的true bbox来比较，只需要和落在cell中的true bbox就可以了。

```python
#box_coor_trans为预测box经过转换后的box。shape为[batch,cell_size,cell_size,box_per_cell,4]
# boxes 为label, shape为[batch,cell_size,cell_size,box_per_cell,4]
 
 # iou为预测的框和真实框的IOU，shape为[batch,cell_size,cell_size,box_per_cell,1]
iou = self.calc_iou(box_coor_trans, boxes)
# 找到IOU小于阈值0.6的预测框做为背景框
best_box = tf.to_float(iou<0.6)
no_obj_mask = tf.expand_dims(best_box, axis = 4)
# 最后no_obj_mask的输出为[batch,cell_size,cell_size,box_per_cell,1]
```

- 第二项 loss是为了稳定前期的训练，由于网络是随机初始化的（即便backbone是训练过的，但后面用于detection的参数还是随机初始化的），所以网络的bbox prediction 可能偏差很大。作者先让 bbox prediction 先与 anchor 来靠近，这样可以稳定训练。等训练12800步之后再去除这个loss。

- 后面三项就为分配了true bbox的 anchor的bbox loss，是否有物体的loss和分类loss。

- 第一项和后面三项都有mask，在全局求完loss之后在卡个mask就可以了，第二项是全局的loss 也就是说[batch,cell_szie,cell_size,box_per_cell]都要算loss。








