总结的相关算法如下：
- 1.逻辑回归
- 2.lasso方法
- 3.树方法
- 4.softmax+cross_entropy 
- 5 SVM和逻辑回归有什么不同

# logistic regression model
## 模型

### 预测函数为：

  ![](https://latex.codecogs.com/gif.latex?h_%5Ctheta%28x%29%3D%5Cfrac%7B1%7D%7B1&plus;e%5E%7B-%5Ctheta%5E%7BT%7D%7Bx%7D%7D%7D)
### loss function：

  ![](https://latex.codecogs.com/gif.latex?cost%28h_%7B%5Ctheta%7D%28x_%7Bi%7D%29%2Cy_%7Bi%7D%29%3Dy_%7Bi%7Dlog%28h_%7B%5Ctheta%7D%28x_%7Bi%7D%29%29&plus;%281-y_%7Bi%7D%29log%281-h_%7B%5Ctheta%7D%28x_%7Bi%7D%29%29)
### 总体的loss function：
  ![](/picture/logistic.png)
  
  
  ![](https://latex.codecogs.com/gif.latex?J%28%5Ctheta%29%3D-%7B1%5Cover%7Bn%7D%7D%5Csum_%7Bi%3D1%7D%5En%5Bcost%28h_%7B%5Ctheta%7D%28x_%7Bi%7D%29%2Cy_%7Bi%7D%29%3Dy_%7Bi%7Dlog%28h_%7B%5Ctheta%7D%28x_%7Bi%7D%29%29&plus;%281-y_%7Bi%7D%29log%281-h_%7B%5Ctheta%7D%28x_%7Bi%7D%29%29%5D)

## 优化

### 参数的导数：

  ![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%5Ctheta%5E%7Bj%7D%7DJ%28%5Ctheta%29%3D%5Cfrac%7B1%7D%7Bm%7D%5Csum%5Climits_%7Bk%3D1%7D%5E%7Bm%7D%28h_%7B%5Ctheta%7D%28x_k%29-y_k%29x_k%5Ej)
### 更新公式

  ![](https://latex.codecogs.com/gif.latex?%5Ctheta_%7Bj%7D%20%3A%3D%5Ctheta_%7Bj%7D-%20%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%5Ctheta%5E%7Bj%7D%7DJ%28%5Ctheta%29%3D%5Ctheta_%7Bj%7D-%5Cfrac%7B1%7D%7Bm%7D%5Csum%5Climits_%7Bk%3D1%7D%5E%7Bm%7D%28h_%7B%5Ctheta%7D%28x_k%29-y_k%29x_k%5Ej)
  
# lasso方法
  lasso方法是一种特征选择的方法，他需要结合任务或者说是其它模型来一起求解。在这里举一个线性回归的例子来说明。
  
  - 原始的回归模型如下：
 
  ![](https://pic3.zhimg.com/80/v2-ab16745b61e0c48623a20647a923a266_hd.jpg)
  
  - loss function为：
  
  ![](https://pic4.zhimg.com/80/v2-d8b279a110d966e717f3f56d66b23d24_hd.jpg)
  
  - 加L1约束项之后：
  
  ![](https://pic3.zhimg.com/80/v2-c7e70d404369662a1b46669a8713902e_hd.jpg)
  
  由上述的公式求解出的系数将变得稀疏，即有的系数为0。这样就起到了特征选择的作用。
  
# 树方法
这里主要介绍的树方法有：一般的决策树、bagging、随机森林、GBDT。分两部分来介绍，第一部分为基础的决策树、树的bagging、随机森林。第二部分为GBDT。

## 第一部分
### 普通的决策树
决策树的优点：决策树可以认为是if-then规则的集合，易于理解，可解释性强，预测速度快。同时，决策树算法相比于其他的算法需要更少的特征工程，比如可以不用做特征标准化，- [可以很好的处理字段缺失的数据](https://blog.csdn.net/u012328159/article/details/79413610)，也可以不用关心特征间是否相互依赖等。决策树能够自动组合多个特征，它可以毫无压力地处理特征间的交互关系并且是非参数化的，因此你不必担心异常值或者数据是否线性可分。
- 1. 从深度为0的树开始，对每个叶节点枚举所有的可用特征
- 2. 针对每个特征，把属于该节点的训练样本根据该特征值升序排列，通过线性扫描的方式来决定该特征的最佳分裂点，并记录该特征的最大收益（采用最佳分裂点时的收益）
- 3. 选择收益最大的特征作为分裂特征，用该特征的最佳分裂点作为分裂位置，把该节点生长出左右两个新的叶节点，并为每个新节点关联对应的样本集
- 4. 回到第1步，递归执行到满足特定条件为止

一般的决策树易于过拟合，有一些方式可以防止过拟合。bagging方法就是对训练样本抽样，基于这些抽样出的样本训练出多个决策树。随机深林方法是对特征抽样，抽样出不同的特征集合来训练多个决策树。给出下图来方便理解决策树的构建：
![](https://muxuezi.github.io/posts/mlslpic/5.1%20decisiontree.png)

## 第二部分
### GBDT（gradient boosting decision tree）
前面提到了的bagging和随机森林方法都是ensemble learning的范畴，而这部分讲的GBDT也是属于集成学习的范畴，集成学习简单来说就是多个模型一同来做决定（三个臭皮匠顶个诸葛亮）。这里讲的GB方法是属于boosting方法中的一种。boosting方法中比较著名的有adaboost。adaboost方法是训练一系列弱分类器来组成一个强分类器，主要思想是前一个分类器分错的样本的权重将会加大，而分对的样本的权重将会减小。这样，下一个分类器将会格外关注之前分错的样本。最后将这些分类器加权的线性组合起来。权重即为误差，即准确度越高的分类器的权重越大。

- Gradient Boosting 是一种 Boosting 的思想，它本质是，每一次建立模型是在之前建立模型损失函数的梯度下降方向。其步骤如下:
![](https://pic4.zhimg.com/80/v2-c75f66da84db9f86f4191903d1d156d9_hd.jpg)

# softmax+cross_entropy
-[推倒过程](https://blog.csdn.net/u014380165/article/details/77284921)

# SVM和逻辑回归有什么不同
- 1.损失函数不一样，逻辑回归的损失函数是log loss，svm的损失函数是hinge loss
- 2.损失函数的优化方法不一样，逻辑回归用剃度下降法优化，svm用smo方法进行优化
- 3.逻辑回归侧重于所有点，svm侧重于超平面边缘的点
- 4.svm的基本思想是在样本点中找到一个最好的超平面

# hinge loss和softmax loss的区别
- 区别在于hinge loss想要错误类别的分数小于真确类别的分数一定的范围，而softmax loss希望所有的概率密度都集中在真确的类别上。
