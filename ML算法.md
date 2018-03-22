总结的相关算法如下：
- 1.逻辑回归
- 2.lasso方法



# logistic regression model
## 模型

### 预测函数为：

  ![](https://latex.codecogs.com/gif.latex?h_%5Ctheta%28x%29%3D%5Cfrac%7B1%7D%7B1&plus;e%5E%7B-%5Ctheta%5E%7BT%7D%7Bx%7D%7D%7D)
### loss function：

  ![](https://latex.codecogs.com/gif.latex?cost%28h_%7B%5Ctheta%7D%28x_%7Bi%7D%29%2Cy_%7Bi%7D%29%3Dy_%7Bi%7Dlog%28h_%7B%5Ctheta%7D%28x_%7Bi%7D%29%29&plus;%281-y_%7Bi%7D%29log%281-h_%7B%5Ctheta%7D%28x_%7Bi%7D%29%29)
### 总体的loss function：

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
  
 
