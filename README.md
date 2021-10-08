# Kaggle木薯叶病分类比赛

## 目的

利用比赛的项目来学习CV，提高代码水平

## 收获

### 数学知识

#### Beta分布

$$
P(θ|α, β) = \frac{Γ(α + β)}{Γ(\alpha)+Γ(\beta)}\theta^{\alpha-1}(1-\theta)^{\beta-1}
$$

其中$\tau(x)$为出名的 gamma 函数，它是阶乘运算的推广
$$
Γ(x) = ∫_0^\infty t^{x-1}e^{-t}dt
$$

![Beta_PDF](https://github.com/LiuZH-enjoy/Kaggle-Cassava-leaf-disease-classification-contest/blob/master/imgs/Beta_PDF.png)

![Beta_CDF](https://github.com/LiuZH-enjoy/Kaggle-Cassava-leaf-disease-classification-contest/blob/master/imgs/Beta_CDF.png)

**重点：** **Beta 分布的随机变量的定义域在 [0,1] 之间。**  

#### 图像的离散傅立叶变换  

时域转频域：  


$$
F(u,v)=\sum_{x-0}^{M-1}\sum_{y=0}^{N-1}f(x,y)e^{-j2\pi(\frac{ux}{M}+\frac{vy}{N})}
$$

$$
e^{-j2\pi(\frac{ux}{M}+\frac{vy}{N})}=cos(2\pi\frac{ux}{M}+jsin(2\pi\frac{vy}{N}))
$$

频域转时域：
$$
f(x,y)=\frac{1}{MN}\sum_{u=1}^{M-1}\sum_{v=0}^{N-1}F(u,v)e^{j2\pi(\frac{ux}{M}+\frac{vy}{N})}
$$
图像等价于⼆维波，图像的傅⽴叶变换，就是将图像变为⼀系列⼆维波的叠加的和。  





### 显存与梯度

训练时，保存在显存中的变量：

- 模型参数
- 优化器的变量（⼀些中间变量，如 momentums）
- 中间临时变量（前向传播和后向传播的中间变量，各个样本计算时得到的隐藏层的值）  
- ⼯作空间（内核实现局部变量的临时内存）  

在一般情况下，batchsize越大，模型学习越快，越容易收敛。但是当 batchsize 变⼤后，样本越多，中间临时变量就会越来越多，很可能导致显存不足，这时候该如何解决？

1. 多GPU
![mutil_GPU](https://github.com/LiuZH-enjoy/Kaggle-Cassava-leaf-disease-classification-contest/blob/master/imgs/mutil_GPU.png)
   

2. 单个GPU或者CPU
![single_GPUorCPU](https://github.com/LiuZH-enjoy/Kaggle-Cassava-leaf-disease-classification-contest/blob/master/imgs/single_GPUorCPU.png)


### 数据增强：MixUp，FMix，CutMix

$$
mix(x_1, x_2, m) = m ⊙ x_1 + (1 - m) ⊙ x_2
$$
$$
mix(y_1, y_2) = rate ⋅ y_1 + (1 - rate) ⋅ y_2
$$

![Mixup&Cutout&CutMix](https://github.com/LiuZH-enjoy/Kaggle-Cassava-leaf-disease-classification-contest/blob/master/imgs/Mixup%26Cutout%26CutMix.png)


- **MixUp**

  公式：
  $$
  \hat{x}=\lambda x_i+(1-\lambda)x_j, \ \ where\ x_i,\ x_j\ are\ raw\ input\ vectors
  $$
  $$
  \hat{y}=\lambda y_i+(1-\lambda)y_j, \ where\ y_i,\ y_j\ are\ one-hot\ label\ encodings
  $$
  其中$x_i,y_i$和$x_j,y_j$都是从训练集中随机选择的，其中lambda取值于beta分布，范围为0-1。

  y是one-hot标签，⽐如yi的标签为[0,0,1]，yj的标签为[1,0,0]，此时lambda为0.2，那么此时的标签就变为0.2*[0,0,1] + 0.8*[1,0,0] = [0.8,0,0.2]，其实Mixup的⽴意很简单，就是通过这种混合的模型来增强模型的泛化性。

  MixUp采用配对的方式进行训练, 通过混合两个甚至是多个样本的分布, 同时加上对应的标签来训练。

  > In essence, mixup trains a neural network on convex combinations of pairs of examples and their labels.

  将随机的两张样本按比例混合，分类的结果按比例分配


- **CutMix**

  CutMix就是把一个物体抠出来, 粘贴到另一张图上去。将一部分区域cut掉但不填充0像素而是随机填充训练集中的其他数据的区域像素值，分类结果按一定的比例分配

- **FMix**

  理论上来说, 因为CutMix引入了一个蒙版的效果, 每个神经元在编码的时候会产生派生的意义, 这就好比人眼去看一个似是而非的东西, 这就会迫使模型去学习局部一致性, 比如说, 前面马路有一滩水, 水里面有一个人, 那么这个分类是分类为水还是人呢? 通过这么一个数据的增强, 模型学会了在大环境下判断这个物体, 它是一滩水, 而不是一个people.

  CutMix是一个PS技术很差的人把猫狗合在一起, 直接就是矩形框粘贴, 而Fmix是一个技术高超的PSer. 它可以通过融合通道颜色来完美的把图片剪切过去.
  
  ![FMix](https://github.com/LiuZH-enjoy/Kaggle-Cassava-leaf-disease-classification-contest/blob/master/imgs/FMix.png)
  

### sklearn--KFold StratifiedKFold

- KFold划分数据集的原理：**根据n_split直接进行划分**
- StratifiedKFold划分数据集的原理：**划分后的训练集和验证集中类别分布尽量和原数据集一样**

导入相关的包

```python
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
```

模拟数据

```python
X = np.array([[10, 1], [20, 2], [30, 3], [40, 4], [50,5], [60,6], [70,7],[80,8],[90,9],[100,10],
             [110, 1], [120, 2], [130, 3], [140, 4], [150,5], [160,6], [170,7],[180,8],[190,9],[200,10]])
# 五个类别：1:1:1:1:1
Y1 = np.array([1,1,2,3,3,2,4,4,5,5,1,1,2,3,3,2,4,4,5,5])
# 两个类别：2:3
Y2 = np.array([1,1,1,1,2,2,2,2,2,2,1,1,1,1,2,2,2,2,2,2])
```

KFold:

```python
kfolds = KFold(n_splits=5, shuffle=False)

# 注：返回的是索引
for (trn_idx, val_idx) in kfolds.split(X, Y1):
    print((trn_idx, val_idx))
    print((len(trn_idx), len(val_idx)))  

Out: 
(array([ 4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]), array([0, 1, 2, 3]))
(16, 4)
(array([ 0,  1,  2,  3,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]), array([4, 5, 6, 7]))
(16, 4)
(array([ 0,  1,  2,  3,  4,  5,  6,  7, 12, 13, 14, 15, 16, 17, 18, 19]), array([ 8,  9, 10, 11]))
(16, 4)
(array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 16, 17, 18, 19]), array([12, 13, 14, 15]))
(16, 4)
(array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]), array([16, 17, 18, 19]))
(16, 4)
```

StratifiedKFold：

```python
# StratifiedKFold: 抽样后的训练集和验证集的样本分类比例和原有的数据集尽量是一样的

#　对(X, Y1)进行抽样
# Y1中有5个类别，比例为1:1:1:1:1
# 所以，每个KFold的样本数必须为 1*x+1*x+1*x+1*x+1*x=5x个样本
stratifiedKFolds = StratifiedKFold(n_splits=2, shuffle=False)
for (trn_idx, val_idx) in stratifiedKFolds.split(X, Y2):
    print((trn_idx, val_idx))
    print((len(trn_idx), len(val_idx)))

print('################################################')
#　对(X, Y2)进行抽样
# Y2中有2个类别，比例为2:3
# 所以，每个KFold的样本数必须为 2x+3x=5x个样本
stratifiedKFolds = StratifiedKFold(n_splits=4, shuffle=False)
for (trn_idx, val_idx) in stratifiedKFolds.split(X, Y2):
    print((trn_idx, val_idx))
    print((len(trn_idx), len(val_idx)))

Out:
(array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19]), array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
(10, 10)
(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19]))
(10, 10)
################################################
(array([ 2,  3,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]), array([0, 1, 4, 5, 6]))
(15, 5)
(array([ 0,  1,  4,  5,  6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]), array([2, 3, 7, 8, 9]))
(15, 5)
(array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 12, 13, 17, 18, 19]), array([10, 11, 14, 15, 16]))
(15, 5)
(array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 14, 15, 16]), array([12, 13, 17, 18, 19]))
(15, 5)
```

