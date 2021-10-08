# Kaggle木薯叶病分类比赛

## 目的

利用比赛来学习CV，提高代码水平

## 收获

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
  

