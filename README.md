# Perceptron
Perceptron without numpy or any other third-party packet 
This is a homework for me
不适用任何第三方包的感知机算法实现

感知机（梯度下降法求解）
感知器（英语：Perceptron）是Frank Rosenblatt在1957年就职于康奈尔航空实验室（Cornell Aeronautical Laboratory）时所发明的一种人工神经网络。它可以被视为一种最简单形式的前馈神经网络，是一种二元线性分类器。

第一章
1.1 概念
感知机主要是应用于二类问题；对于输入的向量x，w为输入向量x连接到感知机的权重，则有如下的公式对其计算：
f(x)={y=+1 if w*x+b≥0   ,y-1 else)

1.2 搭建感知机模型
为了搭建感知机模型，我们需要知道高维数据的线性可分是指什么。为此我们需要定义 “超平面” 的概念：
τ：wx+b=0
其中 w、x 都是 n 维向量，τ 则是 Rn 中的超平面。
此即直线方程。有了 Rn 中超平面的定义后，线性可分的概念也就清晰了：对于一个数据集:
D={(x_1,y_2 ),…,(x_n,y_n)}

（xi为输入，yi为标签），如果存在一个超平面τ，能够将D中正负样本（对于某个样本（xi，yi），若 yi =1 则称其为正样本，若 yi =-1 则称其为负样本，且标签 yi 只能取正负 1 这两个值）分开，那么就称 D 是线性可分的。否则，就称是线性不可分的。对于感知机模型来说，以上的这些信息就足够了。事实上，感知机模型只有 w 和 b 这两个参数，我们要做的就是根据样本的信息来逐步更新 w 和 b、从而使得对应的超平面 τ 能够分开 D。

![image](https://github.com/BitArtificial/Perceptron/blob/master/%E7%A4%BA%E4%BE%8B%E5%9B%BE.svg)
如图，对于n组向量x_1,x_2,x_3…x_n及其对应的两类标签y_1,y_2,y_3…y_n  (y_i=±1)，。若该数据为线性可分，则可以找到一个超平面将其分开。

第二章 感知机算法
对于感知机的参数，将使用以下算法来进行计算
 2.1 感知机损失函数的定义
若(x,y)是正样本则 w*x+b>0，若(x,y)为负样本，则 w*x+b<0 ，而前文提到 (x,y)为正样本，则 y=1，(x,y) 为负样本，则 y=-1 ，因此我们定义损失函数为：Loss=-y_i*(w_i*x_i+b)  当误分类时，Loss>0 ，对错分类的点，损失函数求导有：∂Loss/∂w(x_i,y_i)=-y_i*x_i ，∂Loss/∂b (x_i,y_i )=-y_i
而在更新梯度时，我们只需要找出使得 Loss 最大的那一组样本 (x_i,y_i) 用它来更新梯度即可。


第三章实例：剖腹产的选择
3.1.1 剖腹产数据集的介绍
Caesarian Section Classification Dataset Data Set contains information about caesarian section results of 80 pregnant women with the most important characteristics of delivery problems in the medical field.
We choose age, delivery number, delivery time, blood pressure and heart status. 
We classify delivery time to Premature, Timely and Latecomer. As like the delivery time we consider blood pressure in three statuses of Low, Normal and High moods. Heart Problem is classified as apt and inept. 

@attribute 'Age' { 22,26,28,27,32,36,33,23,20,29,25,37,24,18,30,40,31,19,21,35,17,38 } 
@attribute 'Delivery number' { 1,2,3,4 } 
@attribute 'Delivery time' { 0,1,2 } -> {0 = timely , 1 = premature , 2 = latecomer} 
@attribute 'Blood of Pressure' { 2,1,0 } -> {0 = low , 1 = normal , 2 = high } 
@attribute 'Heart Problem' { 1,0 } -> {0 = apt, 1 = inept } 

@attribute Caesarian { 0,1 } -> {0 = No, 1 = Yes } 
3.1.2 数据集预处理
首先我们需要将源数据集中的数据部分提取出来，共80个数据如图1-2所示，并在实际运用的时候最后一栏：是否选择剖腹产部分原来的0和1变成-1和1，也就是我们的y分类标签值。
