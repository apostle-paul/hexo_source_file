---
title: 'flower book: Optimization for Training DeepModels'
date: 2018-06-29 00:01:16
mathjax: true
tags:
	- machine learning
categories:
	- reading notes
---
换电脑把之前那个博客[apostlepaul.github.io](https://apostlepaul.github.io)的源码弄丢了，没有post了，懒得一篇篇恢复了，所以重新弄一个吧。
本章内容：
1. 学习和优化的区别
2. 神经网络优化中的挑战
3. 基本算法
4. 参数初始化策略
5. 自适应学习率算法
6. 二阶近似方法
7. 优化策略和元算法
<!--more-->

### 1. How Learning Diﬀers from Pure Optimization
深度模型训练的优化算法跟传统的优化算法不一样，主要体现在前者通常不是直接进行的。
具体来讲：我们关心的可能是测试集上某个难解的度量P，但是却需要通过降低一个代价函数J(θ)来间接地优化P。
典型的，我们会去优化训练集上的代价函数，即：
$$
J(\theta) = \mathbb{E}_{(x,y)~\hat{P}_{data}}L(f(x;\theta),y), \tag{1}\label{1}
$$
但其实，我们更期望优化的是数据生成分布pdata上的代价函数：
$$
J(\theta) = \mathbb{E}_{(x,y)~p_{data}}L(f(x;\theta),y). \tag{2}\label{2}
$$

#### 1.1  Empirical Risk Minimization
机器学习算法的目标是降低公式$\eqref{2}$所表达的泛化误差期望。这个量也叫做风险。
需要注意的是，假如我们知道真实的分布pdata(x,y)，那么最小化风险将会是一个优化算法问题。如果我们不知道这个真实的分布，而只有一个训练集，那么我们面对的是一个机器学习问题。
把机器学习问题转变为优化问题的最简单的办法就是最小化训练集上的期望损失。用训练集上的经验分布代替真实的分布：
$$
\mathbb{E}_{(x,y)~\hat{P}_{data}}L(f(x;\theta),y) = 1/m \Sigma^m_{i=1}L(f(x^{(i)};\sigma),y^{(i)}), \tag{3}\label{3}
$$
已经有一些理论证明，在满足一定的条件的时候，经验风险的降低会使风险降低。
但是，使用最小化经验风险的方法有两个问题：
1. 但是经验风险最小化容易导致过拟合，高容量的模型很容易就记住训练集了，
2. 一些有用的代价函数没有有用的微分，例如0-1loss，而大多有效的现代优化算法都是基于梯度下降的。

所以在深度学习的场景中，我们很少直接用经验风险最小化的方法，反之，我们会使用一个稍有不同的方法，我们真正优化的目标会更加不同于我们希望优化的目标。

#### 1.2 Surrogate Loss Functions and Early Stopping
我们经常会使用到代理损失函数，原因是：
1. 有时候真正的目标损失函数是不可解的，例如最简单的0-1分类，如果直接用0-1损失，那么这个优化算法将是输入维度的指数级；
2. 代理损失函数能学到更多，如训练集上的0-1损失已经为0的时候，代理损失函数的继续训练会提供测试集上的效果，会增强鲁棒性。

纯优化算法经常会在一个局部极小值点停止迭代，但是机器学习优化算法不会，机器学习优化算法满足了提前停止的条件（如在测试集上的真实损失函数已经ok）就会停止，而不care当前的代理损失函数的在停止时的梯度是大是小。

#### 1.3 Batch and Minibatch Algorithms
机器学习算法跟一般优化算法的一个区别是其目标函数同城可以分解为训练样本上的求和。
在实践中，通常用一部分样本来估计整个代价函数的期望，然后对参数进行更新。原因在于：
1. 增加样本数，计算代价增大，获得的收益却低于线性的。例如样本均值标准差为δ/(√n)，其中δ是样本真实标准差。分母表明了，样本数从100增加到100000，计算代价增加了100倍，收益却只有10倍。
2. 实际的训练集中有冗余，大量的样本对梯度的贡献是相似的。

有些算法对采样误差很敏感，可能有两个原因：
1. 算法使用的信息很难在小数据集上准确估计；
2. 使用了会放大采用误差的信息。

要保证minibatch是随机抽取的，因为现实中很多样本采集在顺序上是相关的。
从online learning我们可以看到SGD会减小泛化误差的原因：模型看到的每一个样本，都是从数据生成分布pdata(x,y)中得来的，没有重复。所以我们可以从mini batch中计算出泛化误差梯度的无偏估计。
分minibatch常用的做法是，讲数据打乱，分成相应的minibatch并存储，在模型训练的时候，用数据训练多遍模型。
需要注意的是只有在第一遍的时候，得到的是无偏估计，因为数据没有被重用，以后得到的都是有偏估计。但依然要训练多遍的原因是：多遍训练给训练误差带来的降低，与增加训练误差和评估误差的gap相比，还是有收益的。

### 2. Challenges in Neural Network Optimization

#### 2.1 Ill-Conditioning
“病态”指的是Hessian矩阵的处于病态。其发端是是梯度下降法。
对于一个优化目标f(x)，我们对其进行二阶泰勒展开，有：
$$
f(x) \approx f(x^{(0)}) + (x - x^{(0)})^Tg + \frac{1}{2}(x - x^{(0)})^TH(x - x^{(0)}) \tag{4}\label{4}
$$

如果我们的学习率是$\epsilon$，那么一次优化迭代的步长是$-{\epsilon}g$，一次优化迭代后的点为$x^{(0)}-{\epsilon}g$，带入上式，得：
$$
f(x^{(0)}-{\epsilon}g) \approx f(x^{(0)}) - {\epsilon}g^Tg + \frac{1}{2} {\epsilon}^2 g^THg \tag{5}\label{5}
$$

式$\eqref{5}$中等号右边的三项分别是：函数的原始值、函数斜率导致的预期改善、函数曲率导致的校正。
在一次迭代中，$g^THg$：
* 如果为零，优化目标f(x)的下降程度跟我们的预期是一样的，
* 如果为负，f(x)的下降比我们预期得更多，
* 如果为正，f(x)的下降比我们预期得要少。

而如果式$\eqref{5}$中的后两项加起来大于零，即：
$$
\frac{1}{2} {\epsilon}^2 g^THg > {\epsilon}g^Tg
$$
那么一次迭代，目标函数不会下降反而会上升，这就是病态了。

`在很多情况中，梯度范数不会在训练过程中显著缩小，但是g^THg的增长会超过一个数量级。其结果是尽管梯度很强，学习会变得非常缓慢，因为学习率必须收缩以弥补更强的曲率。`

#### 2.2 Local Minima

1. 任何凸优化的问题都可以简化为寻找局部极小点，所有的局部极小点都是全局最小点。
2. 有些凸优化的底部是一个平坦的区域，其中任何一个点都是可以接受的。
3. 神经网络，是非凸的，会存在多个局部极小值，但是通常这些局部极小值并不是主要的问题。
	- 这多个局部极小值，可能是由模型的不可辨识性带来的，在这些局部极小值点，都有相同的代价函数值。
	- 如果局部极小值比全局最小值点大很多，那么会是个问题，但是现在学者们现在猜想，对于足够大的神经网络而言，大部分局部极小值都具有很小的代价函数，而通常我们只是要寻找一个很小的代价函数，而不一定要是最小的。

一种判断是否遇到了局部极小值的办法，是画出梯度的范数随迭代的变化，如果梯度范数没有缩小到一个微小的值，那么该问题既不是局部极小值，也不是其他临界点。
不过，在高维空间，即使范数很微小，也有可能不是局部极小值带来的。

#### 2.3 Plateaus, Saddle Points and other Flat Regions

对于高维非凸函数，鞍点比局部极小值点更常见。
`鞍点激增对于训练算法来说有哪些影响呢?对于只使用梯度信息的一阶优化算 法而言，目前情况还不清楚。鞍点附近的梯度通常会非常小。另一方面，实验中梯度 下降似乎可以在许多情况下逃离鞍点。`

#### 2.4 Cliffs and Exploding Gradients
多层神经网络通常存在像悬崖一样的斜率较大区域，这是由于几个较大的权重相乘导致的。直接梯度更新，情况会非常糟糕，会使参数弹射很远，之前的优化都白做了。
所以现在一般的优化算法都会在迭代上加bound，也就是梯度阶段，来避免这个问题。

#### 2.5 Long-Term Dependencies

#### 2.6 Inexact Gradients

#### 2.7 Poor Correspondence between Local and Global Structure
基本上所有训练神经网络的方法，都是基于局部较小更新，而局部较小更新，可能存在的问题是：
1. 由于有些梯度只能近似计算，局部下降不能确定通往有效解的合理路径。
2. 由于病态，悬崖等问题的存在，即使路径正确，局部下降的步数也可能会非常多。
3. 在一些平坦区域上，局部下降不能找到有效的下降方向。
4. 在有些情况下，可能移动得太过贪心，导致反而远离了有效的解。

因此，寻找合适的初始化点，很重要。

#### 2.8 Theoretical Limits of Optimization

### 3. Basic Algorithms

#### 3.1 SGD
按照数据生成分布，随机抽取m个小批量独立同分布的训练样本，我们能得到梯度的无偏估计。
***

算法1 ： SGD
Require: 学习率$\epsilon_k$
Require: 初始参数$\theta$
$k \leftarrow 1$
while 未满足迭代停止条件 do

- 从训练集合中抽样出m个样本
- 计算梯度估计：$ \hat{g} \leftarrow \frac{1}{m}\Delta_{\theta} \Sigma_{i}L(f(x^{(i)};\theta), y^{(i)})$
- 更新梯度： $\theta \leftarrow \theta - \epsilon_k\hat{g}$
- $k \leftarrow k + 1$

end wile

***
其中一个关键参数是学习率。实践中，SGD的学习率需要随着迭代而减小，这是因为，由于抽样，会引入噪声，在极小值点，当前minibatch算出来的梯度也不一定会是0。
batch模式的梯度下降，由于每次都用了所有的训练数据，所以可以用一个固定的学习率。
SGD收敛的一个充分条件是：
$$
\Sigma_{k=1}^\infty \epsilon_k = \infty,\\
\Sigma_{k=1}^\infty \epsilon_k^2 \lt \infty.
$$
实践中学习率常常随迭代线性减少，设为：
$$
\epsilon_k = (1 - \alpha)\epsilon_0 + \alpha\epsilon_\tau
$$
其中$\alpha = \frac{k}{\tau}$，经过多次迭代后，学习率可以设为一个较小的常数。
学习率可通过试验和误差来选取，通常最好的选择方法是监测目标函数值随时间变化的学习曲线。通常$\epsilon_\tau$ 应设为$\epsilon_0$的1%左右。
难的是如何设置$\epsilon_0$。
- $\epsilon_0$太大，会使代价函数剧烈震荡。
- $\epsilon_0$太小，会使学习变得非常缓慢。

`选取$\epsilon_0$时，最好监督刚开始的几轮迭代，选取比最优的那个高一点的学习率， 但是又不会高到会引起剧烈震荡。`
SGD一个重要的性质是每一步迭代的时间不依赖训练样本的多少。另一个重要性质是，只需要少量样本就可以实现在初始几次迭代的快速更新。本章的其他改进算法都受益于这个性质。

#### 3.2 Momentum
随机梯度下降学习过程有时候会很慢。动量方法可以加速学习，特别是对高曲率，小但一致的梯度，或是带噪声的梯度有效。
* 从形式上看，动量算法引入变量v充当速度角色。我们假设是单位质量，所以动量就是速度向量v。
* 从优化算法上看，v就是描述参数空间移动的方向和速率。
* 从计算逻辑上看，v被设为负梯度的指数衰减平均。

***

Algorithms 2： SGD with momentum
Require: learning rate $\epsilon$, momentum parameter $\alpha$
Require: Initial parameters $\theta$, initial valocity $v$
while stop cretorien not met do:

- sample minibatch with m examples
- compute gradient estimate: $g \leftarrow \frac{1}{m}\Delta_\theta \Sigma_{i}L(f(x^{(i)};\theta), y^{(i)})$
- compute velocity update $v \leftarrow \alpha v - \epsilon g$
- update parameters $\theta \leftarrow \theta + v$

end while
***

`在实践中，α的一般取值为 0.5，0.9 和 0.99。和学习率一样，α也会随着时间不断调整。一般初始值是一个较小的值，随后会慢慢变大。随着时间推移调整 α 没有收缩 ε重要。`

#### 3.3 Nesterov Momentum
Nesterov 梯度加速算法启发了这种Momentum的变形。
Nesterov momentum跟标准momentum的区别是：__Nesterov 中计算梯度实在施加当前速度之后，相当于给标准momentum加了一个校正因子。__
***

Algorithms 3： SGD with Nesterov momentum
Require: learning rate $\epsilon$, momentum parameter $\alpha$
Require: Initial parameters $\theta$, initial valocity $v$
while stop cretorien not met do:

- sample minibatch with m examples
- apply interim update: $\tilde{\theta} \leftarrow \theta + \alpha v
- compute gradient estimate: $g \leftarrow \frac{1}{m}\Delta_\theta \Sigma_{i}L(f(x^{(i)};\tilde{\theta}), y^{(i)})$
- compute velocity update $v \leftarrow \alpha v - \epsilon g$
- update parameters $\theta \leftarrow \theta + v$

end while
***

__效果__: 在凸批量梯度情况下，额外误差收敛率由O(1/k)降低到O(1/k^2)。对随机梯度没有改进效果。


### 4. Parameter Initialization Strategies
有些优化算是非迭代的，只求一个解。还有一些优化算法是迭代的，但是它们可以在可接受的时间范围内收敛到可接受的解。
但是深度学习通常没有这种比较好的性质，初始值对它的影响比较大：
* 会影响到是否收敛
* 会影响收敛的速度
* 会影响收敛到到的loss的高低
* 在差不多的Loss水平，会影响泛化能力的强弱

目前我们对于初始化策略的理解还是简单的，启发式的。

现在完全确定的唯一特性是初始参数在不同的节点间“破坏对称性”。

通常情况下，我们随机初始化节点权重，从高斯或均匀分布中随机挑选出值，可以启发式地挑选偏执常数。

#### 权重
目前有的一些经验是:
* 较大的初始权重有助于破坏对称性，避免冗余单元
* 但是太大的权重，可能在迭代的过程中梯度爆炸，使节点saturate，对于CNN，可能会导致混沌（choas，对很小的绕对很敏感）

优化观点建议权重应该足够大以成功传播信息，但是正则化希望其小一点。

一种简单的启发式方法是从分布$U(-\frac{1}{\sqrt{m}}, \frac{1}{\sqrt{m}})$中采样权重，其中m是输入节点的个数。
Bengio建议采用标准初始化：
$$
W_{i,j} \sim U(-\sqrt{\frac{6}{m+n}}, \sqrt{\frac{6}{m+n}})
$$
其中m和n分别是输入和输出的个数。
这种方法是使节点具有相同激活方差和相同梯度方差的折衷。

Saxe推荐初始化为随机正交矩阵，仔细挑选负责每一层非线性缩 放或增益 (gain) 因子 g。
Marten提出了一种稀疏初始化(sparse initialization)的方案。
以上是针对权重的初始化，其他参数的初始化通常更容易一些。

#### 偏置
设置偏置的方法必须和设置权重的方法协调。设置偏置为零通常在大多数权重初始化方案中是可行的。存在一些我们可能设置偏置为非零值的情况:
* 连在输出上的偏置，设置为输出的边缘统计通常是好的；
* 在非线性的节点上，设置非0偏置以避免saturate;
* 有时，一个单元会控制其他单元能否参与到等式中。例如LSTM中会设置偏置为1。

#### 方差或精确度参数
通常我们能安全地初始化方差或精确度参数为 1。
另一种方 法假设初始权重足够接近零，设置偏置可以忽略权重的影响，然后设定偏置以产生输出的正确边缘均值，并将方差参数设置为训练集输出的边缘方差。

### 5. Algorithms with Adaptive Learning Rates
学习率对模型的性能有着显著的影响。

#### 5.1 AdaGrad
AdaGrad在凸优化背景下效果很好。
累加历史上各参数的梯度平方，对当前梯度进行缩放。
$$
r \leftarrow r + g \bigodot g \\
\Delta \theta \leftarrow - \frac{\epsilon}{\delta + \sqrt{r}} \bigodot g \\
\theta \leftarrow \theta + \Delta \theta
$$

#### 5.2 RMSProp
修改了AdaGrad，改梯度积累为指数加权的移动平均，在非凸情况下表现更好。原因：
1. AdaGrad在凸问题时能快速收敛。
2. 但是对于非凸问题，学习轨迹可能穿过很多不同的结构，最终达到了一个凸碗的区域。
3. 如果用AdaGrad的，这时候学习率可能在达到凸碗之前就已经太小了。
4. RMSProp 使用指数衰减平均以丢弃遥远过去的历史，使其能够在找到凸碗状结构后快速收敛，它就像一个初始化于该碗状结构的 AdaGrad 算法实例。
$$
r \leftarrow (1 - \rho) + \rho g \bigodot g \\
\Delta \theta \leftarrow - \frac{\epsilon}{\delta + \sqrt{r}} \bigodot g \\
\theta \leftarrow \theta + \Delta \theta
$$

#### 5.3 Adam
Momentum + RMSProp， 一阶的指数加权衰减 + 二阶的指数加权衰减。

$$
s \leftarrow \rho_1 s + (1 - \rho_1)g \\
r \leftarrow \rho_2 r + (1 - \rho_2)g \bigodot g \\
\hat{s} \leftarrow \frac{s}{1-\rho_{1}^{t}} \\
\hat{r} \leftarrow \frac{r}{1-\rho_{2}^{t}} \\
\Delta \theta = -\epsilon\frac{\hat{s}}{\sqrt{\hat{r}}+\delta} \\
\theta \leftarrow \theta + \Delta
$$

### 6. Approximate Second-Order Methods

#### 6.1 Newton Method
牛顿法是基于泰勒级数的二次展开。
$$
J(\theta) \approx J(\theta_0) + (\theta - \theta_0)\Delta_{\theta}J(\theta_0) - \frac{1}{2}(\theta - \theta_0)^{T}H(\theta - \theta_0)
$$
对$\theta$求导，令导数为0，得到参数的更新规则：
$$
\theta^* = \theta_0 - H^{-1}\Delta_{\theta}J(\theta_0)
$$
* 对于局部二次函数（H为正定矩阵），牛顿法会直接跳到极值点。
* 是凸的但是非二次函数，可以用更新规则进行迭代。
* 高维非凸，H的特征值并不都是正的（鞍点），可以给H加正则（加对角矩阵），使调整后的H特征值都是正的。

除了鞍点会限制牛顿法的使用，计算H的逆也是一个很大的负担。k*k的矩阵的逆，计算复杂度为$O(k^3)$。

#### 6.2 Conjugate Gradients

#### 6.3 BFGS



### 7. Optimization Strategies and Meta-Algorithms

#### 7.1 Batch Normalization
并不是一个优化算法，而是一个自适应的重参数化的方法，试图解决训练非常深的模型的困难。
主要就是对每一层每个节点的mini-batch的输出，进行normalization操作。这样保证下一层接受到的输入，都是一个标准正态分布。
这意味着，梯度不会再简单地增加 hi 的标准差或均值;标准化操作会 除掉这一操作的影响，归零其在梯度中的元素。
在测试阶段，μ 和 σ 可以被替换为训练阶段收集的运行均值。这使得模型可以 对单一样本评估，而无需使用定义于整个小批量的 μ 和 σ。

标准化一个单元的均值和标准差会降低包含该单元的神经网络的表达能力。为了保持网络的表现力，通常会将批量隐藏单元激活$H$替换为 γH ′ + β，而不是简单地使用标准化的H'。
(后续补充一些细节)

#### 7.2 Coordinate Desent
在某些情况下，将一个优化问题分解成几个部分，可以更快地解决原问题。
简单来说就是固定某些变量，迭代另外一些变量，反复迭代进行。
当优化问题中的不同变量能够清楚地分成相对独立的组，或是当优化一组变量 明显比优化所有变量效率更高时，坐标下降最有意义 。
当一个变量的值很大程度地影响另一个变量的最优值时，坐标下降不是一个很好的方法。

#### 7.3 Polyak avarage
平均优化学习轨迹上的几个点，用于非凸问题上时，就是exponential decay。

#### 7.4 supervise pre-Training
有时，如果模型太复杂难以优化，或是如果任务非常困难，直接训练模型来解 决特定任务的挑战可能太大。有时训练一个较简单的模型来求解问题，然后使模型 更复杂会更有效。训练模型来求解一个简化的问题，然后转移到最后的问题，有时 也会更有效些。

* 贪心监督预训练
* 迁移学习

#### 7.5 Designing Models to Aid Optimization

#### 7.6 Continuation Methods and Curriculum Learning
