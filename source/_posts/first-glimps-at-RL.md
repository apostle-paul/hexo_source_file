---
title: first glimps at RL
date: 2018-09-08 23:23:28
mathjax: true
tags:
  - reinforce learning
categories:
  - study notes
---
以下内容摘录自
* [知乎专栏-强化学习知识大讲堂](https://zhuanlan.zhihu.com/sharerl)
* [知乎专栏-当我们在谈论数据挖掘](https://zhuanlan.zhihu.com/data-miner)
* [知乎专栏-智能单元](https://zhuanlan.zhihu.com/intelligentunit)
<!--more-->

### 基础概念
* 在监督学习和非监督学习中，数据是静态的不需要与环境进行交互，比如图像识别，只要给足够的差异样本，将数据输入到深度网络中进行训练即可
* 强化学习的学习过程是个动态的，不断交互的过程，所需要的数据也是通过与环境不断地交互产生的。
* 与监督学习和非监督学习相比，强化学习涉及到的对象更多，比如动作，环境，状态转移概率和回报函数等

***
#### 马尔科夫
第一个概念是马尔科夫性：所谓马尔科夫性是指系统的下一个状态$s_{t+1}$仅与当前状态$s_t$有关，而与以前的状态无关。

第二个概念是马尔科夫过程
马尔科夫过程的定义：马尔科夫过程是一个二元组$\left(S,P\right)$，且满足：S是有限状态集合，P是状态转移概率。

第三个概念是马尔科夫决策过程
马尔科夫决策过程由元组$(S,A,P,R,\gamma)$描述，其中：S为有限的状态集, A 为有限的动作集, P 为状态转移概率, R为回报函数, $\gamma$为折扣因子，用来计算累积回报。注意，跟马尔科夫过程不同的是，马尔科夫决策过程的状态转移概率是包含动作的即：
$P_{ss'}^{a}=P\left[S_{t+1}=s'|S_t=s,A_t=a\right]$

***

**强化学习的目标是给定一个马尔科夫决策过程，寻找最优策略。**
所谓策略是指状态到动作的映射，策略常用符号$\pi$表示，它是指给定状态s 时，动作集上的一个分布，即
$\pi\left(a|s\right)=p\left[A_t=a|S_t=s\right] (1.1)$
*公式(1.1)的含义是：策略$\pi$在每个状态s 指定一个动作概率。如果给出的策略$\pi$是确定性的，那么策略$\pi$在每个状态s指定一个确定的动作。*
这里的最优是指得到的总回报最大。

当给定一个策略$\pi$时，我们就可以计算累积回报了。首先定义累积回报：

$G_t=R_{t+1}+\gamma R_{t+2}+\cdots =\sum_{k=0}^{\infty}{\gamma^kR_{t+k+1}}\\\\\ \left(1.2\right)$
不过，给定策略$\pi$，从状态$s_1$出发，可能有多个序列状态，每个序列状态都会有一个累积回报。而这些回报的期望，就是状态-价值函数了：
$\upsilon_{\pi}\left(s\right)=E_{\pi}\left[\sum_{k=0}^{\infty}{\gamma^kR_{t+k+1}|S_t=s}\right] (1.3)$
相应地，状态-行为值函数为：
$$
q_{\pi}(s,a)=E_{\pi}[\sum_{k=0}^{\infty}{\gamma^kR_{t+k+1}|S_t=s,A_t=a}](1.4)
$$
***
#### bellman函数
实际编程的时候不会根据定义去实现，而是会去根据bellman方程
由状态值函数的定义式(1.3)可以得到：
$\upsilon\left(s\right)=E\left[G_t|S_t=s\right]$
$=E\left[R_{t+1}+\gamma R_{t+2}+\cdots |S_t=s\right]$ $=E\left[R_{t+1}\gamma\left(R_{t+2}+\gamma R_{t+3}+\cdots\right)|S_t=s\right]$
$=E\left[R_{t+1}+\gamma G_{t+1}|S_t=s\right]$
$=E\left[R_{t+1}+\gamma\upsilon\left(S_{t+1}\right)|S_t=s\right]  (1.5)$

同样的，可以得到状态-动作价值函数的bellman函数
同样我们可以得到状态-动作值函数的贝尔曼方程：
$q_{\pi}\left(s,a\right)=E_{\pi}\left[R_{t+1}+\gamma q\left(S_{t+1},A_{t+1}\right)|S_t=s,A_t=a\right]\left(1.6\right)$

经过推导得到行为状态-行为值函数：

$q_{\pi}\left(s,a\right)=R_{s}^{a}+\gamma\sum_{s'\in S}{P_{ss'}^{a}\sum_{a'\in A}{\pi\left(a'|s'\right)q_{\pi}\left(s',a'\right)}}\left(1.11\right)$

***

计算状态值函数的目的是为了构建学习算法从数据中得到最优策略。每个策略对应着一个状态值函数，最优策略自然对应着最优状态值函数。
定义：最优状态值函数$\epsilon^{\*}(s)$,为在所有策略中值最大的值函数即：$\epsilon^{\*}\left(s\right)=\max_{\pi}\epsilon_{\pi}\left(s\right)$，最优状态-行为值函数$q^\*\left(s,a\right)$为在所有策略中最大的状态-行为值函数，即：

$q^\*\left(s,a\right)=\max_{\pi}q_{\pi}\left(s,a\right)$

若已知最优状态-动作值函数，最优策略可通过直接最大化$q^\*\left(s,a\right)$ 来决定。

***
#### 形式化描述

我们定义一个离散时间有限范围的折扣马尔科夫决策过程$M=\left(S,A,P,r,\rho_0,\gamma,T\right)$，其中S为状态集，A为动作集，$P:S\times A\times S\rightarrow R$是转移概率，$r:S\times A\rightarrow\left[-R_{\max},R_{\max}\right]$为立即回报函数，$\rho_0:S\rightarrow R$是初始状态分布，$\gamma\in\left[0,1\right]$为折扣因子，T为水平范围（其实就是步数）. $\tau$为一个轨迹序列，即$\tau =\left(s_0,a_0,s_1,a_1,\cdots\right)$，累积回报为$R=\sum_{t=0}^T{\gamma^t}r_t$，强化学习的目标是：找到最优策略$\pi$，使得该策略下的累积回报期望最大，即：$\max_{\pi}\int{R\left(\tau\right)}p_{\pi}\left(\tau\right)d\tau$。

根据策略最优定理知道，当值函数最优时采取的策略也是最优的。反过来，策略最优时值函数也最优。

### 基于模型的动态规划方法
根据转移概率P是否已知，可以分为基于模型的动态规划方法和基于无模型的强化学习方法。
![912d3bcb545a17935ac071fb5d91a805.jpeg](evernotecid://74F421DB-DF5E-4EEE-B947-C0C980D2DC0B/appyinxiangcom/14855437/ENNote/p75?hash=912d3bcb545a17935ac071fb5d91a805)
基于模型的强化学习可以利用动态规划的思想来解决。
动态规划问题，需要满足两个条件：
1. 整个优化问题可以分解为多个子优化问题，
2. 子优化问题的解可以被存储和重复利用

强化学习可以利用马尔科夫决策过程来描述，利用贝尔曼最优性原理得到贝尔曼最优化方程，从bellman方程可知，MDP符合使用动态规划的两个条件。
动态规划的核心是找到最优值函数。

***
第一个问题是：给定一个策略$\pi$，如何计算在策略$\pi$下的值函数？
值函数的计算公式是：
${\epsilon_\pi }\left( s \right) = \sum\limits_{a \in A} {\pi \left( {a|s} \right)} \left( {R_s^a + \gamma \sum\limits_{s' \in S} {P_{ss'}^a{\epsilon_\pi }\left( {s'} \right)} } \right){\kern 1pt} {\kern 1pt} {\kern 1pt} {\kern 1pt}  (2.4)$

因为$\upsilon_\pi(s')$也是未知的，所以需要用bootstrapping算法来计算。

#### bootstrapping算法
![c8544b58c8bd224f52c05332e9ad202e.jpeg](evernotecid://74F421DB-DF5E-4EEE-B947-C0C980D2DC0B/appyinxiangcom/14855437/ENNote/p75?hash=c8544b58c8bd224f52c05332e9ad202e)
***

第二个要解决的问题是：如何利用值函数进行策略改善，从而得到最优策略？
一个很自然的方法是当已知当前策略的值函数时，在每个状态采用贪婪策略对当前策略进行改进，即：
$$
\pi_{l + 1}(s) \in \mathop {\arg \max} \limits_a q^{\pi_l}(s,a)
$$

***
至此，我们已经给出了策略评估算法和策略改进算法。万事已具备，将策略评估算法和策略改进算合起来便组成了策略迭代算法。

![b53fa72d9283f865c0af591d1839c0e9.jpeg](evernotecid://74F421DB-DF5E-4EEE-B947-C0C980D2DC0B/appyinxiangcom/14855437/ENResource/p177)
![597f2494f36c7613ccc6f0e8f4f0545d.jpeg](evernotecid://74F421DB-DF5E-4EEE-B947-C0C980D2DC0B/appyinxiangcom/14855437/ENResource/p179)

**这就是策略迭代算法。**
***
在进行策略评估时，值函数的收敛需要很多轮迭代。那么，我们需要等到值函数完全收敛么？不需要，如果我们在进行一次评估之后就进行策略改善，则称为**值函数迭代算法**。![4f65e8291ad800627a3d5e6badfbbef7.jpeg](evernotecid://74F421DB-DF5E-4EEE-B947-C0C980D2DC0B/appyinxiangcom/14855437/ENNote/p75?hash=4f65e8291ad800627a3d5e6badfbbef7)
需要注意的是在每次迭代过程，需要对状态空间进行一次扫描，同时在每个状态对动作空间进行扫描以便得到贪婪的策略。

### 无模型的强化学习算法

无模型的强化学习的思想更有模型的一样：先进行策略评估，即计算出当前策略对应的值函数，然后利用值函数改进当前策略。而区别在于，无模型的强化学习算法，不知道状态转移概率$P_{ss'}$。
回顾值函数的定义：
$$
\epsilon_{\pi}\left(s\right)=E_{\pi}\left[\sum_{k=0}^{\infty}{\gamma^kR_{t+k+1}|S_t=s}\right] (1.3)  \\
q_{\pi}\left(s,a\right)=E_{\pi}\left[\sum_{k=0}^{\infty}{\gamma^kR_{t+k+1}|S_t=s,A_t=a}\right](1.4)
$$
- 动态规划的方法是利用模型对该期望进行计算。
- 在没有模型时，我们可以采用蒙特卡罗的方法计算该期望，即利用随机样本来估计期望。

在计算值函数时，蒙特卡罗方法是利用经验平均代替随机变量的期望。

***
#### 探索

如何获得充足的经验是无模型强化学习的核心所在。
无模型的方法充分评估策略值函数的前提是每个状态都能被访问到。因此，在蒙特卡洛方法中必须采用一定的方法保证每个状态都能被访问到。其中一种方法是探索性初始化。
所谓探索性初始化是指每个状态都有一定的几率作为初始状态。

策略必须是温和的，即对所有的状态s 和a 满足：$\pi\left(a|s\right)>0$。
典型的温和策略是$\varepsilon -soft$策略，即：

$\pi\left(a|s\right)\gets\left\{\begin{array}{c} 1-\varepsilon +\frac{\varepsilon}{\left| A\left(s\right)\right|}\ if\ a=arg\max_aQ\left(s,a\right)\\\\ \frac{\varepsilon}{\left| A\left(s\right)\right|}\ if\ a\ne arg\max_aQ\left(s,a\right)\\ \end{array}\right.\\\\\left(3.5\right)$

***
递增计算平均的方法：
![fc7090e02f644e3c386e5d8be31a74cb.jpeg](evernotecid://74F421DB-DF5E-4EEE-B947-C0C980D2DC0B/appyinxiangcom/14855437/ENNote/p76?hash=fc7090e02f644e3c386e5d8be31a74cb)

根据探索策略（行动策略）和评估的策略是否是同一个策略，蒙特卡罗方法又分为on-policy和off-policy.

***
##### online-policy

![abebc368829fa26b60b8d4016b41f94f.jpeg](evernotecid://74F421DB-DF5E-4EEE-B947-C0C980D2DC0B/appyinxiangcom/14855437/ENNote/p76?hash=abebc368829fa26b60b8d4016b41f94f)
![539a90c4fdf405bd9cf4710071af2073.jpeg](evernotecid://74F421DB-DF5E-4EEE-B947-C0C980D2DC0B/appyinxiangcom/14855437/ENNote/p76?hash=539a90c4fdf405bd9cf4710071af2073)

***
##### offline-policy
异策略是指产生数据的策略与评估和改善的策略不是同一个策略。我们用$\pi$表示用来评估和改进的策略，用$\mu$表示产生样本数据的策略。

利用行为策略产生的数据评估目标策略需要利用**重要性采样importance sampling**方法。
最后给出异策略每次访问蒙特卡罗算法的伪代码：

![3c5e0ea40dab30e91d2eb19439326626.jpeg](evernotecid://74F421DB-DF5E-4EEE-B947-C0C980D2DC0B/appyinxiangcom/14855437/ENNote/p76?hash=3c5e0ea40dab30e91d2eb19439326626)


### 时间差分法
时间差分(TD)方法是强化学习理论中最核心的内容，是强化学习领域最重要的成果，没有之一。与动态规划的方法和蒙特卡罗的方法比，时间差分的方法主要不同点在值函数估计上面。
时间差分方法结合了蒙特卡罗的采样方法（即做试验）和动态规划方法的bootstrapping(利用后继状态的值函数估计当前值函数)，其示意图如图4.4所示。
TD方法更新值函数的公式为(4.3)：

$V\left(S_t\right)\gets V\left(S_t\right)+\alpha\left(R_{t+1}+\gamma V\left(S_{t+1}\right)-V\left(S_t\right)\right) (4.3)$

其中$R_{t+1}+\gamma V\left(S_{t+1}\right)$称为TD目标，与（4.2）中的$G_t$相对应，两者不同之处是TD目标利用了bootstrapping方法估计当前值函数。$\delta_t=R_{t+1}+\gamma V\left(S_{t+1}\right)-V\left(S_t\right)$称为TD偏差。

下面我们从原始公式给出动态规划(DP)，蒙特卡罗方法(MC)，和时间差分方法(TD)的不同之处。

![7b193eb914a3283f2e2ad07b3c95ed4e.jpeg](evernotecid://74F421DB-DF5E-4EEE-B947-C0C980D2DC0B/appyinxiangcom/14855437/ENResource/p202)

TD 算法可以避免每次必须得到整个 Episode 才能改进 Policy。TD 算法的 Value Function V(s) 通过如下方式计算，可以看出其兼具 MC 与 TD 的特点，即 Sample 和 Bootstrap。

$V(S_t) = V(S_t) + \alpha (G_t - V(S_t)) = V(S_t) + \alpha (R_{t+1} + \gamma V(S_{t+1}) - V(S_t))$


***
online-policy

Sarsa算法

![a418b0fe2aa23743029418435c89c2ca.jpeg](evernotecid://74F421DB-DF5E-4EEE-B947-C0C980D2DC0B/appyinxiangcom/14855437/ENNote/p77?hash=a418b0fe2aa23743029418435c89c2ca)

***
offline-policy
Q-Learning


![311f6eba28e0f6b53bbe62bacd3c65af.jpeg](evernotecid://74F421DB-DF5E-4EEE-B947-C0C980D2DC0B/appyinxiangcom/14855437/ENNote/p77?hash=311f6eba28e0f6b53bbe62bacd3c65af)

***
#### MC与TD比较
* MC 需要一条完整的 Episode，而 TD 不需要；这也意味着 MC 只能处理有终止 State 的任务，而 TD 无此限制
* 由于 MC 是根据完整的 Episode 来估计 V(s) 的，因此它是 unbiased 的，但是会由于结果会受整个 Episode 的影响，带来较高的 Variance，即较大波动；相对的，TD 是根据 Episode 的中相邻状态来估计 V(s) ，结果只受少数状态影响，因此具有较低的 Variance，但必然是 biased 的
* MC 并不严格要求任务的 Markov 性，且具有很好的收敛性，但是计算量大，更新缓慢； TD 对任务的 Markov 性要求比较高，但是效率很高，一般也会收敛

***
#### DQN

Q-learning 通过贪婪的方式改进 Q(s,a) ，得到最优 Policy。但是 Q-learning 仅适用于求解小规模，离散空间问题。当 State 空间或 Action 空间规模很大，或者为连续空间时，无法再通过采样遍历足够多的 State时，Q-learning 可能就不再有效。此时，可以通过 Function Approximator 的方式解决，即利用函数 $Q(s,a;\theta)$ 逼近最优的 $Q^\*(s,a)$ ，于是我们只需要求解 $Q(s,a;\theta)$ 即可。若更进一步，还可以利用 DNN 来学习 $Q(s,a;\theta)$ ，于是就有了 DQN 的基本思想。

##### LOSS

如上文所述，有最优 Q 的迭代式为

$Q^\*(s,a) = E_{s'}[R^a_{ss'} + \gamma \max_{a'} Q^\*(s',a')|s,a]$

当我们用函数来逼近它，即

$Q^\* = E_{s'}[r + \gamma \max_{a'} Q(s',a';\theta^{-}_i)]$

于是，整个迭代过程的 Loss 可以被定义为

$
L_i(\theta_i) = E_{s,a,r}[(E_{s'}[Q^{\*}|s,a] - Q(s,a;\theta_i))^2] \\
= E_{s,a,r,s^{'}}[(Q^{\*} - Q(s,a;\theta_i))^2] + E_{s,a,r}[var_{s'}[Q^{\*}]]
$

##### 样本
除了 Loss，训练 DQN 还需要训练样本。对于 Atari 游戏过程这个时间序列事件，一个 Episode 其相邻状态相关性是非常强的，如果直接用整个 Episode 来训练，样本的强关联性与 Episode 采样数量不足会使结果方差很大。在 DQN 中，采用了 Experience Replay 的方法，即先将所有样本存储起来，然后随机采样，通过破坏样本序列强相关，同时间接增加了训练样本数量的方式，降低了方差。


##### 算法
![f704b13fe3af4fa72312a9b858b6404e.jpeg](evernotecid://74F421DB-DF5E-4EEE-B947-C0C980D2DC0B/appyinxiangcom/14855437/ENNote/p77?hash=f704b13fe3af4fa72312a9b858b6404e)
