---
title: Deep Reinforcement Learning for Sponsored Search Real-time Bidding
date: 2019-07-05 11:42:50
mathjax: true
tags:
  - Reinforcement learning
categories:
  - paper notes
---

# 1. problem definition
最开始，问题的定义是：
$$
max cv/cost
s.t. cv > g
$$
都是per day, per userid的setting。即某个广告主，在满足一天转化量大于某个最小值g的限定下， 最大化cpa的倒数。

论文讲问题等价为：
$$
max cv
s.t. cost = c
$$
即，在一天消费一定的情况下，最大化cv量。

# 2. model setup
state s: s = <b, t, auct>
* b: 剩下的预算
* t: 决策序列的步数
* auct: 跟竞拍相关的特征向量

Action a: 实时竞拍的bid

Reward r(s, a): 在s下执行a，得到的cv，即cv。

Episode ep： 一天是一个ep
模型目标是找到最优的$$\pi(s)$$，讲s映射到一个动作a，使累积的reward最大：$$\Sigma \gamma^(i-1) r(s_i, a_i)$$。

秒级别和分钟级别的流量状态变化太剧烈，没有固定的pattern，所以在小时级别进行聚合。这样一个episode，被分为了24个step，每个小时一个。
而状态s中的auct，包括了clk_sum, show_sum, cost_sum, ctr, cvr, acp等。
这样在不同的天，就可以得到一个固定的状态转移概率。

action也不是直接设置bid，而是设置pcvr前面乘的系数$$\alpha$$。

# 3. Algorithms
套用了一下DQN

# 4. Massive-agent Model
当大量agent一起竞争的时候，之前建立的模型的效果会减弱。
所以，引入了一个全局合作的机制，在各个agent选择了动作之后，agent会受到自己的reward和一个全局的reward。

# 5. Experiment
随机选了1000个大客户来进行实验，覆盖1亿pv。用pcvr来模拟reward。
