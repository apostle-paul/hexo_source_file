---
title: notes for machine learning yearning
date: 2019-02-27 16:30:40
tags:
        - machine learning
categories:
        - reading notes
---

key points for << machine learning yearning >>
<!--more-->

### 0. scale drives machine learning progress
为什么一些深度学习的想法最近起飞了?
* 电子化的数据更多了
* 算力够大了

现在提升模型效果的可靠的路径之一仍然是：
* 弄更多的数据
* 训更大的网络

### 1. setting up development and test sets
一般的分法：
* training set: 用来跑模型的；
* dev(development) set: 用来调参，选特征等对模型进行优化调整的，有时候也叫 hold-out cross validation set；
* test set: 用来评估最终的模型效果的。
dev set 和test set很重要，核心一点：测试集要反应你的算法真实使用场景下的分布。即使你的训练集跟这个分布有差别。没有真正的测试集，也要想办法去获得。如果没有任何办法获得，那需要明白可能泛化不太好的风险。
dev set和test set应该来自同一个分布。否则提升算法的效率会很低，原因是：
* 如果是同一个分布，一旦dev set上表现得好，test set上不好，那问题很明确：算法在dev set上overfit了，需要增加dev set的数据。
* 如果是不同的分布，一旦test set上表现不好，那可能的原因就多了：
    1. dev set上overfit了。
    2. test set比 dev set上的分布更难学，所以不是算法的问题
    3. test set不是更难学，只是不同。

dev set应该多大？大到足以区分不同的算法在指标上带来的diff。例如，准确率从90.0%提升到了90.1%，那100个样本的dev set显然是不够的。
test set应该多大？大到可以非常置信地评估整个系统的性能。
处理多个指标上的优化：
* 想办法给出一个单值的指标，例如F1 score, 有利于模型的快速迭代：它可以让你从多个结果中快速选出最好的那个算法，会有一个明确的提升方向。
* 区分出来satisficing metric和optimizing metric。

什么时候更换dev/test sets和metrics? 对于一个新的项目，通常会很快先初始化一个dev/test sets和metrics，但是这些不一定真的合适，那么什么时候换呢？
* 发现dev/test set跟应用中的真实分布不一致
* 在dev set上overfit了。这里要注意，不要用test set来做任何关于算法的优化，否则很快就会在test set上也overfit了，就再也没有对系统无偏的评估了。
* 优化目标变了，就要换metric了。例如，一开始的优化目标是识别猫，后来，你发现，增加一个过滤掉色情图片的优化目标。
总之就是，一旦发现方向不对，就该换了。

### 2. Basic Error Analysis
不要一开始就尝试建立一个完美的系统，先建立一个基本的系统，然后不断迭代。所谓小步快跑。
错误分析的时候，可以抽一些错误样本来一是看某一种类型的错误占比，由
