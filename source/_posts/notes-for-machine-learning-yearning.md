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
* 为什么一些深度学习的想法最近起飞了?
  1. 电子化的数据更多了
  2. 算力够大了

* 现在提升模型效果的可靠的路径之一仍然是：
  1. 弄更多的数据
  2. 训更大的网络

### 1. setting up development and test sets
* 一般的分法：
  1. training set: 用来跑模型的；
  2. dev(development) set: 用来调参，选特征等对模型进行优化调整的，有时候也叫 hold-out cross validation set；
  3. test set: 用来评估最终的模型效果的。

* dev set 和test set很重要，核心一点：测试集要反应你的算法真实使用场景下的分布。即使你的训练集跟这个分布有差别。没有真正的测试集，也要想办法去获得。如果没有任何办法获得，那需要明白可能泛化不太好的风险。

* dev set和test set应该来自同一个分布。否则提升算法的效率会很低，原因是：
  1. 如果是同一个分布，一旦dev set上表现得好，test set上不好，那问题很明确：算法在dev set上overfit了，需要增加dev set的数据。
  2. 如果是不同的分布，一旦test set上表现不好，那可能的原因就多了：
    1. dev set上overfit了。
    2. test set比 dev set上的分布更难学，所以不是算法的问题
    3. test set不是更难学，只是不同。

* dev set应该多大？大到足以区分不同的算法在指标上带来的diff。例如，准确率从90.0%提升到了90.1%，那100个样本的dev set显然是不够的。
test set应该多大？大到可以非常置信地评估整个系统的性能。

* 处理多个指标上的优化：
  1. 想办法给出一个单值的指标，例如F1 score, 有利于模型的快速迭代：它可以让你从多个结果中快速选出最好的那个算法，会有一个明确的提升方向。
  2. 区分出来satisficing metric和optimizing metric。

* 什么时候更换dev/test sets和metrics? 对于一个新的项目，通常会很快先初始化一个dev/test sets和metrics，但是这些不一定真的合适，那么什么时候换呢？

* 发现dev/test set跟应用中的真实分布不一致

* 在dev set上overfit了。这里要注意，不要用test set来做任何关于算法的优化，否则很快就会在test set上也overfit了，就再也没有对系统无偏的评估了。

* 优化目标变了，就要换metric了。例如，一开始的优化目标是识别猫，后来，你发现，增加一个过滤掉色情图片的优化目标。

* 总之就是，一旦发现方向不对，就该换了。

### 2. Basic Error Analysis
* 不要一开始就尝试建立一个完美的系统，先建立一个基本的系统，然后不断迭代。所谓小步快跑。

* 然后进行错误分析，即，check模型misclassify的样本。

* 进行错误分析的时候，通常情况下，会首先进行分类。check这些错误的样本属于那种错误分类，以此来确定后续需要重点解决的问题。
在进行错误分析的时候，在dev set 中会遇到mislabel的样本，是否需要进行纠正，取决于这部分mislabel的样本的占比。如果要进行纠正，需要注意对正确分类的样本，以及test set都进行纠正，以保证无偏和同分布。

* 如果有一个比较大的dev set，可以抽出来一部分样本进行观察。这部分抽出来的样本叫做eyeball dev set，剩下的叫blackbox dev set。这样做的好处是，可以根据eyeball 上和blackbox上的差别，来确定是否开始在eyeball上过拟合了。如果是，那就需要换eyeball dev set了。
这两个set要多大呢？

* 对于human可以做好的任务，能有100个error的eyeball set是比较好的。

* blackbox, 则是要保证能够对不同算法好和参数的好还进行区分，比如1000-10000的样本量。
* 如果dev set太小，优选设置eyeball set。

### 3. Bias and Variance
* 机器学习中error的两大来源，bias和variance。分析清楚bias和variance，有助于我们理解模型所处的状态。

* 可以简单地理解为：bias是training set上的error，variance是dev/test set上的error。

* 首先要把bias和variance跟optimal error rate进行比较。即，人工可以达到的最低错误率。因为机器学习终究还是在学人工的结果（label是人工标的）。optimal error rate 也叫unvoidable bias，或者Bayes error rate。

* 降低bias常用的手段：
  1. 增加模型的容量。
  2. 根据error analyse 调整特征
  3. 减少正则
  4. 调整模型结构
  5. 增加训练数据

* 降低variance常用的手段：
  1. 增加训练数据
  2. 增加正则
  3. early stopping
  4. feature selection，减少特征：对小模型可能更好使。对深度模型没那么必要。
  5. 减少模型容量：不推荐，要小心。
  6. 根据error analyse 调整特征
  7. 调整模型结构。

### 4. Learning Curve
* 一种learning curve是dev error随训练样本数量的变化曲线，可以帮助确定是否还需要搜集更多的数据。
* 再加上training error，可以帮助分析模型的bias和variance。

### 5. Comparing to human-level preformance
* 在人工可以做好的任务上建模会比较容易，因为：
  1. 比较容易获取置信的label
  2. 容易做error analyse
  3. 有明确的optimal error rate

* 模型表现比人工还好的时候，继续增加高质量的人工数据还是会有帮助，因为模型还是可以从中获取有用的信息。

### 6. Training and testing on different distributions
* 目标是要在数据集A上做好，现在有个数据集B，A B来自不同的分布，怎么用？

* 这里推荐的办法是，把A 分成两半，A/2作为test/dev set， A/2 + B来进行训练。

* 另外需要考虑的是，模型的容量是否够大。如果够大，模型不会把真实的分布A/2忘记，就可以加。否则，可能会变得更差。

* 完全不相关的数据，也不要加。

* 还有一个判断是否要加的办法是根据人工的表现。如果人工可以通过学习B，得到A的结果，那加了B是有益的。

* 还有一个技巧是调整样本的权重。

* 使用不一致的数据进行训练，可能遇到data mismatch的问题：模型已经是low variance了，但是在目标数据集上，error还是很大。

* 怎么处理data mismatch问题？
  1. 尝试去理解分布到底有哪些不同
  2. 找其他match的更好的数据。

* 如果能够做到第一点，那就可以尝试去做数据的人工合成。但是需要注意的是，人工合成数据容易overfit，需要很小心仔细。

### 7. Debugging inference algorithms
* 在实际inference的工程中，还需要考虑召回。最终的效果不好，可能是召回得不好。

### 8. End-to-end deep Learning
* end to end 虽好，但是要求在两端都有足够多的数据。数减少了不行。
* 如果是很复杂的任务，其实应该分解成几个简单的。比如，要判断是否是波斯猫，需要先判断是否是猫，再判断这个猫是否是波斯猫。

### 9. Error analysis by parts
