# article-classification
一种基于关键句评分策略的篇章分类算法。该算法源自新浪财经面试时给出的一道开放性题目。经过两天的紧张准备，制定了该解决方案，并付诸实施，效果还算理想，在测试集能在95%以上的准确率。
+ 题目描述：新浪财经给出五千余篇财经新闻数据，同时带有不同的标签。数据主要内容是关于“苹果”，主要判别文章是属于期货苹果还是苹果公司。数据共有两类的标签，一种是“期货”标签，一种是“股票”标签。目的是训练一个分类器，能够准确区分这些篇章数据。
+ 方案思路：使用期货、股票中具有代表性的关键词筛选出每天文章中的关键句，这些关键句具有明显的类别特征；然后，将筛选的数据生成训练数据集，训练一个基于bert+softmax的二分类判别模型；该训练后的模型，在训练集上随机抽取的验证数据中，准确达到98%。具体评分思路，详细请看代码。
+ 由于训练数据和预训练模型过大，不上传github，如有调试需要，请邮件作者。
<p/>
# Author：hailei.yan
# Data: 3月20日
# Email：hailei2014@163.com
