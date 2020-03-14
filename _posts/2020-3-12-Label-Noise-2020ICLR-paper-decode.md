---
layout: post
title: "论文解读--DivideMix: Learning With Noisy Labels as Semi-supervised Learning"
categories: 
- Noisy Labels
tags: 
- Noisy Labels
- Semi-supervised Learning
- ICLR2020
author: 千暮云兮
mathjax: true
---

* content
{:toc}

本篇博客将分享一篇来自2020ICLR的噪声标签处理文章。在噪声标签处理方向目前比较流行的主要方法有三个：

1) 利用网络优先拟合易学习的样本（干净标签样本）的特性设计相应的训练方式，使得模型在正确的阶段得到训练而减少其过多拟合噪声标签数据；

2）设计无偏的优化函数，使得模型在含噪声标签的数据集上训练与在干净标签数据集训练效果相同；

3）重新对数据集的标签进行规划，将数据集分为干净数据和未标记的噪声数据，利用半监督学习的方法进行模型的训练；而该文章是属于第三种方法。

## 1 主要贡献

1）使用高斯混合模型 (Gaussian Mixture Model) 拟合每个样本的损失分布，并以此将数据集样本划分为有标签样本（对应于干净标签样本）和无标签样本（对应于噪声标签样本）。为了减少验证性偏误 (confirmation bias) 文章采用两个独立的网络交替处理划分的数据集，即网络1划分的数据集由网络2处理，反之亦如此。

2）在MixMatch模型基础上，增加了协同微调 (co-refinement) 和协同预测 (co-guessing) 。协同微调是针对于网络预测为干净标签样本的数据，利用网络的输出对该干净标签 (ground-truth labels) 进行微调（相当于为干净标签也增加了不确定性，有利于网络的鲁棒训练），而协同的含义在于预测网络处理的数据集为另一个网络在上轮迭代中划分的数据集。协同预测是针对网络预测为噪声标签样本（无标签样本），使用两个网络预测的综合来替换原来的标签。

```
注: 验证性偏误 (confirmation bias) 的意思是当使用模型本身划分的数据集来训练模型时，数据集划分的错误之处很可能会被模型忽略掉，且模型会不断去拟合这些错误。具体详见 Antti Tarvainen and Harri Valpola. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. In NIPS, pp. 1195–1204, 2017.
```

## 2 整体思路

DivideMix模型主要是通过构建两个同构不同参的网络，利用网络的预测输出对数据集进行标记集（干净标签样本集）和未标记集（噪声标签样本集）进行划分，然后各自网络针对另一个网络划分的数据集进行优化，具体模型示意图如图1所示。首先，每个epoch内含有多个mini-batch，而每个epoch间会存在一个co-divide过程。co-divide就是对数据集进行划分的过程，具体的操作过程下文将详细介绍。在每个epoch中A和B网络各自接收来自对方划分的数据集进行训练，而在每个mini-batch之间各自的数据集内标签将会根据网络的输出进行调整（MixMatch过程）。下面，将对模型两个关键点Co-Divide和MixMatch进行详细说明。

![avatar](https://i.loli.net/2020/03/13/VvpwM4lEzoT9RDj.png) <center> **图1.** DivideMix整体模型示意图</center>

## 3 关键技术

首先需要知道深度网络的一个特性：即在训练过程中会优先学习干净标签样本（噪声标签处理三种主流方法的第一种就是利用这一特性）。而反应在交叉熵损失函数值上就是对于干净标签样本损失值都较小，对于噪声标签样本网络的损失值较大。对于干净标签样本损失值的集合便可以看成是一个分布，这个分布理想情况下均值应该为0（但实际肯定是大于0），实际上分布可以类似成一个正态分布（个人觉得应该是均值为0的右半边正态分布）。同理，独眼噪声标签样本损失值的集合也是可以类似于一个正态分布，而这个正态分布的均值和方差未知（与网络参数有关）。在这样的前提下，我们需要做的是尽可能优化模型参数，使得这两个分布尽可能分开，理想情况是两者没有交叠。而可想而知，在模型未经过很好的训练时，两者往往是交叠在一起的（如图2）。因此，该模型使用EM算法拟合二维高斯混合模型。文章中提到，对于某个样本$x_{i}$而言，该样本标签干净的概率$w_{i}$为后验概率$p(g|l_{i})$，其中$g$为混合高斯中均值较小的那个。对于这点做一下解释，该二维高斯混合模型第一个高斯分布为干净标签样本损失值分布，第二个高斯分布为噪声标签样本损失值分布。首先，需要明确的是干净标签样本损失值往往比噪声标签样本损失值小，也就是第一个的高斯分布均值小于第二个。而样本属于标签干净高斯分布的概率也就混合高斯模型中第一项高斯分布的权重，故而其对应的$g$均值更小（相比噪声标签高斯分布）。在得到每个样本属于干净标签样本的概率后，只需要设置一个阈值便可以对数据集进行标记数据和未标记数据的划分了。

以上便是Co-DibideMix的核心思想，具体算法见图3伪代码中4-8行。大致流程为，将数据集分别输入到两个网络中，根据各自网络计算得到损失值构建GMM，继而得到样本标签干净的概率分布。然后根据阈值分别对数据集进行划分，划分的数据集将用于另一个网络的后续训练。而这一数据集划分过程在每个batch都会进行。

需要注意的是，在整个算法开始开始前需要对网络进行 “warm up”（即热启动），而在本文中的热启动与一般网络的热启动还不太一样。一般网络的热启动通常是在正常训练前先使用小学习率进行训练，而该文的热训练主要利用网络优先学习干净标签样本数据的特点，在进行co-dividemix前有一个对干净标签样本数据利好的网络。但对于非均匀噪声 (asymmetric noisy)，由于该噪声是数据集中部分类别有序的被标记为另一些类别，使得相对均匀噪声而言其熵更低（均匀噪声随机性更大，噪声标签样本熵更高）。因此，存在非均匀噪声的数据集整体的交叉熵损失值都偏小（见图2a），影响网络的warm up效果。这里作者使用最常见的方法，即使用香农熵正则项进行约束。对于输入$x$而言，该正则项为：

在交叉熵损失基础上增加该熵正则项可以使得网络倾向输出0或者1，从而增强网络对输入样本的鉴别能力，防止输出的类别概率预测都非常接近。具体效果见图2b，可以看出两个分布的区分度相比2a图更加明显且分布更加接近正态。图2c表示的是在经过warm up后20个epoch的dividemix训练分布，最主要的变化是干净标签样本的损失更加趋近于0了，说明网络更多的在拟合干净标签样本而遗弃非干净标签样本。

![avatar](https://i.loli.net/2020/03/13/gcbzl1UwJiRQD4m.png) <center> **图2.** 在CIFAR-10中损失函数经验分布</center>

因此我们可以在获取剩余时间的时候，每次 new 一个设备时间，因为设备时间的流逝相对是准确的，并且如果设备打开了网络时间同步，也会解决这个问题。

但是，如果用户修改了设备时间，那么整个倒计时就没有意义了，用户只要将设备时间修改为倒计时的 endTime 就可以轻易看到倒计时结束是页面的变化。因此一开始获取服务端时间就是很重要的。

![avatar](https://i.loli.net/2020/03/13/7L2oZCWrlKkzHwn.png) <center> **图3.** DivideMix模型算法</center>

简单的说，一个简单的精确倒计时原理如下：

- 初始化时请求一次服务器时间 serverTime，再 new 一个设备时间 deviceTime
- deviceTime 与 serverTime 的差作为时间偏移修正
- 每次递归时 new 一个系统时间，解决 setTimeout 不准确的问题

## 代码

获取剩余时间的代码如下：

```js
/**
 * 获取剩余时间
 * @param  {Number} endTime    截止时间
 * @param  {Number} deviceTime 设备时间
 * @param  {Number} serverTime 服务端时间
 * @return {Object}            剩余时间对象
 */
let getRemainTime = (endTime, deviceTime, serverTime) => {
    let t = endTime - Date.parse(new Date()) - serverTime + deviceTime
    let seconds = Math.floor((t / 1000) % 60)
    let minutes = Math.floor((t / 1000 / 60) % 60)
    let hours = Math.floor((t / (1000 * 60 * 60)) % 24)
    let days = Math.floor(t / (1000 * 60 * 60 * 24))
    return {
        'total': t,
        'days': days,
        'hours': hours,
        'minutes': minutes,
        'seconds': seconds
    }
}
```

<del>获取服务器时间可以使用 mtop 接口 `mtop.common.getTimestamp` </del>

然后可以通过下面的方式来使用：

```js
// 获取服务端时间（获取服务端时间代码略）
getServerTime((serverTime) => {

    //设置定时器
    let intervalTimer = setInterval(() => {

        // 得到剩余时间
        let remainTime = getRemainTime(endTime, deviceTime, serverTime)

        // 倒计时到两个小时内
        if (remainTime.total <= 7200000 && remainTime.total > 0) {
            // do something

        //倒计时结束
        } else if (remainTime.total <= 0) {
            clearInterval(intervalTimer);
            // do something
        }
    }, 1000)
})
```

这样的的写法也可以做到准确倒计时，同时也比较简洁。不需要隔段时间再去同步一次服务端时间。

## 补充

在写倒计时的时候遇到了一个坑这里记录一下。

**千万别在倒计时结束的时候请求接口**。会让服务端瞬间 QPS 峰值达到非常高。

![](https://img.alicdn.com/tfs/TB1LBzjOpXXXXcnXpXXXXXXXXXX-154-71.png)

如果在倒计时结束的时候要使用新的数据渲染页面，正确的做法是：

在倒计时结束前的一段时间里，先请求好数据，倒计时结束后，再渲染页面。

关于倒计时，如果你有什么更好的解决方案，欢迎评论交流。
