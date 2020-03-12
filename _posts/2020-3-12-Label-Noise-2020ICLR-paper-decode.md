---
layout: post
title: "论文解读--DivideMix: Learning With Noisy Labels as /Semi-supervised Learning"
categories: 噪声标签
tags: 噪声标签 半监督学习 ICLR2020
author: 千暮云兮
published: true
---

* content
{:toc}

本篇博客将分享一篇来自2020ICLR的噪声标签处理文章。在噪声标签处理方向目前比较流行的主要方法有三个：<br>
1）利用网络优先拟合易学习的样本（干净标签样本）的特性设计相应的训练方式，使得模型在正确的阶段得到训练而减少其过多拟合噪声标签数据；<br>
2）设计无偏的优化函数，使得模型在含噪声标签的数据集上训练与在干净标签数据集训练效果相同；<br>
3）重新对数据集的标签进行规划，将数据集分为干净数据和未标记的噪声数据，利用半监督学习的方法进行模型的训练；而该文章是属于第三种方法。

## 主要贡献及优势

1）使用高斯混合模型(Gaussian Mixture Model)拟合每个样本的损失分布，并以此将数据集样本划分为有标签样本（对应于干净标签样本）和无标签样本（对应于噪声标签样本）。为了减少验证性偏误(confirmation bias)[^1]文章采用两个独立的网络交替处理划分的数据集，即网络1划分的数据集由网络2处理，反之亦如此。<br>
2）在MixMatch模型基础上，增加了协同微调(co-refinement)和协同预测(co-guessing)。协同微调是针对于网络预测为干净标签样本的数据，利用网络的输出对该干净标签(ground-truth labels)进行微调（相当于为干净标签也增加了不确定性，有利于网络的鲁棒训练），而协同的含义在于预测网络处理的数据集为另一个网络在上轮迭代中划分的数据集。协同预测是针对网络预测为噪声标签样本（无标签样本），使用两个网络预测的综合来替换原来的标签。

[^1]: 验证性偏误(confirmation bias)的意思是当使用模型本身划分的数据集来训练模型时，数据集划分的错误之处很可能会被模型忽略掉，且模型会不断去拟合这些错误。

因此我们可以在获取剩余时间的时候，每次 new 一个设备时间，因为设备时间的流逝相对是准确的，并且如果设备打开了网络时间同步，也会解决这个问题。

但是，如果用户修改了设备时间，那么整个倒计时就没有意义了，用户只要将设备时间修改为倒计时的 endTime 就可以轻易看到倒计时结束是页面的变化。因此一开始获取服务端时间就是很重要的。

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
