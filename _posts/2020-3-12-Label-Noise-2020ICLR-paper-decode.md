---
layout: post
title:  "2020ICLR噪声标签论文解读--DivideMix: Learning With Noisy Labels as Semi-supervised Learning"
categories: 噪声标签
tags:  噪声标签 半监督学习
author: 千暮云兮
---

* content
{:toc}

本篇博客将分享一篇来自2020ICLR的噪声标签处理文章。在噪声标签处理方向目前比较流行的主要方法有三个：1）利用网络优先拟合易学习的样本（干净标签样本）的特性设计相应的训练方式，使得模型在正确的阶段得到训练而减少其过多拟合噪声标签数据；2）设计无偏的优化函数，使得模型在含噪声标签的数据集上训练与在干净标签数据集训练效果相同；3）重新对数据集的标签进行规划，将数据集分为干净数据和未标记的噪声数据，利用半监督学习的方法进行模型的训练；而该文章是属于第三种方法。

## 绪论
小样本学习（或类似）这一概念已经提出很长时间，但在初期由于定义不明确以及基于小样本学习的图像处理任务的不统一，使得其并没有
引起学界的关注。但2016年的Matching Networks (NIPS2016)首先对基于小样本学习的图像分类任务进行了明确定义并给出了一种基于eposide的训练方式成功的给该任务定下了benchmark，其主要是将每个batch的数据将分成support set和query set，通过support set和样本和标签给query set中样本进行标签的预测。自此两年间不断有新的小样本分类模型被提出，而在小样本分类标准数据集（Omniglot, mini-ImageNet, tiered-ImageNet等）上的识别精度也在不断的提高。

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
