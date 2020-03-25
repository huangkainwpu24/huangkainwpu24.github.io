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

![1](https://i.loli.net/2020/03/13/VvpwM4lEzoT9RDj.png) <center>图1. DivideMix整体模型示意图</center>

## 3 关键技术

### 3.1 Co-DivideMix

首先需要知道深度网络的一个特性：即在训练过程中会优先学习干净标签样本（噪声标签处理三种主流方法的第一种就是利用这一特性）。而反应在交叉熵损失函数值上就是对于干净标签样本损失值都较小，对于噪声标签样本网络的损失值较大。对于干净标签样本损失值的集合便可以看成是一个分布，这个分布理想情况下均值应该为0（但实际肯定是大于0），实际上分布可以类似成一个正态分布（个人觉得应该是均值为0的右半边正态分布）。同理，独眼噪声标签样本损失值的集合也是可以类似于一个正态分布，而这个正态分布的均值和方差未知（与网络参数有关）。

在这样的前提下，我们需要做的是尽可能优化模型参数，使得这两个分布尽可能分开，理想情况是两者没有交叠。而可想而知，在模型未经过很好的训练时，两者往往是交叠在一起的（如图2）。因此，该模型使用EM算法拟合二维高斯混合模型。文章中提到，对于某个样本$x_{i}$而言，该样本标签干净的概率$w_{i}$为后验概率$p\left(g \mid \ell_{i}\right)$，其中$g$为混合高斯中均值较小的那个。

对于这点做一下解释，该二维高斯混合模型第一个高斯分布为干净标签样本损失值分布，第二个高斯分布为噪声标签样本损失值分布。首先，需要明确的是干净标签样本损失值往往比噪声标签样本损失值小，也就是第一个的高斯分布均值小于第二个。而样本属于标签干净高斯分布的概率也就混合高斯模型中第一项高斯分布的权重，故而其对应的$g$均值更小（相比噪声标签高斯分布）。在得到每个样本属于干净标签样本的概率后，只需要设置一个阈值便可以对数据集进行标记数据和未标记数据的划分了。

以上便是Co-DibideMix的核心思想，具体算法见图3伪代码中4-8行。大致流程为，将数据集分别输入到两个网络中，根据各自网络计算得到损失值构建GMM，继而得到样本标签干净的概率分布。然后根据阈值分别对数据集进行划分，划分的数据集将用于另一个网络的后续训练。而这一数据集划分过程在每个batch都会进行。

需要注意的是，在整个算法开始前需要对网络进行 “warm up”（即热启动），而在本文中的热启动与一般网络的热启动还不太一样。一般网络的热启动通常是在正常训练前先使用小学习率进行训练，而该文的热训练主要利用网络优先学习干净标签样本数据的特点，在进行co-dividemix前有一个对干净标签样本数据利好的网络。但对于非均匀噪声 (asymmetric noisy)，由于该噪声是数据集中部分类别有序的被标记为另一些类别，使得相对均匀噪声而言其熵更低（均匀噪声随机性更大，噪声标签样本熵更高）。因此，存在非均匀噪声的数据集整体的交叉熵损失值都偏小（见图2a），影响网络的warm up效果。这里作者使用最常见的方法，即使用香农熵正则项进行约束。对于输入$x$而言，该正则项为：

$$\mathcal{H}=-\sum_{c} \mathrm{p}_{\text {model }}^{\mathrm{c}}(x ; \theta) \log \left(\mathrm{p}_{\text {model }}^{\mathrm{c}}(x ; \theta)\right)$$

在交叉熵损失基础上增加该熵正则项可以使得网络倾向输出0或者1，从而增强网络对输入样本的鉴别能力，防止输出的类别概率预测都非常接近。具体效果见图2b，可以看出两个分布的区分度相比2a图更加明显且分布更加接近正态。图2c表示的是在经过warm up后20个epoch的dividemix训练分布，最主要的变化是干净标签样本的损失更加趋近于0了，说明网络更多的在拟合干净标签样本而遗弃非干净标签样本。

![2](https://i.loli.net/2020/03/13/gcbzl1UwJiRQD4m.png) <center>图2. 在CIFAR-10中损失函数经验分布</center>

![3](https://i.loli.net/2020/03/13/7L2oZCWrlKkzHwn.png) <center>图3. DivideMix算法伪代码</center>

### 3.2 Modified MixMatch

之前提及，作者在MinMatch的基础增加了协同微调 (co-refinement) 和协同预测 (co-guessing)，那么接下来将详细介绍这两点。

首先是co-refinement模块，其主要是针对标记的样本集，即将标记样本数据的ground-truth标签与网络的预测输出根据该样本属于干净标签的概率进行线性融合。

假设一个mini-batch中标记的样本、对应的one-hot标签以及属于干净标签的概率用集合$\hat{\mathcal{X}}$表示，其中有

$$
\hat{\mathcal{X}} = \left\{\left(x_{b}, y_{b}, w_{b}\right);b\in(1,\ldots, B)\right\}
$$

 网络对标记样本数据的类别预测输出为$p_{b}$，那么线性融合过程为：

$$
\bar{y}_{b}=w_{b} y_{b}+\left(1-w_{b}\right) p_{b}
$$

同时对线性融合的输出进行锐化（不太清楚其作用）：

$$
\hat{y}_{b}=\operatorname{Sharpen}\left(\bar{y}_{b}, T\right)={\bar{y}_{b}^{c}}^{\frac{1}{T}} / \sum_{c=1}^{C} {\bar{y}_{b}^{c}}^ \frac{1}{T}, \text { for } c=1,2, \dots, C
$$

那么最终得到的$\hat{y}_{b}$将作为标记样本的新标签参与后续的训练，具体伪代码片段为图3中17-19行。

接下来是co-guessing模块，其主要是针对未标记的样本集，与对标记数据标签微调不同的是，未标记样本是没有ground-truth标签的，因此作者便使用两个网络的预测融合（加权平均）的结果作为未标记样本的预测标签（说到底还是尽可能利用上可以利用的信息，增强标签的鲁棒性）。这一点文章里面也是略提了一下，其实和标记样本的协同微调十分相似，只是融合过程不同，后续都经过了输出锐化。

那么最后便是MixMatch部分了，MixMatch主要是研究如何将两个样本（和标签）进行融合。具体的算法内容可参见[Unsupervised label noise modeling and loss correction](https://arxiv.org/abs/1904.11238)这篇文章，这里就不对其进行详细介绍了（主要是还没看），下面是其计算过程：

$$
\begin{aligned}
&\lambda \sim \operatorname{Beta}(\alpha, \alpha)\\
&\lambda^{\prime}=\max (\lambda, 1-\lambda)\\
&x^{\prime}=\lambda^{\prime} x_{1}+\left(1-\lambda^{\prime}\right) x_{2}\\
&p^{\prime}=\lambda^{\prime} p_{1}+\left(1-\lambda^{\prime}\right) p_{2}
\end{aligned}
$$

第一个公式代表beta分布，由于其两个参数相等（文章设置为4），因此其是关于0.5对称的概率分布。由于beta分布可以描述概率的概率分布，因此将从beta分布中随机出概率值作为两个样本融合的权重。第二个公式是为了保证第一个样本的权重始终大于0.5，相当于在样本融合过程中控制结果更倾向于第一个样本（其实全部倾向于第二个也是一样的），需要注意的是文章在对该公式的解释是MixMatch会将$\hat{\mathcal{X}}$和$\hat{\mathcal{U}}$转换为$\mathcal{X'}$和$\mathcal{U'}$，而该权重设置将会使得$\mathcal{X'}$更加接近于$\hat{\mathcal{X}}$而非$\hat{\mathcal{U}}$，这一点就目前想确实有点难以理解，但在后续的代码解读部分会涉及到这点，通过该部分的代码段便可以很轻松理解这话的含义了。然后第三个和第四个公式分布就是数据的融合和标签的融合。

```
注：模型还使用于一个叫作多重数据增强 (multiple augmentations)的技术，即对于听一个样本使用不同的图像变化及增强，如随机裁剪或随机翻转等。之后同时将这些由同一个样本不同数据增强的复样本（暂且这么命名）输入到网络中，网络对该样本的预测将是这些复样本预测的加权平均。而上述标记样本和未标记样本都会用多重数据增强的方式，大致的作用应该就是可以进一步加强网络输出的稳定性和鲁棒性。
```

损失部分就不做过多介绍了，作者在优化$\mathcal{X'}$时使用的是交叉熵损失，而$\mathcal{U'}用的是均方差损失。同时为了防止在噪声占比较大时模型将样本全预测成同一类别，增加了一个正则项：

$$
\mathcal{L}_{\mathrm{reg}}=\sum_{c} \pi_{c} \log \left(\pi_{c} / \frac{1}{\left|\mathcal{X}^{\prime}\right|+\left|\mathcal{U}^{\prime}\right|} \sum_{x \in \mathcal{X}^{\prime}+\mathcal{U}^{\prime}} \operatorname{p}_{\text {model }}^{\mathrm{c}}(x ; \theta)\right)
$$

总的损失是这几项的加权和：

$$
\mathcal{L}=\mathcal{L}_{\mathcal{X}}+\lambda_{u} \mathcal{L}_{\mathcal{U}}+\lambda_{r} \mathcal{L}_{\text {reg }}
$$

到此，该模型的几个核心思想及内容都已全部陈述完毕了。

## 4 模型实验效果

和其他噪声标签算法一样，作者将该模型分别应用于CIFAR-10, CIFAR-100, Clothing1M和WebVision数据集，具体各数据集就不详细介绍了。对于CIFAR-10和CIFAR-100数据集使用的base network是PreAct Resnet-18，对于Clothing1M和WebVision数据集使用的是预训练的Resnet-50网络。

对于CIFAR-10和CIFAR-100数据集由于其本身数据标签是干净的，因此需要人为加入噪声以模拟噪声标签的场景。通常有两种噪声标签，一个是均匀噪声 (symmetric noise)，其相当于是在整个数据集中随机对部分标签加噪（将正确类别变成其他错误类别）。另一种是非均匀噪声 (asymmetric noise)，其是针对数据集中几个较为容易被混淆的类别数据加噪（将部分特定类别的样本标签变成容易混淆的另一个类别）。文章中给出的实验表格显示，在这两个数据集两种噪声的实验中该模型均有肉眼可见的精度提高，具体详见文章。

对于Clothing1M和WebVision数据集，由于其本身就是从互联网上采集而制作的数据集，本身就含有很多复杂标签噪声，因此不需要人为加噪。相比上面两个数据集，这两个数据集可以相当于实际情况下含噪声标签的数据集了。同样，该模型在这两个数据集上测试的结果也明显优于其他算法。

最后，作者在CIFAR-10和CIFAR-100数据集上做了灵敏度分析，如研究测试阶段双网络输出融合（平均）相比单网络预测输出的实验、研究训练阶段双网络的co-training相比单网络的self-training的实验、研究标签微调和输入数据增强的实验和研究将self-divide加入MixMatch的对照实验等等，当时种种这些实验也进一步说明了原模型的优势。

## 5 额外点

下面介绍两个这篇文章所提及的我比较感兴趣的点，这两个都是作者所做的额外实验，放在附录上的。

### 5.1 AUC (Area Under a Curve)

这里作者进一步分析了不同噪声占比下的auc随着训练epoch的变化实验，以说明随着训练的进行auc不断增加直至稳定，模型训练后的综合分类能力得到提升。

这里简单介绍下AUC这一个概念，AUC其实和precision差不多，都是评价分类模型的效果与能力的指标。对于二分类问题来说，precision通常需要设定一个阈值（如0.5）以将模型的输出进行类别的划分。但是有时候我们是无法十分明确地确定这个阈值的，或者说这个阈值会随着不同数据集的变化而变化，那么auc就是用来评价不同阈值下模型整体的分类性能的，即这个阈值将在0到1变化，AUC的值将反应了阈值不固定情况下模型性能。而多分类问题其实可以看成是多个二分类的叠加问题，后续介绍AUC计算方法时也会用到这一点。

上面说到precision指标几乎所有分类模型都会有，其可以非常直观的展示模型的好坏。但对于样本不平衡的数据集而言，仅使用precision是不准确的，其会随着训练集和测试集样本类别分布的不同而体现出不同的识别精度。AUC指标此时就可以发挥重要作用了，当数据集的样本类别分布变化很大时该曲线仍可以维持不变。

对于一个二分类问题，其标签要么是真 (positive)要么是假 (negative)。那么模型预测输出就会出现以下四种情况：模型预测样本为真而实际标签也是真--真阳性 (true positive)、模型预测样本为真但实际标签为假--假阳性 (false positive)、模型预测样本为假但实际标签为真--假阴性 (false negative)、模型预测样本为假而实际标签也为假--真阴性 (true negative)，可以看出positive和negative是针对模型对样本的预测情况，而true和false是针对预测的标签和真实标签对应关系，一致则为true否则为false。在清楚这些概念后便可以引出ROC (receiver operating characteristic curve) 曲线了，该曲线的横坐标为假阳性率 (false positive rate)，纵坐标为为真阳性率 (true positive rate)。假阳性率则为假阳性与真实标签为假的比例，真阳性率为真阳性与真实标签为真的比例。可以看出，对于模型来说我们希望其真阳性率越高越好，也即该曲线越往左上靠越好。而AUC的定义即该ROC曲线下的面积，因此该面积越大则越说明模型更加优异。

那么接下来该如何实现计算多分类情况下的AUC呢，目前主要有两种方法。

在介绍这两种方法之前，有如下说明。首先假设测试集样本的个数为$N$个，类别数为$C$个，那么模型对每个测试样本将会输出各个类别的概率（经过softmax），即一个样本得到的类别概率分布向量为$1 \times C$，对于全部测试集样本得到的类别概率分布矩阵为$N \times C$。同时将测试样本标签写成one-hot形式，同样可以得到一个$N \times C$的01稀疏矩阵。

第一种方法就是把多分类看成多个单分类，先求多个二分类的ROC曲线，继而通过取平均的方法得到总ROC曲线，最后得到多分类的AUC。对于$C$个类别中的任一类别，这$N$个测试样本都将有个概率值，那么根据这些概率值与one-hot标签的关系可以得到各阈值下的假阳性率和真阳性率，从而可以得到该类别的ROC曲线。$C$个类别就将有$C$个这样的ROC曲线，那么将这些曲线取平均，即可得到该多分类的ROC曲线，从而可以得到其AUC。

第二种方法是直接在矩阵层面将其转换成二分类问题。具体操作为将类别概率分布矩阵和one-hot标签按行展开，那么就相当于一个二分类问题了。其实这和第一种方法从根本上来说是一致的，但是第一种相当于把每个类别单独取出计算ROC曲线，然后使用加权平均的方法得到总的ROC曲线。而该方法相当于总体看成二分类（通过将标签one-hot化），直接得到对应的ROC曲线。

这两个方法相比，在数据集正常的情况下两者的结果相近。但是在数据集类别不平衡的情况下，第一种方法相比第二种方法无法便无法体现出类别不平横情况下的AUC。

在python的sklearn中，第一种方法和第二中方法对应于sklearn.metrics.roc_auc_score函数中参数average值为macro和micro的情况。下面为这两种方法应用于cifar10数据集中的代码实现部分，选用的噪声为80%的均匀噪声，对应于文章附录中图3的绿线。

```python
1. one_hot_targets = label_binarize(all_targets.cpu().tolist(), classes=[i for i in range(args.num_class)])
2. for i in range(args.num_class):
3.     fpr[i], tpr[i], _ = roc_curve(one_hot_targets[:, i], all_outputs.cpu().numpy()[:, i])
4.     roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area (2)
5. fpr['micro'], tpr['micro'], _ = roc_curve(one_hot_targets.ravel(), all_outputs.cpu().numpy().ravel())
6. roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
# Compute macro-average ROC curve and ROC area (1)
7. all_fpr = np.unique(np.concatenate([fpr[i] for i in range(args.num_class)]))
8. mean_tpr = np.zeros_like(all_fpr)
9. for i in range(args.num_class):
10.     mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
11. mean_tpr /= args.num_class
12. fpr['macro'] = all_fpr
13. tpr['macro'] = mean_tpr
14. roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])
```

其中，all_targets为每个batch样本对应的真实标签，all_outputs为模型对batch样本的预测输出。第1行表示将样本标签one-hot化，2-4行为计算各类别的假阳性率、真阳性率及对应的AUC，5-6行为方法二对应的实现代码，7-14行对应于方法一。最终的结果如下：

![4](https://i.loli.net/2020/03/25/Txz8wqP3IfLKF6J.png)<center>图4. 含80%均匀噪声CIFAR10数据集训练过程测试集AUC随着epoch的变化曲线</center>

<center>
<figure>
<img src="https://i.loli.net/2020/03/25/Txz8wqP3IfLKF6J.png" />

<img src="https://i.loli.net/2020/03/25/GfShrnQ4JszLuD5.png" />

</figure>
</center>