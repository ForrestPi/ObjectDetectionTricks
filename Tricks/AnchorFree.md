Anchor-free or Not?

最近一段时间里，出现了很多针对目标检测中的 anchor 方法的反思，有些是抛弃，有些是改进。
比如：

    CornerNet -> CenterNet -> ExtremeNet 通过像素分类（Segmentation）预测关键点的方法来组织一个bboxFCOS，FoveaBox 这一类直接回归边界，这个在 YOLOv1 还有更早之前的各种 One-stage 方法上都被广泛使用，也是 One-stage 的传统方法Guided Anchor 这种尝试自适应的学习 anchor，类似的，虽然是 anchor-free 但是也是希望更加平滑的根据目标特点来分配尺度： Feature Selective Anchor-Free Module for Single-Shot Object Detection

1. Anchor的意义

从自己的工作经验上来回答 anchor 的好处的话，那么就是下面的两点：

    容易收敛，降低训练难度容易对应目标大小与对应feature map

第一点

不使用 anchor 的情况，boundingbox 回归的范围比较大。只要是不使用 anchor 直接回归 bbox 的方法，都绕不开这个问题。
这个问题会带来训练难度的增加和网络偏好的倾斜：

    bbox 回归范围大，会让计算的 loss 范围变化很大，而且对大小目标的精度有不同的偏好。

前者会破坏训练的稳定，后者会让目标预测不准确。

如果看回归方法 anchor-free 的论文，都会提到如何缓解这个问题（不论是几年前的，还是现在新出的），一般的套路：

    使用相对值，而非像素值Log、Sqrt这类非线性的转换IoU Loss 一类，让偏好更加平衡

使用 anchor 是另外一种缓解这个问题的方法：

anchor 给了回归的一个基点，不论大目标、小目标，相对 anchor 的变换范围都是接近的，这就把回归值范围过大的问题解决掉了。

第二点

在没有FPN的时代，或者由于性能问题放弃FPN结构的时候，大小目标都是依赖少数几个level 的 featuremap 来预测的。

如果不对大小做区分，会造成一个结果：

    不同 level 的 featuremap 负责预测的目标大小是不太可控的。

这样学习过程中不是显式的区分不同的特征，而是希望网络权值能够适应性的学习到不同大小目标的特征。这个路子在理论上当然没有问题，但是实际训练中，增加了训练难度，也可能是强“网”所难。

有了 anchor，就可以把不同尺寸的目标分配到对应的 anchor 上，这样对应 anchor 的网络权值就仅仅负责一个比较小的目标尺寸范围，同时学习过程中，是显式的通过不同的分支、路径来传递不同尺寸目标的 loss，让网络更加符合“逻辑”的得到训练。

以上这两点，就是我们之前的检测工作加入 anchor 方法的原因。
2. Anchor可能的两个问题

第一，Anchor的分配是 0-1 的硬指标

目标分配在哪个 anchor 上是很粗暴的（谁IoU大就归谁），而且是一个0-1的选择。另外 anchor 本身的尺寸、比例是人工指定的，虽然和不同level的featuremap有一定的关系，但并没有和网络结构的设计有良好的耦合（虽然 anchor 可以通过数据集的聚类来获得，但是这也只是单方面的与数据集有了耦合）。

实际训练中，可能同一个目标适合两个 anchor ，那么在 loss 的设计上就需要考虑孰轻孰重的问题。对应的两个 anchor 计算的 loss，是不是可以用一种更加平滑的方法来加权，而非简单粗暴的0-1选择？

Anchor 本身的设计，是不是可以通过 网络结构+数据分布 两个因素在训练中也可以不断更新变化？

第二，Anchor的非平滑

Anchor 可以理解为针对目标大小的离散划分，将不同大小的目标放入到不同的桶里面。而且，这些桶是先验+固定的。

这带来的问题是：

    anchor 打断了目标大小预测上的平滑。

从训练的角度来看，都希望能够有一个更加平滑的预测，从经验上看，可以增加网络的泛化性能。

当然，即使现在的 anchor-free 的方法，也由于不同的 featuremap level，再另外一个维度上带来了类似的打断问题：

    很多 anchor-free 的方法在 featuremap level 的选择也使用人工先验。

FSAF这篇论文给了一个根据loss大小选择不同featuremap level的方法，或许未来会有更加平滑、模糊的 featuremap level 的分配方法。
3. Anchor-free的前提条件

Anchor-free变得具有竞争性的原因在我看看来就是一个：

    FPN 或者类似网络结构带来的多尺度的 feature map 让 anchor-free 重新具备了竞争力

在这样的前提下，是否需要anchor变得相对不重要，不过即使目前的 anchor-free 也需要解决没有 anchor 带来的回归范围变化太大的问题，怎么设计 loss 和控制预测值的范围？

以及，怎样更加合理的分配目标大小与 featuremap 对应的问题（事实上就是卷积网络的 receptive field 与数据的对应）。