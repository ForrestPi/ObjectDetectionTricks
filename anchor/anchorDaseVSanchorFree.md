## 目标检测中anchor的优缺点
目标检测中自从RPN 被提出来以后，anchor-based 的方法便成了物体检测模型的主流。这种方法在特征图的每一个像素点预设几个不同尺度和纵横比的 bounding box，称之为 anchor。之后对每一个 anchor 进行分类，并对正类的 anchor 进行回归（位置及大小调整）。之后的 SSD，YOLO 等 single-stage 方法也是基于 anchor 进行检测。
anchor 的引入带来了很多优点：
1）很大程度上减少了计算量，并将 proposal 数量放到可控范围内以便后面的计算和筛选。
2）通过调整不同的 anchor 设置覆盖尽可能多的物体，也可针对不同任务设置不同的 anchor 尺度范围。
3）由于 anchor 的尺度是人工定义的，物体的定位是通过 anchor 的回归来实现，仅仅计算偏移量而不是物体的位置大大降低了优化难度。

anchor 的设置也有着它自身的缺点:
1）在 Faster RCNN 以后的很多论文中我们不难发现，单纯通过使用更多不同大小和纵横比的 anchor 以及更多的训练技巧就可以达到更好的效果，然而这种通过增加算力而改进网络的方法很难落实到实际的应用中（模型速度和大小）。
2）anchor 的设定需要人为设定大量的参数，且离散的 anchor 尺度设定会导致一些物体无法很好的匹配到 anchor，从而导致遗漏。


anchor-base存在的问题：

•与锚点框相关超参 (scale、aspect ratio、IoU Threshold) 会较明显的影响最终预测效果；

•预置的锚点大小、比例在检测差异较大物体时不够灵活；

•大量的锚点会导致运算复杂度增大，产生的参数较多；

•容易导致训练时negative与positive的比例失衡。

Anchor-free算法的优点：

•使用类似分割的思想来解决目标检测问题；

•不需要调优与anchor相关的超参数；

•避免大量计算GT boxes和anchor boxes 之间的IoU，使得训练过程占用内存更低。

#### 语义模糊性，即两个物体的中心点落在了同一个网格中。



我的看法是anchor-free和anchor-based实际上最大的区别应该是解空间上的区别。anchor-free，无论是keypoint-based的方法（e.g. CornerNet和CenterNet）还是pixel-wise prediction的方法（e.g. FCOS），本质上都是dense prediction的方法，庞大的解空间使得简单的anchor-free的方法容易得到过多的false positive，而获得高recall但是低precision的检测结果；

anchor-based由于加入了人为先验分布，同时在训练的时候prediction（尤其是regression）的值域变化范围实际上是比较小的，这就使得anchor-based的网络更加容易训练也更加稳定；

目前的anchor-free方法，一方面通过了各种方式来进一步re-weight检测结果的质量（e.g. FCOS的centerness），另一方面通过FPN也在一定程度上缓解了高度重合带来的影响（当然还有很多不同的筛选方法这里不多赘述）所以说anchor-free的优点在于其更大更灵活的解空间、摆脱了使用anchor而带来计算量从而让检测和分割都进一步走向实时高精度；

缺点就是其检测结果不稳定，需要设计更多的方法来进行re-weight至于说正负样本极端不平衡，实际上anchor-based方法也存在，讨论度没有那么高主要是因为都是用了模型中使用了各种方式进行了筛选，包括two-stage的RPN和one-stage的Focal loss

首先对于每个预选框我们都要根据不同的任务去设置其参数，如长宽比，尺度大小，以及anchor的数量，这就造成了不同参数所导致的AP有很大的不同，同时调参耗时耗力。产生的预选框在训练阶段要和真实框进行IoU的计算，这会占用很大的运行内存空间和时间。对于单阶段算法来说，为了使其更为有效的检测小目标，通常进行FPN结构以及使用更低的特征图，这样来说产生预选框的数量就会增加很多很多。针对不同的任务，比如人脸识别和通用物体检测，所有的参数都需要重新调节，这样一个模型的迁移能力就体现不出来了。

## anchor-free类算法归纳：
### A.基于多关键点联合表达的方法
* CornerNet/CornerNet-lite：左上角点+右下角点
* ExtremeNet：上下左右4个极值点+中心点
* CenterNet:Keypoint Triplets for Object Detection：左上角点+右下角点+中心点
* RepPoints：9个学习到的自适应跳动的采样点
* FoveaBox：中心点+左上角点+右下角点f.PLN：4个角点+中心点
### B.基于单中心点预测的方法
* CenterNet:Objects as Points：中心点+宽度+高度
* CSP：中心点+高度（作者预设了目标宽高比固定，根据高度计算出宽度）
* FCOS：中心点+到框的2个距离

