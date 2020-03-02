本文首先将介绍在目标跟踪任务中常用的匈牙利算法（Hungarian Algorithm）和卡尔曼滤波（Kalman Filter），然后介绍经典算法DeepSORT的工作流程以及对相关源码进行解析。

目前主流的目标跟踪算法都是基于Tracking-by-Detecton策略，即基于目标检测的结果来进行目标跟踪。DeepSORT运用的就是这个策略，上面的视频是DeepSORT对人群进行跟踪的结果，每个bbox左上角的数字是用来标识某个人的唯一ID号。

这里就有个问题，视频中不同时刻的同一个人，位置发生了变化，那么是如何关联上的呢？答案就是匈牙利算法和卡尔曼滤波。

* 匈牙利算法可以告诉我们当前帧的某个目标，是否与前一帧的某个目标相同。
* 卡尔曼滤波可以基于目标前一时刻的位置，来预测当前时刻的位置，并且可以比传感器（在目标跟踪中即目标检测器，比如Yolo等）更准确的估计目标的位置。

匈牙利算法（Hungarian Algorithm）

首先，先介绍一下什么是分配问题（Assignment Problem）：假设有N个人和N个任务，每个任务可以任意分配给不同的人，已知每个人完成每个任务要花费的代价不尽相同，那么如何分配可以使得总的代价最小。

举个例子，假设现在有3个任务，要分别分配给3个人，每个人完成各个任务所需代价矩阵（cost matrix）如下所示（这个代价可以是金钱、时间等等）：

怎样才能找到一个最优分配，使得完成所有任务花费的代价最小呢？

匈牙利算法（又叫KM算法）就是用来解决分配问题的一种方法，它基于定理：

    如果代价矩阵的某一行或某一列同时加上或减去某个数，则这个新的代价矩阵的最优分配仍然是原代价矩阵的最优分配。

算法步骤（假设矩阵为NxN方阵）：

    对于矩阵的每一行，减去其中最小的元素对于矩阵的每一列，减去其中最小的元素用最少的水平线或垂直线覆盖矩阵中所有的0如果线的数量等于N，则找到了最优分配，算法结束，否则进入步骤5找到没有被任何线覆盖的最小元素，每个没被线覆盖的行减去这个元素，每个被线覆盖的列加上这个元素，返回步骤3

继续拿上面的例子做演示：

step1 每一行最小的元素分别为15、20、20，减去得到：

step2 每一列最小的元素分别为0、20、5，减去得到：

step3 用最少的水平线或垂直线覆盖所有的0，得到：

step4 线的数量为2，小于3，进入下一步；

step5 现在没被覆盖的最小元素是5，没被覆盖的行（第一和第二行）减去5，得到：

被覆盖的列（第一列）加上5，得到：

跳转到step3，用最少的水平线或垂直线覆盖所有的0，得到：

step4：线的数量为3，满足条件，算法结束。显然，将任务2分配给第1个人、任务1分配给第2个人、任务3分配给第3个人时，总的代价最小（0+0+0=0）：

所以原矩阵的最小总代价为（40+20+25=85）：

sklearn里的linear_assignment()函数以及scipy里的linear_sum_assignment()函数都实现了匈牙利算法，两者的返回值的形式不同：

```python
import numpy as np 
from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment
 

cost_matrix = np.array([
    [15,40,45],
    [20,60,35],
    [20,40,25]
])
 
matches = linear_assignment(cost_matrix)
print('sklearn API result:\n', matches)
matches = linear_sum_assignment(cost_matrix)
print('scipy API result:\n', matches)
 

"""Outputs
sklearn API result:
 [[0 1]
  [1 0]
  [2 2]]
scipy API result:
 (array([0, 1, 2], dtype=int64), array([1, 0, 2], dtype=int64))
"""


在DeepSORT中，匈牙利算法用来将前一帧中的跟踪框tracks与当前帧中的检测框detections进行关联，通过外观信息（appearance information）和马氏距离（Mahalanobis distance），或者IOU来计算代价矩阵。

```python
#  linear_assignment.py
def min_cost_matching(distance_metric, max_distance, tracks, detections, 
                      track_indices=None, detection_indices=None):
    ...
    
    # 计算代价矩阵
    cost_matrix = distance_metric(tracks, detections, track_indices, detection_indices)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    
    # 执行匈牙利算法，得到匹配成功的索引对，行索引为tracks的索引，列索引为detections的索引
    row_indices, col_indices = linear_assignment(cost_matrix)
 
    matches, unmatched_tracks, unmatched_detections = [], [], []
 
    # 找出未匹配的detections
    for col, detection_idx in enumerate(detection_indices):
        if col not in col_indices:
            unmatched_detections.append(detection_idx)
     
    # 找出未匹配的tracks
    for row, track_idx in enumerate(track_indices):
        if row not in row_indices:
            unmatched_tracks.append(track_idx)
    
    # 遍历匹配的(track, detection)索引对
    for row, col in zip(row_indices, col_indices):
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        # 如果相应的cost大于阈值max_distance，也视为未匹配成功
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
 
    return matches, unmatched_tracks, unmatched_detections
```

## 卡尔曼滤波（Kalman Filter）

卡尔曼滤波被广泛应用于无人机、自动驾驶、卫星导航等领域，简单来说，其作用就是基于传感器的测量值来更新预测值，以达到更精确的估计。

假设我们要跟踪小车的位置变化，如下图所示，蓝色的分布是卡尔曼滤波预测值，棕色的分布是传感器的测量值，灰色的分布就是预测值基于测量值更新后的最优估计。

在目标跟踪中，需要估计track的以下两个状态：

    均值(Mean)：表示目标的位置信息，由bbox的中心坐标 (cx, cy)，宽高比r，高h，以及各自的速度变化值组成，由8维向量表示为 x = [cx, cy, r, h, vx, vy, vr, vh]，各个速度值初始化为0。协方差(Covariance )：表示目标位置信息的不确定性，由8x8的对角矩阵表示，矩阵中数字越大则表明不确定性越大，可以以任意值初始化。

卡尔曼滤波分为两个阶段：(1) 预测track在下一时刻的位置，(2) 基于detection来更新预测的位置。

下面将介绍这两个阶段用到的计算公式。（这里不涉及公式的原理推导，因为我也不清楚原理(ಥ_ಥ) ，只是说明一下各个公式的作用）

预测

    基于track在t-1时刻的状态来预测其在t时刻的状态。 

[公式]

[公式]

在公式1中，x为track在t-1时刻的均值，F称为状态转移矩阵，该公式预测t时刻的x'：

矩阵F中的dt是当前帧和前一帧之间的差，将等号右边的矩阵乘法展开，可以得到cx'=cx+dt*vx，cy'=cy+dt*vy...，所以这里的卡尔曼滤波是一个匀速模型（Constant Velocity Model）。

在公式2中，P为track在t-1时刻的协方差，Q为系统的噪声矩阵，代表整个系统的可靠程度，一般初始化为很小的值，该公式预测t时刻的P'。

源码解读：

```python
#  kalman_filter.py
def predict(self, mean, covariance):
    """Run Kalman filter prediction step.
    
    Parameters
    ----------
    mean: ndarray, the 8 dimensional mean vector of the object state at the previous time step.
    covariance: ndarray, the 8x8 dimensional covariance matrix of the object state at the previous time step.
 
    Returns
    -------
    (ndarray, ndarray), the mean vector and covariance matrix of the predicted state. 
     Unobserved velocities are initialized to 0 mean.
    """
    std_pos = [
        self._std_weight_position * mean[3],
        self._std_weight_position * mean[3],
        1e-2,
        self._std_weight_position * mean[3]]
    std_vel = [
        self._std_weight_velocity * mean[3],
        self._std_weight_velocity * mean[3],
        1e-5,
        self._std_weight_velocity * mean[3]]
    
    motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))  # 初始化噪声矩阵Q
    mean = np.dot(self._motion_mat, mean)  # x' = Fx
    covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov  # P' = FPF(T) + Q
 
    return mean, covariance
```