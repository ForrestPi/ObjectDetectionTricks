## YOLO

https://github.com/flkraus/bayesian-yolov3
Uncertainty Estimation in One-Stage Object Detection
https://arxiv.org/abs/1905.10296
https://tongtianta.site/paper/77840



https://github.com/BadUncleBoy/No_Anchor_Yolo
1、目标
抛弃特征金字塔思想，在多个特征图上做预测
抛弃基于Anchor的目标检测算法
抛弃Two-Stage目标检测流程，只做One-Stage目标检测算法
2、技术难点
解决物体遮挡难题
提升对小物体定位的准确度


https://github.com/Realwhisky/Tricks-for-better-detection-performance-of-YOLO-v3
Attention+Focal

https://github.com/JKBox/YOLOv3-quadrangle

https://github.com/rayjan0114/yolov3-spp-tiny

https://github.com/dlyldxwl/Stronger-One-stage-detector-with-much-Tricks/blob/master/SSD/README.md


https://github.com/ruinmessi/ASFF
YOLO+ASFF


https://github.com/laycoding/YOLO_V3
 data augmentation(release)
 multi-scale training(release)
 Focal loss(increase 2 mAP, release)
 Single-Shot Object Detection with Enriched Semantics(incrase 1 mAP, not release)
 Soft-NMS(drop 0.5 mAP, release)
 Group Normalization(didn't use it in project, release)
 Deformable convolutional networks
 Scale-Aware Trident Networks for Object Detection
 Understanding the Effective Receptive Field in Deep Convolutional Neural Networks


https://github.com/jwchoi384/Gaussian_YOLOv3
http://openaccess.thecvf.com/content_ICCV_2019/html/Choi_Gaussian_YOLOv3_An_Accurate_and_Fast_Object_Detector_Using_Localization_ICCV_2019_paper.html
An Accurate and Fast Object Detector Using Localization Uncertainty for Autonomous Driving (ICCV, 2019)


https://github.com/wlguan/Stronger-yolo-pytorch
KLloss+ASFF


https://github.com/ShuiXianhua/YOLO_V3
Group Normalization
 Focal loss
 Soft-NMS(no useful)
 data augmentation
 multi-scale training
 Single-Shot Object Detection with Enriched Semantics
 SNIPER

https://github.com/FateScript/CenterNet-better


https://github.com/rayjan0114/YOLO_IEEE


https://github.com/xytpai/fcos
Some modifies are adopted:


https://github.com/yvan674/watershed-fcos
uses a watershed transform instead of the centerness layer in FCOS

https://github.com/sidml/Understanding-Centernet
https://github.com/FateScript/CenterNet-better



https://github.com/VectXmy/FCOS.Pytorch
https://github.com/feifeiwei/FCOS.pytorch
https://github.com/yjh0410/FCOS-LITE
https://github.com/neilctwu/FCOS-pytorch_Simplified

https://github.com/WJ1214/Trident-Grasp


https://github.com/VectXmy/FasterRCNN.Pytorch
https://github.com/VectXmy/LFFD.Pytorch



Remove center-ness branch for simplicity.
Add center sample mechanism to improve performance.
Predect yxhw instead of tlbr +0.3Map.
Note: GPU Compute Capability >= 6.1 (pytorch>=1.0.0)


