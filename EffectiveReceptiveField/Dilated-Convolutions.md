Below you can find a short mathematical explanation on dilated convolutions and how they differ from a standard convolution. If you need a a more detailed illustration you can look for example in [1].

## Standard Convolution
The discrete standard convolution can be described by the following equation:

<p align="center">
  <img src="https://github.com/mcFloskel/houseNet/blob/master/images/eq_convolution.png" width=300/>
</p>

where _F_ is filter map and _k_ a convolutional filter.

## Dilated Convolution
A _l_-dilated convolution is a convolution which dilates the filter _k_ by a factor _l_:

<p align="center">
  <img src="https://github.com/mcFloskel/houseNet/blob/master/images/eq_dilated_convolution.png" width=300/>
</p>

The equation shows that the filter weights stay the same but are applied to different elements of the feature map in comparison to a standard convolution. The elements are chosen such that they have free space with a distance of _l_ between them. However if _l=1_, the dilated convolution resembles the standard convolution. The following figure illustrates how a _2_-dilated _3x3_ convolution looks like in a _2D_ space:

<p align="center">
  <img src="https://github.com/mcFloskel/houseNet/blob/master/images/2_dilated_convolution.png" width=300/>
</p>

The filter has still the same amount of parameters but the values are applied at different positions in the feature map. This results in an increased receptive field without the drawback of a higher parameter size (bigger filter) or lower output resolution (pooling).

A more detailed comparison between different methods for increasing the perceptive field can be found [here](https://github.com/mcFloskel/houseNet/wiki/Methods-for-increasing-the-field-of-view).

### References:
[1] Fisher Yu, Vladlen Koltun **Multi-Scale Context Aggregation by Dilated Convolutions** ([arXiv](https://arxiv.org/abs/1511.07122))