"""Fast R-CNN (Girshick, ICCV 2015; arXiv:1504.08083) re-implemented in PyTorch.

Architecture follows rbgirshick/fast-rcnn VGG16 model: conv5_3 feature map (stride 16),
RoIPool 7x7, fc6/fc7 (4096, dropout), softmax classifier + class-specific box regression,
multi-task loss (CE + smooth-L1). Region proposals come from selective search.
"""
