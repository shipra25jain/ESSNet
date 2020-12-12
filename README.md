
# ESSNet - Embedding-based Scalable Segmentation Network
![GitHub Logo](Overview.png)


## Scaling Semantic Segmentation Beyond 1K Classes on a Single GPU


In our embedding-based scalable segmentation approach, we reduce the space complexity of the segmentation model's output from O(C) to O(1), propose an approximation method for ground-truth class probability, and use it to compute cross-entropy loss. The proposed approach is general and can be adopted by any state-of-the-art segmentation model to gracefully scale it for any number of semantic classes with only one GPU. Our approach achieves similar, and in some cases, even better mIoU for Cityscapes, Pascal VOC, ADE20k, COCO-Stuff10k datasets when adopted to DeeplabV3+ model with different backbones. We demonstrate a clear benefit of our approach on a dataset with 1284 classes, bootstrapped from LVIS and COCO annotations, with three times better mIoU than the DeeplabV3+ model.

**Instructions to use**

Clone our github repository
```
git clone https://github.com/shipra25jain/ESSNet.git
```
Create and activate conda environment
```
conda env create -f environment.yml
conda activate env
```
