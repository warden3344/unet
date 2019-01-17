Implementation of deep learning framework -- Unet, using caffe

The architecture was inspired by U-Net: Convolutional Networks for Biomedical Image Segmentation.and the repository:https://github.com/zhixuhao/unet which is impelement the unet by Keras 

Data
The original dataset is from isbi challenge, and the dataset on this repository is the same as the data in repository:https://github.com/zhixuhao/unet 

How to train:
1.change the directory in the imglist.txt and the masklist.txt
2.change the directory and the path in the mydatalayer.py 
3.change the path in the solver.prototxt
4.change the path in the train.sh
5.run the train.sh
after serveral epochs the loss will be lower the 0.01.

