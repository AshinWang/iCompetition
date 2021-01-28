# [2021广东工业智造创新大赛—智能算法赛](https://tianchi.aliyun.com/competition/entrance/531846/information)

## 比赛简介

大赛深入到佛山瓷砖知名企业，在产线上架设专业拍摄设备，实地采集生产过程真实数据，解决企业真实的痛点需求。大赛数据覆盖到了瓷砖产线所有常见瑕疵，包括粉团、角裂、滴釉、断墨、滴墨、B孔、落脏、边裂、缺角、砖渣、白边等。实拍图示例如下：

![img](./img/1.jpg)

## 数据集

**[Train](https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531846/tile_round1_train_20201231.zip)**	**[TestA](https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531846/tile_round1_testA_20201231.zip)**	[**TestB**](https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531846/tile_round1_testB_20210128_2.zip)



## 非官方 Baseline

+ PaddlePaddle  [Baseline](https://aistudio.baidu.com/aistudio/projectdetail/1420200?shared=1) 64+
+ 切图 + yolov5 [Baseline](https://github.com/DLLXW/data-science-competition/tree/main/%E5%A4%A9%E6%B1%A0/2021%E5%B9%BF%E4%B8%9C%E5%B7%A5%E4%B8%9A%E6%99%BA%E9%80%A0%E5%88%9B%E6%96%B0%E5%A4%A7%E8%B5%9B) 55+ 



## EfficientDet Baseline

基于上面两个 Baseline 的数据处理方法，采用谷歌官方 [EfficientDet](https://github.com/google/automl/tree/master/efficientdet) 和 [Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) 框架搭建了训练模型。官方的框架需要将数据集转换成 TFRecord 格式，非常耗时，于是转向基于 Pytorch 版的 EfficientDet。

采用  [Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)  D3 模型，在 Colab(Tesla P100 16GiB) 和共享云 GPU(2080Ti) 平台训练6 个 epoch(Step 30000)，其中

+ Resize 896 (D3 默认尺寸) 17+ 
+ 切图 640 22+ 

这种瑕疵检测用大尺寸训练应该能提分。