[TOC]

# 一、大赛简介

## 大赛简介

> **大赛名称：**安全AI挑战者计划第五期：伪造图像的篡改检测-赛道2
>
> **官方简介：** https://tianchi.aliyun.com/competition/entrance/531812/introduction  赛道2（10月12日10:00AM (UTC+8)开启，赛道1报名选手无需重复报名）伪造图像的对抗攻击比赛的赛道1——攻击比赛已经接近尾声，很多高质量的P图不但骗过人眼，还成功骗过我们提供的4个经典检测模型，那是否就真的是“魔高一丈”（反取证技术）呢？我们的对抗攻击比赛开始进入赛道2——检测比赛将在10月12日10:00AM (UTC+8)拉开帷幕！设计出“火眼金睛”（检测算法），把别人的“挖坑”（篡改区域）一一识别出来。区别于以往的图像取证比赛专注于自然内容图像，我们比赛采用的数据为大量伪造的证书文档类图像。任务是通过提供的训练集学习出有效的检测算法，对测试集的伪造图像进行篡改定位。为了更好的评价选手的检测定位效果，我们设计了全面的得分计算准则。



## 数据形式

- 数据包括训练集和测试集，训练集有1500张JPEG图像及对应mask（分辨率与原图保持一致，像素值0表示该像素标识为未篡改，像素值1表示该像素标识为篡改），JPEG图像的EXIF信息均被擦除，除部分无后处理外，其它可能经过裁边、平滑、下采样、社交工具传输（没有使用组合方式）；测试集有1500张JPEG图像，处理过程与训练集一致；允许使用集外数据进行训练学习。
- 参赛者提交数据时，利用我们提供的python程序生成mask，对1500张mask图像打包上传。
- 篡改图像可能包括如splicing、copy-move、object removal等任意操作，部分进行后处理（JPEG压缩、重采样、裁剪边缘等）。
- 不需要考虑图像的元数据（已经被擦除）。

+ **示例**
  ![](https://tva1.sinaimg.cn/large/0081Kckwgy1gl1hbblod0j30iu083acy.jpg)



## 评估标准

![](https://tva1.sinaimg.cn/large/0081Kckwly1gl1h4m89brj30hb0gumy6.jpg)



# 二、解决方案

## 运行环境

### 1  NoteBook

+ Colab Pro
  + GPU   Tesla v100
  
  + RAM   25G
  
  + tensorflow ==2.1.0
  
  + keras == 2.3.0
  
    
  

### 2  .py

1. 安装依赖库

   `pip3 install -r requirements.txt `

2. 训练

   `$ cd code`
   `$ ./trian.sh`

3. 测试

   `$ cd code`/
   `$ ./run.sh`

   

## 数据处理

+ 将图像数据准换为 np 形式的正方形矩阵，其中
  + **trian**             resize(512, 512, 3)
  + **train_mask** resize(512, 512, 1)
  + **test**               resize(512, 512, 3)

1024\*1024，1555张图大概占用 80-90 G内存， 512\*512 (15 G+)在 Colab 上基本上达到了极限，256\*256 尽管占用小了很多，但是模型泛化不好，最终还是采用了 512\*512。



## 模型训练

Segmentation model：采用 `Unet + efficientnetb3`

Repo: https://github.com/qubvel/segmentation_models

Backbone:

| Type         | Names                                                        |
| ------------ | ------------------------------------------------------------ |
| VGG          | 'vgg16' 'vgg19'                                              |
| ResNet       | 'resnet18' 'resnet34' 'resnet50' 'resnet101' 'resnet152'     |
| SE-ResNet    | 'seresnet18' 'seresnet34' 'seresnet50' 'seresnet101' 'seresnet152' |
| ResNeXt      | 'resnext50' 'resnext101'                                     |
| SE-ResNeXt   | 'seresnext50' 'seresnext101'                                 |
| SENet154     | 'senet154'                                                   |
| DenseNet     | 'densenet121' 'densenet169' 'densenet201'                    |
| Inception    | 'inceptionv3' 'inceptionresnetv2'                            |
| MobileNet    | 'mobilenet' 'mobilenetv2'                                    |
| EfficientNet | 'efficientnetb0' 'efficientnetb1' 'efficientnetb2' 'efficientnetb3' 'efficientnetb4' 'efficientnetb5' efficientnetb6' efficientnetb7' |



## 模型预测

预测测试集的图像时，如果不 resize 测试集的图像，直接报错。所以我将 test 转化为 512\*512\*3 的图像，再经过模型篡改区域预测后得到 512*512\*1的 mask，再将 mask resize 到图像本身的长宽。这样应该会影响预测效果吧！！！





# 三、总结

公榜排名：31/1470

观察了测试集的预测效果，发现模型对没有篡改的图片识别效果很差。对未篡改图像进行了扩增，但是 Colab 内存太小，多加 200 张就爆内存。在距离比赛前一天，在 mistgpu 上租了一个 128G  的服务器，但时间不够了'^'。



如果采用一些图像增强，模型的泛化能力应该会有所提升。在训练集中，未篡改的图像可以通过 mask 来判断。

```python
for i,img in tqdm(zip(list({i.split('.')[0] for i in os.listdir(TRAIN_PATH)}), range(0,len(os.listdir(TRAIN_PATH))))):
  
    image= cv2.imread(train_data[img])
    mask = cv2.imread(mask_data[img],0)

    if cv2.countNonZero(mask) == 0:
        cv2.imwrite(path+'train_ok/{}.jpg'.format(i), image)
        cv2.imwrite(path+'train_ok_mask/{}.png'.format(i), mask)
    else: pass    
print(len(path+ 'train_ok/'), len(path+ 'train_ok_mask/'))
```



+ 训练集中的未篡改图像扩充；
+ 把测试集中的一些明显未篡改的图像，打上全黑的 mask，加入到训练集。



  

  

