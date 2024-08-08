# YOLOv5-transformer-CBMA
是第一次对yolo的学习和尝试

使用了yolov5的源码，并在源码上做了一点修改，加入了transformer和CBMA模块，增强了模型的鲁棒性。

该模型训练使用了 VOC2012的数据集（数据集并没有在仓库中，还需自行下载）

以下是模型训练和测试的效果：



![ious](ious.png)

![precision](precision.png)

![train_loss](train_loss.png)



![bird1](.\img\bird1.jpg)

![bottle1](.\img\bottle1.jpg)

![bycycle2](.\img\bycycle2.jpg)

![car3](.\img\car3.jpg)

![horse2](.\img\horse2.jpg)

![pottedplant1](.\img\pottedplant1.jpg)

![street2](.\img\street2.jpg)



![detect_bird1](.\img_output\bird1.jpg)

![detect_bottle1](.\img_output\bottle1.jpg)

![detect_bycycle2](.\img_output\bycycle2.jpg)

![detect_car3](.\img_output\car3.jpg)

![detect_horse2](.\img_output\horse2.jpg)

![detect_pottedplant1](.\img_output\pottedplant1.jpg)

![detect_street2](.\img_output\street2.jpg)
