# SelfAlign

## 关于snr定义以及如何添加噪声

### snr对应关系表

这个表格参数用于之后的添加噪声部分设置参数

| SNR100 | SNR0.1 | SNR0.05 | SNR0.03 | SNR0.001 |
| :----: | :----: | :-----: | :-----: | :------: |
|  5LQW  |  4.4   |   2.1   |  1.28   |  0.415   |
|  5MPA  |   80   |   40    |   248   |    8     |
|  5T2C  |  12.5  |    6    |   3.7   |   1.25   |
|  6A5L  |   10   |    5    |    3    |    1     |

由于GumNet中的snr计算方式不是通常意义上的信号功率比，同时原代码找不到添加噪声部分，所以按照论文补充材料所给出的计算公式近似计算信噪比。
GumNet原论文地址：https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7955792/
GumNet补充材料关于snr的定义如下图所示：
![image](https://github.com/IAMCHF/SelfAlign/assets/56794463/9d46a502-6937-4783-9cb9-4d634d0febdf)

可以用SelfAlign/preprocessing/snr.py中注释部分代码correlation_coefficient计算c，然后输出snr。

```
例子：

a属于"5LQW, 5MPA, 5T2C, 6A5L"其中一个。

b属于"snr01, snr005, snr003, snr001"其中一个。

利用snr.py中注释掉的代码部分可以查看/class_32/normalized/$b/$a/$a.mrc 与 /class_32/normalized/snr100/$a/$a.mrc 的snr。
```

***由于噪声是固定方差的随机高斯噪声，所以计算出来的snr有一定随机性，但是snr基本是符合表格或者低于表格所给的。***



### 添加噪声的具体步骤

```
1. snr中add_gaussian_noise用于添加噪声。添加噪声的过程为先对数据使用normalize_z_score_foler.py进行整体归一化处理，然后根据snr对应关系的表格对每个样本添加噪声。
2. 添加完噪声后再使用normalize_z_score_foler.py进行整体归一化处理。
3. 处理得到的数据用于最终的模型训练。
```

