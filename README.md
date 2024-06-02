[toc]

# SelfAlign-English

《SelfAlign: Achieving Subtomogram Alignment with Self-Supervised Deep Learning》



## Usage

### requirements

```
ubuntu20.04 or ubuntu22.04
tensorflow-2.12.0 python3.8-3.11 GCC 9.3.1 cuda11.8 cudnn8.6
```

You can create a conda environment by using the following command:

```
conda create -n selfalign python=3.8
pip install tensorflow==2.12.0
pip install -r requirements.txt
```

requirements.txt is located in the SelfAlign folder

Note: After creating the Conda environment, it is necessary to test whether TensorFlow can detect the GPU

Use the following command to detect

```
python
import tensorflow as tf
tf.config.list_physical_devices('GPU')
```

### Specific steps

```
conda activate selfalign
```

#### 1、data preparation

```
python ./SelfAlign/bin/selfalign.py prepare_subtomo_star \
            ./normalized/snr01/6A5L \
            --output_star 6A5L.star
```

This command is used to generate an .star file, an example of which is as follows:

```
data_

loop_
_rlnSubtomoIndex #1 
_rlnImageName #2 
1 	/newdata3/chf/normalized/snr001/5T2C/5T2C.mrc
```

You can also manually modify the .star file.

After generating the corresponding .star file, the dataset for training the model and testing can be officially prepared. Taking 5LQW as an example:

##### (1)prepare test data

First, modify the data_dir and subtomo_star paths in SelfAlign/reprocessing/prepareutest5LQW.py to correspond to the paths on your own computer

Among them, data_dir is the path specified by oneself. It is recommended to only modify the content of "/newdata3/chf/test_data_rotation_strategy"+"/snr001" in data_dir="/newdata3/chf/", and fill in other parts according to the actual signal-to-noise ratio and comments in the code; subtomo_star is the location of the previously generated 5lqw.star file.

After modification, execute the command on the terminal: python SelfAlign/reprocessing/prepareutest5LQW.py

##### (2)prepare train data

Similar to prepare test data, then execute: python SelfAlign/reprocessing/prepare_train5LQW.py

##### (3)apply wedge

First, modify the mrc path in the sixth line of the SelfAlign/reprocessing/5lqw_apply_widge.py file to your own path

```
with mrcfile.open("/HBV/Caohaofan/selfalign/mask_wedge_32.mrc", permissive=True) as mrc:
    mask_wedge_32 = mrc.data.astype(np.float32)
```

mask_widge_32.mrc has been placed in the SelfAlign folder, but it is not set as fixed because the corresponding mask_wedge may be different for different electron microscopy data and needs to be modified according to the actual situation.

Afterwards, it is necessary to modify the *src_folder* and *dst_folder*. *src_folder* is the *rota* address in prepareutest5LQW.py and prepare_train5LQW.py, while *dst_folder* is a custom address.

It is necessary to add a wedge to both training and testing data, but in reality, alignment accuracy is higher without adding a wedge. We add a wedge to simulate the worst-case scenario, and theoretically, adding a wedge in other ways would achieve higher accuracy with self alignment.

##### (4)normalize

Firstly, modify the *rota* path in normalized_z_score_5lqw.py based on the previously prepared data. Then execute: python normalize_z_score_5lqw.py

#### 2、train model

Modify the "*result_dir, src_file, predict_txt_path*" in SelfAlign/models/gunet/train_5LQW.py

The test data does not participate in training throughout the entire process, but is only used to save experimental records. Therefore, you can choose to comment out predict_txt_path and correspondingly comment out 240 lines of code *predict (i)*.

```
result_dir:Save the address of the training model
src_file:Record the txt file address of the training data location, which is located in 'data_dir' in the (2) prepare train data.
predict_txt_path:Record the txt file address of the test data location, which is located in 'data_dir' in the (1) prepare test data.
```

#### 3、eval

Calculate evaluation metrics: Modify the *y_true* and *y_pred* addresses in SelfAlign/reprocessing/eval. py, and then execute the command:

`python SelfAlign/preprocessing/eval.py`

## About the definition of SNR and how to add noise

The baseline method compared in this article is GumNet

Due to the fact that the snr calculation method in GumNet is not the typical signal power ratio, and the original code cannot find the part that adds noise, the signal-to-noise ratio is approximately calculated according to the calculation formula provided in the supplementary materials of the paper.
GumNet original paper address:https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7955792/
The definition of snr in the supplementary materials of GumNet is shown in the following figure:
![image](https://github.com/IAMCHF/SelfAlign/assets/56794463/9d46a502-6937-4783-9cb9-4d634d0febdf)

You can use the annotated code in SelfAlign/reprocessing/snr. py to calculate *c* and then output *snr*.

### Specific steps for adding noise

```
1. add_gaussian_noise.py is used to add noise. The process of adding noise is to first perform overall normalization on the data using normalized_z_score_foler. py, and then add noise to each sample based on SNR.
2. After adding noise, use normalizer_z_score_foler.py for overall normalization.
3. The processed data is used in the data preparation stage.
```

## About Loss

View the specific implementation of *selfalign_loss* in SelfAlign/models/gumnet/tf_util_loss.py

## Discussions

(1)SelfAlign provides a feasible approach for subtomo alignment by constructing a self supervised learning task, which can serve as the fundamental framework for using deep learning to achieve Subtomogram Alignment in the future. At the same time, SelfAlign is preparing to refer to the traditional field of multiple alignments and gradually refine the idea to construct a complete implementation of the  subtomogram averaging (STA). Currently, the average. py and refine. py in SelfAlign/bin are being modified and tested, and it is expected to achieve a complete STA in the future. Stay tuned……

(2) there are some examples of module tests in selfalign

[IAMCHF/test_for_selfalign at master (github.com)](https://github.com/IAMCHF/test_for_selfalign/tree/master)

# SelfAlign-中文

《SelfAlign: Achieving Subtomogram Alignment with Self-Supervised Deep Learning》

## 使用方法

### 条件

```
ubuntu20.04 or ubuntu22.04
tensorflow-2.12.0 python3.8-3.11 GCC 9.3.1 cuda11.8 cudnn8.6
```

可以通过下面命令来创建一个conda环境：

```
conda create -n selfalign python=3.8
pip install tensorflow==2.12.0
pip install -r requirements.txt
```

requirements.txt 在SelfAlign文件夹下

注意：创建conda环境后还需要测试tensorflow是否能检测到gpu

使用下面命令检测

```
python
import tensorflow as tf
tf.config.list_physical_devices('GPU')
```

### 具体步骤

```
conda activate selfalign
```

#### 1、data preparation

```
python ./SelfAlign/bin/selfalign.py prepare_subtomo_star \
            ./normalized/snr01/6A5L \
            --output_star 6A5L.star
```

这段命令用于生成.star文件，一个.star文件示例如下：

```
data_

loop_
_rlnSubtomoIndex #1 
_rlnImageName #2 
1 	/newdata3/chf/normalized/snr001/5T2C/5T2C.mrc
```

也可以手动修改.star文件。

生成相应的.star文件后就可以正式准备用于训练模型和用于测试的数据集了，以5LQW举例：

##### (1)prepare test data

首先修改 SelfAlign/preprocessing/prepare_test5LQW.py 中 data_dir 和 subtomo_star 路径为自己电脑上对应的路径

其中data_dir为自己指定的路径，建议仅修改 data_dir = "/newdata3/chf/test_data_rotation_strategy" + "/snr001" 中 "/newdata3/chf/" 这部分内容，其他部分按照实际信噪比和代码中注释部分填入；subtomo_star为之前生成的 5lqw.star 文件的位置。

修改后后在终端执行命令：python SelfAlign/preprocessing/prepare_test5LQW.py

##### (2)prepare train data

与prepare test data类似，然后执行：python SelfAlign/preprocessing/prepare_train5LQW.py

##### (3)apply wedge

首先修改 SelfAlign/preprocessing/5lqw_apply_wedge.py 文件中 第6行代码中mrc路径为自己的路径

```
with mrcfile.open("/HBV/Caohaofan/selfalign/mask_wedge_32.mrc", permissive=True) as mrc:
    mask_wedge_32 = mrc.data.astype(np.float32)
```

mask_wedge_32.mrc 已经放在 SelfAlign 文件夹中，之所以不设置为固定的，是因为对于不同的电镜数据，对应的 mask_wedge 可能是不同的，需要根据实际情况修改。

之后还需要修改 src_folder 和 dst_folder。src_folder 就是 prepare_test5LQW.py 和 prepare_train5LQW.py 中的 rota 地址，dst_folder为自定义的地址。

需要对训练数据和测试数据都添加wedge，实际上不添加wedge的对齐精度更高。我们添加wedge是模拟最坏情况，通过其他的方式添加wedge理论上selfalign会获得更高的精度。

##### (4)normalize

首先根据之前准备的数据修改 normalize_z_score_5lqw.py 中 rota 路径。然后执行：python normalize_z_score_5lqw.py

#### 2、train model

修改 SelfAlign/models/gumnet/train_5LQW.py 中的 result_dir, src_file, predict_txt_path

其中测试数据全程不参与训练，仅仅只是有用于保存实验记录。因此可以选择注释掉 predict_txt_path 并且相应的把240行代码 predict(i) 注释掉。

```
result_dir:保存训练模型的地址
src_file:记录训练数据位置的txt文件地址，txt文件在(2)prerare train data 步骤中的 data_dir 中
predict_txt_path:记录测试数据位置的txt文件地址，txt文件在(2)prerare test data 步骤中的 data_dir 中
```

#### 3、eval

计算评价指标：修改 SelfAlign/preprocessing/eval.py 中的 y_true 和 y_pred 地址，执行 python SelfAlign/preprocessing/eval.py

## 关于snr定义以及如何添加噪声

本文对比的基线方法是GumNet

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

### 添加噪声的具体步骤

```
1. snr中add_gaussian_noise用于添加噪声。添加噪声的过程为先对数据使用normalize_z_score_foler.py进行整体归一化处理，然后根据snr对应关系的表格对每个样本添加噪声。
2. 添加完噪声后再使用normalize_z_score_foler.py进行整体归一化处理。
3. 处理得到的数据用于最终的模型训练。
```

## 关于Loss

查看 SelfAlign/models/gumnet/tf_util_loss.py 中关于 selfalign_loss 的具体实现

## 讨论

(1)SelfAlign 为subtomo alignment 提供了一种可行的思路，构建了一个自监督学习任务，可以作为今后利用深度学习实现冷冻电镜子断层图对齐的基础框架。同时SelfAlign 正准备参考传统领域多次对齐，逐步细化的思路构建完整实现 子断层图平均法STA 。目前在 SelfAlign/bin 中的 average.py 和 refine.py 正在修改测试，预计在未来实现完整的 STA，敬请期待...... 

(2) there are some examples of module tests in selfalign

[IAMCHF/test_for_selfalign at master (github.com)](https://github.com/IAMCHF/test_for_selfalign/tree/master)
