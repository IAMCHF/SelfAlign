import pickle
import os
import mrcfile
import numpy as np


def read_and_serialize_to_pickle(ground_truth_text_path, input_dir_x, input_folder_y, output_pickle_path):
    # 初始化numpy数组来存储x_test和y_test
    x_test_list = []
    y_test_list = []

    y_test_dir_list = os.listdir(input_folder_y)
    test_len = len(y_test_dir_list)
    # 读取y_test数据
    for filename in y_test_dir_list:
        if filename.endswith(".mrc"):
            file_path = os.path.join(input_folder_y, filename)
            with mrcfile.open(file_path) as mrc:
                data = mrc.data.astype(np.float32)
                y_test_list.append(data[np.newaxis, ...])

    # 将y_test数据堆叠成一个大数组
    y_test = np.concatenate(y_test_list, axis=0)

    # 读取x_test数据
    with mrcfile.open(input_dir_x) as mrc:
        data = mrc.data.astype(np.float32)
    for _ in range(test_len):
        x_test_list.append(data[np.newaxis, ...])

    # 将x_test数据堆叠成一个大数组
    x_test = np.concatenate(x_test_list, axis=0)

    # 假设这里我们也从对应的MRC文件中读取observed_mask和missing_mask
    observed_mask_file_path = "/media/hao/Hard_disk_1T/datasets/gum_test_data/32/observed_mask.mrc"
    with mrcfile.open(observed_mask_file_path) as mrc:
        observed_mask = mrc.data.astype(np.float32)

    missing_mask_file_path = "/media/hao/Hard_disk_1T/datasets/gum_test_data/32/missing_mask.mrc"
    with mrcfile.open(missing_mask_file_path) as mrc:
        missing_mask = mrc.data.astype(np.float32)

    # 读取ground_truth文本文件
    ground_truth_list = []
    with open(ground_truth_text_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            items = line.strip().split("\t")
            # 只保留ground_truth值，忽略前两列图片路径
            ground_truth_values = [float(i) for i in items[2:]]
            ground_truth_list.append(ground_truth_values)

    # 将ground_truth列表转换为numpy数组
    ground_truth = np.array(ground_truth_list)

    # 将所有数据打包在一起准备序列化
    dataset = (x_test, y_test, observed_mask, missing_mask, ground_truth)

    # 序列化并保存到新的pickle文件
    with open(output_pickle_path, 'wb') as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    # 定义你的MRC文件所在输入文件夹路径以及输出的.pkl文件路径
    ground_truth_text_path = "/media/hao/Hard_disk_1T/datasets/gum_test_data/32/100-snr100.txt"
    input_dir_x = "/media/hao/Hard_disk_1T/datasets/gum_test_data/32/EMD-3228-32.mrc"
    input_folder_y = "/media/hao/Hard_disk_1T/datasets/gum_test_data/32/100-snr100"
    output_pickle_path = "/media/hao/Hard_disk_1T/datasets/gum_test_data/32/new_dataset.pickle"

    # 调用函数读取MRC文件并序列化到新的.pickle文件
    read_and_serialize_to_pickle(ground_truth_text_path, input_dir_x, input_folder_y, output_pickle_path)
