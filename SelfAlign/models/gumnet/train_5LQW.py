import random
import shutil
from typing import List
import logging
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import CustomObjectScope
from SelfAlign.models.gumnet.DCTPooling3D import DCTPooling3D
from SelfAlign.models.gumnet.FeatureCorrelation import FeatureCorrelation
from SelfAlign.models.gumnet.FeatureL2Norm import FeatureL2Norm
from SelfAlign.models.gumnet.data_sequence import *
from SelfAlign.models.gumnet.Gum_Net import get_model
from SelfAlign.models.gumnet.tf_util_loss import correlation_coefficient_loss, selfalign_loss
from SelfAlign.preprocessing.img_processing import get_mrc_data, read_parameters

tf.get_logger().setLevel(logging.ERROR)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Visible devices cannot be modified...
        print(e)


def get_cubes_list(iteration_count):
    import os
    dirs_tomake = ['train', 'valid']
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    for d in dirs_tomake:
        folder = '{}/{}'.format(data_dir, d)
        if not os.path.exists(folder):
            os.makedirs(folder)

    train_outfile = '{}/train/train_{}.txt'.format(data_dir, iteration_count)
    valid_outfile = '{}/valid/valid_{}.txt'.format(data_dir, iteration_count)
    # num_train = int(64800 * 0.8)
    # num_valid = int(64800 * 0.2)
    # num_train = int(20000 * 0.8)
    # num_valid = int(20000 * 0.2)
    num_train = int(16200 * 0.8)
    num_valid = int(16200 * 0.2)
    select_and_shuffle_lines(src_file, train_outfile, num_train)
    select_and_shuffle_lines(src_file, valid_outfile, num_valid)


def select_and_shuffle_lines(src_file, dest_file, num_lines):
    with open(src_file, 'r') as file:
        lines = file.readlines()
    if len(lines) < num_lines:
        print("select_and_shuffle_lines: lines in src_file is smaller than num_lines")
        return
    selected_lines = random.sample(lines, num_lines)
    random.shuffle(selected_lines)
    print("====================   select_and_shuffle_lines   ==================")
    with open(dest_file, 'w') as file:
        for line in selected_lines:
            file.write(line)


def train_pcrnet_continue(iter_count, out_file, model_file, lr=0.0001,
                          steps_per_epoch=128, batch_size=64):
    custom_objects = {'DCTPooling3D': DCTPooling3D, 'FeatureL2Norm': FeatureL2Norm,
                      'FeatureCorrelation': FeatureCorrelation,
                      'correlation_coefficient_loss': correlation_coefficient_loss,
                      'selfalign_loss': selfalign_loss}
    with CustomObjectScope(custom_objects):
        model = load_model(model_file)
        optimizer = Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss=selfalign_loss)
    logging.info("Loaded model from disk")
    logging.info("begin fitting")
    train_data, valid_data = prepare_custom_dataseq(data_dir, batch_size, iter_count)
    # model.fit(train_data,
    #           validation_data=valid_data,
    #           epochs=1,
    #           steps_per_epoch=steps_per_epoch,
    #           validation_steps=int(0.1 * steps_per_epoch),
    #           verbose=1)
    model.fit(train_data,
              validation_data=valid_data,
              epochs=1,
              steps_per_epoch=steps_per_epoch,
              validation_steps=int(0.25 * steps_per_epoch),
              verbose=1)
    model.save(out_file)


def prepare_first_model():
    logging.info("prepare_first_model")
    input_shape = (32, 32, 32, 1)
    model = get_model(input_shape)
    init_model_name = os.path.join(result_dir, 'model_iter00.h5')
    model.save(init_model_name)


def train_pcrnet(iter_count, model_file, steps_per_epoch, batch_size, learning_rate):
    train_pcrnet_continue(
        iter_count=iter_count,
        out_file=os.path.join(result_dir, f'model_iter{iter_count:02d}.h5'),
        model_file=model_file,
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
        lr=learning_rate)


def get_mrc_files_in_dir(directory):
    mrc_file_paths = []
    for root, dirs, files in os.walk(directory):
        mrc_file_paths.extend([os.path.join(root, file) for file in files if file.endswith('.mrc')])
    return mrc_file_paths


def predict(iter_count):
    custom_objects = {'DCTPooling3D': DCTPooling3D, 'FeatureL2Norm': FeatureL2Norm,
                      'FeatureCorrelation': FeatureCorrelation,
                      'correlation_coefficient_loss': correlation_coefficient_loss,
                      'selfalign_loss': selfalign_loss}
    with CustomObjectScope(custom_objects):
        model = load_model('{}/model_iter{:0>2d}.h5'.format(result_dir, iter_count),
                           custom_objects=custom_objects)
        # model = load_model('{}/model_iter{:0>2d}.h5'.format(result_dir, iter_count - 1),
        #                    custom_objects=custom_objects)

    ori_path, temple_path, _, _ = read_parameters(predict_txt_path)
    logging.info("--------------Start predict!-------------------")
    predict_and_save_data(ori_path, temple_path, 64,
                          model, iter_count)
    logging.info("-------------------Done predict!------------------")
    K.clear_session()


def load_and_preprocess_source(source_path, template_path):
    def _py_func(source_path1, template_path1):
        source_path_str = source_path1.numpy().decode() if tf.executing_eagerly() \
            else source_path1.numpy().decode('utf-8')
        template_path_str = template_path1.numpy().decode() if tf.executing_eagerly() \
            else template_path1.numpy().decode('utf-8')

        source_data1 = np.expand_dims(get_mrc_data(source_path_str), axis=-1)
        template_data1 = np.expand_dims(get_mrc_data(template_path_str), axis=-1)

        return source_data1, template_data1

    source_data, template_data = tf.py_function(
        _py_func, [source_path, template_path],
        Tout=[tf.float32, tf.float32])
    return {"source_volume_input": source_data, "template_volume_input": template_data}


def load_source_to_dataset(source_path_list: List[str], template_mrc_paths: List[str], batch_size: int):
    AUTOTUNE = tf.data.AUTOTUNE
    path_dataset = tf.data.Dataset.from_tensor_slices((source_path_list, template_mrc_paths))

    data_dataset = path_dataset.map(load_and_preprocess_source, num_parallel_calls=AUTOTUNE)
    data_dataset = data_dataset.batch(batch_size=batch_size, drop_remainder=False)
    data_dataset = data_dataset.prefetch(buffer_size=AUTOTUNE)

    paths_dataset = path_dataset.map(lambda sp, tp: (sp, tp)).batch(batch_size=batch_size, drop_remainder=False)
    paths_dataset = paths_dataset.prefetch(buffer_size=AUTOTUNE)

    return data_dataset, paths_dataset


def predict_and_save_data(mrc_paths: List[str], template_mrc_paths: List[str], batch_size: int, model,
                          iter_count: int):
    data_dataset, paths_dataset = load_source_to_dataset(mrc_paths, template_mrc_paths, batch_size)
    iter_result_dir = f'{result_dir}/iter_{iter_count}'
    if not os.path.exists(iter_result_dir):
        os.makedirs(iter_result_dir)
    with open(f'{iter_result_dir}/params_scaled.txt', 'a') as params_file:
        with open(f'{iter_result_dir}/params_ori.txt', 'a') as params_file2:
            for batch_data, (batch_source_paths, batch_template_paths) in zip(data_dataset, paths_dataset):
                pred = model.predict(batch_data, verbose=0)
                combined_params = pred['combined_params']
                for transformation, source_path, template_path in zip(combined_params,
                                                                      batch_source_paths.numpy(),
                                                                      batch_template_paths.numpy()):
                    source_path = source_path.decode('utf-8') if isinstance(source_path, bytes) else source_path
                    ori_transformation_str = ["{:.16f}".format(qi) for qi in transformation]
                    # scaled_transformation = [(qi * 2 - 1) * np.pi for qi in transformation[:3]] + [(qi * 2 - 1) * 32 for
                    #                                                                                qi in
                    #                                                                                transformation[3:]]
                    scaled_transformation = [(qi * 2 - 1) * np.pi for qi in transformation[:3]] + [(qi * 2 - 1) * 4 for
                                                                                                   qi in
                                                                                                   transformation[3:]]
                    transformation_str = ["{:.16f}".format(qi) for qi in scaled_transformation]
                    line_to_write = f"{source_path}\t{template_path}\t"
                    line_to_write += "\t".join(transformation_str)
                    line_to_write += "\n"
                    params_file.write(line_to_write)

                    line_to_write2 = f"{source_path}\t{template_path}\t"
                    line_to_write2 += "\t".join(ori_transformation_str)
                    line_to_write2 += "\n"
                    params_file2.write(line_to_write2)


"5LQW, 5MPA, 5T2C, 6A5L"
"snr100, snr01, snr005, snr003, snr001"
"train_data, train_data_067, train_data_133, train_data_2, train_data_random, train_data_wedge"
"test_data, test_data_067, test_data_133, test_data_2, test_data_random, test_data_wedge, test_data_rotation_strategy"
"result_iter30, result_067, result_133, result_2, result_random, result_wedge, result_rotation_strategy"

result_dir = "/newdata3/chf/result_rotation_strategy/snr001/5LQW"
src_file = "/newdata3/chf/train_data_rotation_strategy/snr001/5LQW.txt"
predict_txt_path = "/newdata3/chf/test_data_rotation_strategy/snr001/5LQW.txt"

data_dir = result_dir + "/data"
if os.path.exists(result_dir):
    os.rename(result_dir, result_dir+"~")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)


# ln_rate = 0.0001
ln_rate = 1e-4
times = 6
for i in range(times):
    print("------------------ i ========= ", i, "----------------")
    if i == 0:
        prepare_first_model()
        # logging.info("prepare_first_model")
        # init_model_name = os.path.join(result_dir, 'model_iter00.h5')
        # shutil.copy("/newdata3/chf/final_model_result/result_iter30/snr001/5LQW/model_iter04.h5", init_model_name)
    else:
        init_model = "{}/model_iter{:0>2d}.h5".format(result_dir, i - 1)
        get_cubes_list(i)
        # train_pcrnet(i, init_model, int(20000 * 0.8) // 64, 64, ln_rate)
        # train_pcrnet(i, init_model, int(64800 * 0.8) // 64, 64, ln_rate)
        train_pcrnet(i, init_model, int(16200 * 0.8) // 64, 64, ln_rate)
        if 0 < i < times:
            os.remove("{}/model_iter{:0>2d}.h5".format(result_dir, i - 1))
        ln_rate *= 0.9
        predict(i)
# predict(times)
