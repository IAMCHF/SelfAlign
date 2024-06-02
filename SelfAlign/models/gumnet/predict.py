from typing import List
import tensorflow as tf
import logging
from tensorflow.keras.models import load_model
from SelfAlign.models.gumnet.DCTPooling3D import DCTPooling3D
import numpy as np
import tensorflow.keras.backend as K
import os
from tensorflow.keras.utils import CustomObjectScope
from SelfAlign.models.gumnet.FeatureCorrelation import FeatureCorrelation
from SelfAlign.models.gumnet.FeatureL2Norm import FeatureL2Norm
from SelfAlign.models.gumnet.tf_util_loss import correlation_coefficient_loss
# from SelfAlign.models.gumnet.tf_util_loss import quaternion_angle_loss, translation_loss
from SelfAlign.preprocessing.img_processing import get_mrc_data

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


def predict(settings):
    custom_objects = {'DCTPooling3D': DCTPooling3D, 'FeatureL2Norm': FeatureL2Norm,
                      'FeatureCorrelation': FeatureCorrelation,
                      'correlation_coefficient_loss': correlation_coefficient_loss}
    with CustomObjectScope(custom_objects):
        model = load_model('{}/model_iter{:0>2d}.h5'.format(settings.result_dir, settings.iter_count - 1),
                           custom_objects=custom_objects)
    temple_mrc_path = '{}/tmp_{}.mrc'.format(settings.result_dir, settings.iter_count - 1)
    temple_data = get_mrc_data(temple_mrc_path)
    mrc_paths = list(settings.mrc_list)

    predict_and_save_data(mrc_paths, np.expand_dims(temple_data, axis=-1), settings.predict_batch_size,
                          model, settings.result_dir, settings.iter_count, temple_mrc_path)
    K.clear_session()


def load_source_to_dataset(source_path_list: List[str], template_mrc_data: np.ndarray, batch_size: int):
    def load_and_preprocess_source(source_path_tensor):
        def _py_func(source_path_tensor1):
            source_path = source_path_tensor1.numpy().decode() if tf.executing_eagerly() \
                else source_path_tensor1.numpy().decode('utf-8')
            source_data = np.expand_dims(get_mrc_data(source_path), axis=-1)
            # return source_data
            return source_data, source_path

        source_data, source_path = tf.py_function(_py_func, [source_path_tensor], Tout=[tf.float32, tf.string])
        # source_data = tf.py_function(_py_func, [source_path_tensor], Tout=tf.float32)
        template_mrc_data_tensor = tf.convert_to_tensor(template_mrc_data)
        return {"source_volume_input": source_data, "template_volume_input": template_mrc_data_tensor,
                "source_path": source_path}

    AUTOTUNE = tf.data.AUTOTUNE
    dataset = tf.data.Dataset.from_tensor_slices(source_path_list).map(load_and_preprocess_source,
                                                                       num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset


def predict_and_save_data(mrc_paths: List[str], template_mrc_data: np.ndarray, batch_size: int, model,
                          result_dir: str, iter_count: int, temple_path):
    dataset = load_source_to_dataset(mrc_paths, template_mrc_data, batch_size)

    iter_result_dir = f'{result_dir}/iter_{iter_count}'
    if not os.path.exists(iter_result_dir):
        os.makedirs(iter_result_dir)
    with open(f'{iter_result_dir}/params.txt', 'a') as params_file:
        for batch_data in dataset:
            pred = model.predict(batch_data, verbose=1)
            combined_params = pred['combined_params']
            for transformation, source_path in zip(combined_params, batch_data['source_path']):
                scaled_transformation = [(qi * 2 - 1)*np.pi for qi in transformation[:3]] + [(qi * 2 - 1)*4 for qi in transformation[3:]]
                transformation_str = ["{:.16f}".format(qi) for qi in scaled_transformation]
                # transformation_str = ["{:.16f}".format(qi) for qi in transformation]
                line_to_write = f"{source_path}\t{temple_path}\t"
                line_to_write += "\t".join(transformation_str)
                line_to_write += "\n"
                params_file.write(line_to_write)
