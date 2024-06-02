import os
import numpy as np
import mrcfile
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import concurrent.futures

MASK_PATH = "mask_32.mrc"


def load_mask(mask_path=MASK_PATH):
    with mrcfile.open(mask_path, permissive=True) as mrc:
        mask = mrc.data.astype(np.float32)
    return mask


def preprocess_mrc_data(mrc_path, mask):
    with mrcfile.open(mrc_path, permissive=True) as mrc:
        volume = mrc.data.astype(np.float32)
    return volume * mask


class CustomDataSequence(Sequence):
    def __init__(self, txt_records, batch_size, shuffle=True):
        self.txt_records = txt_records
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(txt_records))
        if shuffle:
            np.random.shuffle(self.indexes)
        self.preload_buffer = {}
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def __len__(self):
        return len(self.txt_records) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
        # Preload the first batch of the next epoch
        self.preload()

    def preload(self):
        next_batch_start = (len(self) - 1) * self.batch_size
        indexes = self.indexes[next_batch_start:next_batch_start + self.batch_size]
        futures = []
        for i in indexes:
            record = self.txt_records[i]
            future = self.executor.submit(load_and_preprocess, record, mask_binned5)
            futures.append(future)
        self.preload_buffer = {i: future.result() for i, future in enumerate(futures)}

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]

        sources, templates, transformations = [], [], []

        for i in indexes:
            if i in self.preload_buffer:
                source_vol, template_vol = self.preload_buffer.pop(i)
            else:
                record = self.txt_records[i]
                source_vol, template_vol = load_and_preprocess(record, mask_binned5)

            sources.append(source_vol[:, :, :, np.newaxis])
            templates.append(template_vol[:, :, :, np.newaxis])

            record = self.txt_records[i]
            transformation = record['combined_params']
            transformations.append(transformation)

            # Preload the next batch while training is happening
            if len(self.preload_buffer) < self.batch_size:
                next_idx = (idx + 1) * self.batch_size
                next_record = self.txt_records[next_idx % len(self.txt_records)]
                future = self.executor.submit(load_and_preprocess, next_record, mask_binned5)
                self.preload_buffer[next_idx] = future

        sources = np.array(sources, dtype=np.float32)
        templates = np.array(templates, dtype=np.float32)
        transformations = np.array(transformations, dtype=np.float32)

        sources_tf = tf.convert_to_tensor(sources, dtype=tf.float32)
        templates_tf = tf.convert_to_tensor(templates, dtype=tf.float32)
        transformation_tf = tf.convert_to_tensor(transformations, dtype=tf.float32)

        return ({
                    'source_volume_input': sources_tf,
                    'template_volume_input': templates_tf
                }, {
                    'combined_params': transformation_tf
                })


def load_and_preprocess(record, mask):
    source_mrc_path = record['source_mrc_path']
    template_mrc_path = record['template_mrc_path']
    source_vol = preprocess_mrc_data(source_mrc_path, mask)
    template_vol = preprocess_mrc_data(template_mrc_path, mask)
    return source_vol, template_vol


def create_signature(batch_size):
    input_signature = (
        {
            'source_volume_input': tf.TensorSpec(shape=(batch_size, None, None, None, 1), dtype=tf.float32),
            'template_volume_input': tf.TensorSpec(shape=(batch_size, None, None, None, 1), dtype=tf.float32)
        },
        {
            'combined_params': tf.TensorSpec(shape=(batch_size, 6), dtype=tf.float32)
        }
    )
    return input_signature


def custom_sequence_to_dataset(seq, batch_size):
    ds = tf.data.Dataset.from_generator(
        lambda: seq,
        output_signature=create_signature(batch_size)
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def prepare_custom_dataseq(data_folder, batch_size, iter_count):
    global mask_binned5
    mask_binned5 = load_mask()

    train_txt_path = os.path.join(data_folder, 'train/train_{}.txt'.format(iter_count))
    valid_txt_path = os.path.join(data_folder, 'valid/valid_{}.txt'.format(iter_count))

    train_records = read_non_empty_lines(train_txt_path)
    valid_records = read_non_empty_lines(valid_txt_path)

    train_data_seq = CustomDataSequence(train_records, batch_size)
    valid_data_seq = CustomDataSequence(valid_records, batch_size)

    train_data = custom_sequence_to_dataset(train_data_seq, batch_size)
    valid_data = custom_sequence_to_dataset(valid_data_seq, batch_size)
    return train_data, valid_data


def read_non_empty_lines(file_path):
    records = []
    with open(file_path, 'r') as file:
        for line in file:
            line_contents = line.strip().split()
            source_mrc_path = line_contents[0]
            template_mrc_path = line_contents[1]
            combined_params = np.array(line_contents[2:], dtype=np.float32)
            record = {
                'source_mrc_path': source_mrc_path,
                'template_mrc_path': template_mrc_path,
                'combined_params': combined_params
            }
            records.append(record)

    return records
