import logging

from sklearn.metrics import euclidean_distances
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import CustomObjectScope
from SelfAlign.models.gumnet.DCTPooling3D import DCTPooling3D
from SelfAlign.models.gumnet.FeatureCorrelation import FeatureCorrelation
from SelfAlign.models.gumnet.FeatureL2Norm import FeatureL2Norm
from SelfAlign.models.gumnet.data_sequence import *
from SelfAlign.models.gumnet.Gum_Net import get_model
from SelfAlign.models.gumnet.tf_util_loss import correlation_coefficient_loss

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


def train_pcrnet_continue(iter_count, out_file, model_file, data_dir='data', epochs=40, lr=0.001,
                          steps_per_epoch=128, batch_size=64, n_gpus=2):
    custom_objects = {'DCTPooling3D': DCTPooling3D, 'FeatureL2Norm': FeatureL2Norm,
                      'FeatureCorrelation': FeatureCorrelation,
                      'correlation_coefficient_loss': correlation_coefficient_loss}
    with CustomObjectScope(custom_objects):
        model = load_model(model_file)
        optimizer = Adam(learning_rate=lr)
        loss_fn = {'correlation_coefficient_loss': correlation_coefficient_loss}
        model.compile(optimizer=optimizer, loss=loss_fn)

    logging.info("Loaded model from disk")
    logging.info("begin fitting")
    train_data, valid_data = prepare_custom_dataseq(data_dir, batch_size, iter_count)
    history = model.fit(train_data, validation_data=valid_data, epochs=epochs, steps_per_epoch=steps_per_epoch,
                        validation_steps=int(0.4 * steps_per_epoch), verbose=1)
    model.save(out_file)
    return history


def prepare_first_model(settings):
    logging.info("prepare_first_model")
    subtomo_size = settings.subtomo_size
    input_shape = (subtomo_size, subtomo_size, subtomo_size, 1)
    model = get_model(input_shape)
    init_model_name = os.path.join(settings.result_dir, 'model_iter00.h5')
    model.save(init_model_name)
    return settings


def train_pcrnet(settings):
    history = train_pcrnet_continue(
        iter_count=settings.iter_count,
        out_file=os.path.join(settings.result_dir, f'model_iter{settings.iter_count:02d}.h5'),
        model_file=settings.init_model,
        data_dir=settings.data_dir,
        epochs=settings.epochs,
        steps_per_epoch=settings.steps_per_epoch, batch_size=settings.batch_size,
        lr=settings.learning_rate, n_gpus=settings.ngpus)
    return history
