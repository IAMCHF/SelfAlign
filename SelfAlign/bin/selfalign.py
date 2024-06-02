#!/usr/bin/env python3
import logging
import os
import sys
import traceback
import fire
from fire import core
from SelfAlign.util.dict2attr import Arg, check_parse
from SelfAlign.util.metadata import MetaData, Label, Item


class SELFALIGN:
    """
    SELFALIGN: Train on subtomos and self-align\n
    for detail description, run one of the following commands:

    selfalign.py prepare_subtomo_star -h
    selfalign.py refine -h
    selfalign.py predict -h
    """

    # log_file = "log.txt"

    def prepare_subtomo_star(self, folder_name, output_star='subtomo.star'):
        """
        \nThis command generates a subtomo star file from a folder containing only subtomogram files (.mrc).
        This command is usually not necessary in the traditional workflow, because "isonet.py extract" will generate this subtomo.star for you.\n
        isonet.py prepare_subtomo_star folder_name [--output_star] [--cube_size]
        :param folder_name: (None) directory containing subtomogram(s).
        :param output_star: (subtomo.star) output star file for subtomograms, will be used as input in refinement.
        """
        # TODO check folder valid, logging
        if not os.path.isdir(folder_name):
            print("the folder does not exist")
        md = MetaData()
        md.addLabels('rlnSubtomoIndex', 'rlnImageName')
        subtomo_list = sorted(os.listdir(folder_name))
        for i, subtomo in enumerate(subtomo_list):
            subtomo_name = os.path.join(folder_name, subtomo)
            it = Item()
            md.addItem(it)
            md._setItemValue(it, Label('rlnSubtomoIndex'), str(i + 1))
            md._setItemValue(it, Label('rlnImageName'), subtomo_name)
        md.write(output_star)

    def refine(self,
               subtomo_star: str,
               subtomo_size: int = 32,
               rota: str = None,
               gpuID: str = None,
               iterations: int = None,
               data_dir: str = None,
               pretrained_model: str = None,
               log_level: str = None,
               result_dir: str = None,
               select_subtomo_number: int = None,
               preprocessing_ncpus: int = 16,
               continue_from: str = None,
               epochs: int = 40,
               batch_size: int = None,
               steps_per_epoch: int = None,
               learning_rate: float = None,
               drop_out: float = 0.3
               ):
        """
        \ntrain neural network to correct missing wedge\n
        isonet.py refine subtomo_star [--iterations] [--gpuID] [--preprocessing_ncpus] [--batch_size] [--steps_per_epoch] [--noise_start_iter] [--noise_level]...
        :param particle_radius:
        :param rota: where rotated_mrcs
        :param select_subtomo_number:
        :param subtomo_size: high = width = depth, the size of subtomo.
        :param subtomo_star: (None) star file containing subtomogram(s).
        :param gpuID: (0,1,2,3) The ID of gpu to be used during the training. e.g 0,1,2,3.
        :param pretrained_model: (None) A trained neural network model in ".h5" format to start with.
        :param iterations: (30) Number of training iterations.
        :param data_dir: (data) Temporary folder to save the generated data used for training.
        :param log_level: (info) debug level, could be 'info' or 'debug'
        :param continue_from: (None) A Json file to continue from. That json file is generated at each iteration of refine.
        :param result_dir: ('results') The name of directory to save refined neural network models and subtomograms
        :param preprocessing_ncpus: (16) Number of cpu for preprocessing.

        ************************Training settings************************

        :param epochs: (10) Number of epoch for each iteraction.
        :param batch_size: (None) Size of the minibatch.If None, batch_size will be the max(2 * number_of_gpu,4). batch_size should be divisible by the number of gpu.
        :param steps_per_epoch: (None) Step per epoch. If not defined, the default value will be min(num_of_subtomograms * 6 / batch_size , 200)

        ************************Network settings************************

        :param drop_out: (0.3) Drop out rate to reduce overfitting.
        :param learning_rate: (0.0004) learning rate for network training.
        """
        from SelfAlign.bin.refine import run
        d = locals()
        d_args = Arg(d)
        with open('log.txt', 'a+') as f:
            f.write(' '.join(sys.argv[0:]) + '\n')
        run(d_args)

    def predict(self, star_file: str, model: str, output_dir: str = './corrected_tomos', gpuID: str = None,
                batch_size: int = None, log_level: str = "info"):
        """
        \nPredict tomograms using trained model\n
        isonet.py predict star_file model [--gpuID] [--output_dir] [--cube_size] [--crop_size] [--batch_size] [--tomo_idx]
        :param star_file: star for tomograms.
        :param output_dir: file_name of output predicted tomograms
        :param model: path to trained network model .h5
        :param gpuID: (0,1,2,3) The gpuID to used during the training. e.g 0,1,2,3.
        :param batch_size: The batch size of the cubes grouped into for network predicting, the default parameter is four times number of gpu
        :param log_level: ("debug") level of message to be displayed, could be 'info' or 'debug'
        :raises: AttributeError, KeyError
        """
        d = locals()
        d_args = Arg(d)
        from SelfAlign.bin.predict import predict

        if d_args.log_level == "debug":
            logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',
                                datefmt="%m-%d %H:%M:%S", level=logging.DEBUG,
                                handlers=[logging.StreamHandler(sys.stdout)])
        else:
            logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',
                                datefmt="%m-%d %H:%M:%S", level=logging.INFO,
                                handlers=[logging.StreamHandler(sys.stdout)])
        try:
            predict(d_args)
        except:
            error_text = traceback.format_exc()
            f = open('log.txt', 'a+')
            f.write(error_text)
            f.close()
            logging.error(error_text)

    def check(self):
        print('SelfAlign --version 0.1 installed')


def Display(lines, out):
    text = "\n".join(lines) + "\n"
    out.write(text)


def pool_process(p_func, chunks_list, ncpu):
    from multiprocessing import Pool
    with Pool(ncpu, maxtasksperchild=1000) as p:
        # results = p.map(partial_func,chunks_gpu_num_list,chunksize=1)
        results = list(p.map(p_func, chunks_list))
    # return results


if __name__ == "__main__":
    core.Display = Display
    if len(sys.argv) > 1:
        check_parse(sys.argv[1:])
    fire.Fire(SELFALIGN)
