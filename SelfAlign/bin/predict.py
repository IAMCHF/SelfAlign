#!/usr/bin/env python3
import os, sys
from SelfAlign.util.metadata import MetaData, Label, Item
from SelfAlign.util.dict2attr import idx2list
# from SelfAlign.models.gumnet.predict import predict_one


# def predict(args):
#     import logging
#     # tf_logger = tf.get_logger()
#     # tf_logger.setLevel(logging.ERROR)
#
#     logger = logging.getLogger('predict')
#     if args.log_level == "debug":
#         logging.basicConfig(format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
#                             datefmt="%H:%M:%S", level=logging.DEBUG, handlers=[logging.StreamHandler(sys.stdout)])
#     else:
#         logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',
#                             datefmt="%m-%d %H:%M:%S", level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
#     logging.info('\n\n######SelfAlign starts predicting######\n')
#
#     args.gpuID = str(args.gpuID)
#     args.ngpus = len(list(set(args.gpuID.split(','))))
#
#     if args.batch_size is None:
#         args.batch_size = 4 * args.ngpus  # max(4, 2 * args.ngpus)
#     # print('batch_size',args.batch_size)
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuID
#     # check gpu settings
#     from SelfAlign.bin.refine import check_gpu
#     check_gpu(args)
#
#     logger.debug('percentile:{}'.format(args.normalize_percentile))
#
#     logger.info('gpuID:{}'.format(args.gpuID))
#
#     if not os.path.isdir(args.output_dir):
#         os.mkdir(args.output_dir)
#     md = MetaData()
#     md.read(args.star_file)
#     if not 'rlnAlignedSubtomoName' in md.getLabels():
#         md.addLabels('rlnAlignedSubtomoName')
#         for it in md:
#             md._setItemValue(it, Label('rlnAlignedSubtomoName'), None)
#     args.tomo_idx = idx2list(args.tomo_idx)

