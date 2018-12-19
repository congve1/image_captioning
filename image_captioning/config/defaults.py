import os

from yacs.config import CfgNode as CN

# ------------------------------------------------------------------------------
# Convention about Tranining / Test specific parameters
# ------------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# ------------------------------------------------------------------------------
# Config definition
# ------------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda:0"
_C.MODEL.META_ARCHITECTURE = "GeneralizedCaptionModel"

# if the WEIGHT starts with a catalog://, like :R-50, the code will look for
# the path in paths_catalog. Else, it will use it as the specified absolute
# path

_C.MODEL.WEIGHT = ""


# ------------------------------------------------------------------------------
# INPUT
# ------------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.SIZE = 256
# ------------------------------------------------------------------------------
# VOCAB
# ------------------------------------------------------------------------------
_C.VOCAB = CN()
_C.VOCAB.WORD_COUNT_THRESHOLD = 5
# ------------------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.SEQ_MAX_LEN = 20  # 50 in all coco captions
_C.DATASET.SEQ_PER_IMG = 5
_C.DATASET.TRAIN = ''
_C.DATASET.VAL = ''
_C.DATASET.TEST = ''
# ------------------------------------------------------------------------------
# DataLoader
# ------------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 0
# ------------------------------------------------------------------------------
# Encoder options
# ------------------------------------------------------------------------------
_C.MODEL.ENCODER = CN()
# The encoder conv body to use
# The string must match a function that is imported in modeling.model_builder
_C.MODEL.ENCODER.CONV_BODY = "R-101-C5"

# Add StopGrad at a specified stage so the bottom layers are frozen
_C.MODEL.ENCODER.FREEZE_CONV_BODY_AT = 0

# Whether to use fc features or not
#_C.MODEL.ENCODER.USE_FC_FEATUES = True

_C.MODEL.ENCODER.ATT_SIZE = 14
# 2048 for C5; 1024 for C4
_C.MODEL.ENCODER.FEATURE_SIZE = 2048

# ------------------------------------------------------------------------------
# Decoder options
# ------------------------------------------------------------------------------
_C.MODEL.DECODER = CN()
_C.MODEL.DECODER.CORE = 'TopDownCore'
_C.MODEL.DECODER.ATTENTION = "TopDownAttention"
# word embedding size
_C.MODEL.DECODER.EMBEDDING_SIZE = 512
# num of hidden units of the rnn
_C.MODEL.DECODER.HIDDEN_SIZE = 512
# strength of dropout in the language model rnn.
_C.MODEL.DECODER.DROPOUT_PROB = 0.5
# the hidden size of the attention in MLP, only useful in show_attend_tell;
# 0 if not using hidden layer
_C.MODEL.DECODER.ATT_HIDDEN_SIZE = 512


_C.MODEL.DECODER.BEAM_SIZE = 3

# ------------------------------------------------------------------------------
# ResNe[X]t options (ResNets = {ResNet, ResNeXt})
# ------------------------------------------------------------------------------
_C.MODEL.RESNETS = CN()

# Number of groups to use: 1 ==> ResNet; > 1 ==> ResNeXt
_C.MODEL.RESNETS.NUM_GROUPS = 1

# Baseline width of each group
_C.MODEL.RESNETS.WIDTH_PER_GROUP = 64

# Place the stride 2 conv on the 1x1 filter
# Use True only for the original MSRA ResNet; Use False for C2 and Torch models
_C.MODEL.RESNETS.STRIDE_IN_1X1 = False

# Residual Transformation function
_C.MODEL.RESNETS.TRANS_FUNC = "BottleneckWithBatchNorm"
# ResNet's stem function (conv1 and pool1)
_C.MODEL.RESNETS.STEM_FUNC = "StemWithBatchNorm"

# Apply dilation in stage "res5"
_C.MODEL.RESNETS.RES5_DILATION = 1

_C.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
_C.MODEL.RESNETS.STEM_OUT_CHANNELS = 64

# ------------------------------------------------------------------------------
# Solver
# ------------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.MAX_ITER = 40000

_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.BIAS_LR_FACTOR = 2
# after how many iterations to start self-critical training
# -1 for disable, 0 from the beginning
_C.SOLVER.SCST_AFTER = -1
# clip gradients at this value
_C.SOLVER.GRAD_CLIP = 1.0

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30000, )

_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.CHECKPOINT_PERIOD = 2500

_C.SOLVER.LOG_PERIOD = 100

_C.SOLVER.VAL_PERIOD = 1000
# Number of images per batch
# This is global
_C.SOLVER.IMS_PER_BATCH = 16

_C.SOLVER.METRIC_LOGGER_NAME = 'model'
# ------------------------------------------------------------------------------
# Specific test options
# ------------------------------------------------------------------------------
_C.TEST = CN()

# Number of images per batch
# This is global
_C.TEST.IMS_PER_BATCH = 8


# ------------------------------------------------------------------------------
# Misc options
# ------------------------------------------------------------------------------
_C.OUTPUT_DIR = "save"
_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")
