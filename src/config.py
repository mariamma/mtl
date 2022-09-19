CROP_SIZE = 512
IMG_SIZE = 512
STAGE = 2
DATA_DIR = '/scratch/mariamma/xraysetu/dataset/input'
TRAIN_DIR = '/scratch/mariamma/xraysetu/dataset/input/stage_2_train_images'
TEST_DIR = '/scratch/mariamma/xraysetu/dataset/input/stage_2_test_images'
CACHE_DIR = '../../temp/cache'
WEIGHTS_DIR = '/scratch/mariamma/xraysetu/minmax-mtl/weights'
RESULTS_DIR = '/scratch/mariamma/xraysetu/minmax-mtl/results'
TEST_PREDICTIONS_DIR = '../../output/test'

SAMPLE_SUBMISSION_FILE = DATA_DIR + "stage_{STAGE}_sample_submission.csv"