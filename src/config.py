set_global_policy('float32')

drive.mount('/content/drive')

warnings.filterwarnings("ignore", category=DeprecationWarning)

base_dir = '/content/drive/MyDrive/wd_aug(5)'
original_test_dir = '/content/drive/MyDrive/test(7)'
cache_dir = '/content/drive/MyDrive/cache_dir'
memory = Memory(cache_dir, verbose=0)  # Cache memory setup
ORIGIN_DIR = '/content/drive/MyDrive/wd_origin(2)'
AUG_DIR    = '/content/drive/MyDrive/wd_aug(3)'

# GPU memory control
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass

# Fix random seed for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

input_shape = (None, 64, 64, 3)  # Input image shape
sequence_length = 6  # Sequence length
sequence_tensor = tf.constant(5)
batch_size = 4  # Batch size

model_checkpoint = ModelCheckpoint(filepath='creator_model.keras', monitor='val_cos', save_best_only=True, mode='max', verbose=1)
early_stopping = EarlyStopping(monitor='val_cos', patience=20, mode='max', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5, verbose=1)

os.cpu_count()

K_MIN, K_MAX = 0, 5
K_VAL = 5
SEQLEN = sequence_length
IMG_SIZE = (64, 64)
IMG_H = 64
IMG_W = 64

LATENT_DIM = 256
CTX_K      = 6      # Number of past frames used in Stage1/2
DELTA_MIN  = 1
DELTA_MAX  = 5
TAU_NCE    = 0.1
TAU_ANTI   = 0.03
PATCH_K    = 32
TOPK_RATIO = 0.3

LR_STAGE1  = 1e-4
LR_STAGE2  = 1e-4

SEQ_LEN = 6
STRIDE  = 1
NUM_CLASSES = 5
WOUND_IDX   = 1

AUTOTUNE = tf.data.AUTOTUNE

IMG_SIZE     = (64, 64)
BATCH        = 8
VAL_RATIO    = 0.10
MAX_T = 12

DEBUG_ROOT = Path("/content/drive/MyDrive/debug")
DEBUG_ROOT.mkdir(parents=True, exist_ok=True)

debug_dir = DEBUG_ROOT / "dbg_inverted"
debug_dir.mkdir(exist_ok=True)

DBG_LIST = debug_dir / "inverted_list.txt"


def log_inverted(tag, img_path, mask_path):
    with open(DBG_LIST, "a") as f:
        f.write(f"{tag}\t{img_path}\t{mask_path}\n")

def _to_str(x):
    x = x.numpy() if hasattr(x, "numpy") else x
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8", "ignore")
    return str(x)