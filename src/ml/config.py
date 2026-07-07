import os
import random
import torch
import numpy as np
from pathlib import Path

# ==============================================================================
# ENVIRONMENT DETECTION & PATHS
# ==============================================================================
IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ

if IS_KAGGLE:
    print(">>> ENVIRONMENT DETECTED: Kaggle Notebook <<<")
    # In Kaggle, we'll assume the user pushed the whole repo and is running from inside it,
    # OR they uploaded it. If they cloned it, they will be inside the repo directory.
    # We will use /kaggle/working/ for heavy outputs to avoid git pollution and respect limits.
    PROJECT_ROOT = Path("/kaggle/working/project_outputs")
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
    
    # Check if data exists in standard Kaggle input, otherwise fallback to local repo data
    if Path("/kaggle/input/dataset/engineered_features.csv").exists():
        DATA_PATH = Path("/kaggle/input/dataset/engineered_features.csv")
    else:
        # Fallback to repo data folder if they cloned the repo with data inside
        DATA_PATH = Path("./data/processed/engineered_features.csv").resolve()
else:
    print(">>> ENVIRONMENT DETECTED: Local Machine <<<")
    # Resolve the root dynamically (src/ml/config.py -> parents[2] is root)
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_PATH = PROJECT_ROOT / "data" / "processed" / "engineered_features.csv"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
FINAL_OUTPUT_DIR = PROJECT_ROOT / "final_output"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# CONFIGURATION CONSTANTS
# ==============================================================================
TARGET_REG = "usd_zar_logret_next"
RANDOM_SEED = 42
MAX_OPTUNA_TRIALS = 15
CV_SPLITS = 3
DL_MAX_EPOCHS = 30
DL_CV_EPOCHS = 10
BATCH_SIZE = 32

# ==============================================================================
# DEVICE CONFIGURATION
# ==============================================================================
def get_device():
    if torch.cuda.is_available():
        try:
            # Test if CUDA and recurrent cuDNN kernels are actually functional
            x = torch.zeros(1, 1, 1).cuda()
            rnn = torch.nn.RNN(1, 1).cuda()
            _ = rnn(x)
            return torch.device("cuda")
        except Exception as e:
            print(f"WARNING: CUDA is available but failed verification ({e}). Falling back to CPU.")
            return torch.device("cpu")
    return torch.device("cpu")

DEVICE = get_device()

def print_gpu_info():
    print("\n" + "="*80)
    print("  HARDWARE / GPU CONFIGURATION")
    print("="*80)
    print(f"Device Target: {DEVICE}")
    if torch.cuda.is_available():
        print(f"CUDA Available: True")
        print(f"GPU Name:       {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version:   {torch.version.cuda}")
        
        # Calculate memory
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r - a  # free inside reserved
        print(f"Total VRAM:     {t / (1024**3):.2f} GB")
    else:
        print("CUDA Available: False (Falling back to CPU)")
    print("="*80 + "\n")

# ==============================================================================
# REPRODUCIBILITY
# ==============================================================================
def set_seeds(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Force os env for python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
