import subprocess
import sys
import importlib

def install_if_missing(package_name, install_name=None):
    if install_name is None:
        install_name = package_name
        
    try:
        importlib.import_module(package_name)
        print(f"[OK] {package_name} is already installed.")
    except ImportError:
        print(f"[INSTALLING] {package_name} (via pip install {install_name})...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", install_name])

if __name__ == "__main__":
    print("Checking and installing necessary dependencies...")
    
    # Kaggle already has torch, pandas, numpy, scikit-learn, etc.
    # We only check for the more specific ones or gradient boosting libraries
    
    install_if_missing("xgboost")
    install_if_missing("lightgbm")
    install_if_missing("catboost")
    install_if_missing("prophet")
    install_if_missing("pykalman")
    install_if_missing("optuna")
    install_if_missing("joblib")
    install_if_missing("statsmodels")
    
    print("All dependencies are satisfied!")
