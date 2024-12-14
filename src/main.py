import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from train import main

if __name__ == "__main__":
    main(mode="ldm")
