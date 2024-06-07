import os
import multiprocessing as mp

os.environ["KERAS_BACKEND"] = "torch"

try:
   mp.set_start_method('spawn', force=True)
except RuntimeError:
   pass
