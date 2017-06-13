from scipy.io import savemat
import numpy as np
import sys

if sys.version_info.major==3:
    models = np.load('lab2_models_python3.npz', encoding='latin1')['models']
    example = np.load('lab2_example_python3.npz', encoding='latin1')['example'].item()
    savemat('lab2_models.mat', models)
    savemat('lab2_example.mat', example)
else:
    models = np.load('lab2_models.npz', encoding='latin1')['models']
    example = np.load('lab2_example.npz', encoding='latin1')['example'].item()
    savemat('lab2_models.mat', models)
    savemat('lab2_example.mat', example)
