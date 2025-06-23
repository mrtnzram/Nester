# System
import sys
import os
import math
import json
import multiprocessing as mp
from datetime import datetime

# Scientific Computing
import numpy as np
import pandas as pd
import librosa
from joblib import Parallel, delayed

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.colors import to_hex
from matplotlib.cm import get_cmap
from matplotlib.collections import PatchCollection
from matplotlib import colormaps

# Dimensionality Reduction + Clustering
import umap.umap_ as umap
import hdbscan

# tqdm
from tqdm.autonotebook import tqdm

# Chipper output preprocessing
from paths import initialize_paths
from paths import find_gzip_directories
from chipper_output_to_json import SongAnalysis

# AVGN + Chipper + VocalSeg moduless
from avgn.dataset import DataSet
from avgn.utils.paths import DATA_DIR, ensure_dir, FIGURE_DIR
from avgn.utils.hparams import HParams
from avgn.utils.general import save_fig
from avgn.signalprocessing.create_spectrogram_dataset import (
    pad_spectrogram, flatten_spectrograms, make_spec, mask_spec, log_resize_spec
)
from avgn.visualization.barcodes import plot_sorted_barcodes, indv_barcode
from avgn.visualization.spectrogram import draw_spec_set
from avgn.visualization.projections import scatter_spec
from vocalseg.utils import butter_bandpass_filter, spectrogram, int16tofloat32, plot_spec
from avgn.utils.audio import load_wav, read_wav
from vocalseg.dynamic_thresholding import dynamic_threshold_segmentation
from vocalseg.dynamic_thresholding import plot_segmented_spec, plot_segmentations

# Correcting Syntaxes

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_hex
from matplotlib import colormaps
import ipywidgets as widgets
from IPython.display import display