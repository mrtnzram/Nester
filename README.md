# Nester
*created by Ramiel Martinez as an extension of [AVGN (Sainburg, et al., 2020)](https://github.com/timsainb/avgn_paper?tab=readme-ov-file) with the assistance of Ximena Leon, Kate Snyder, and Nicole Creanza*

Nester is a software compiled inside JupyterLab with matplotlib interactive widgets which utilizes machine-learning algorithms for semi-automated syntaxxing of birdsong syllables. It is the intended step after using [Chipper](https://github.com/CreanzaLab/chipper/tree/master).
It mainly utilizes UMAP and HDBSCAN algorithms curated by the avgn package for automatic clustering of bird song syllables using spectrogram features. This software extends the capabilities of the algorithm by introducing 
human supervision. It features real time dataframe corrections, UMAP and HDBSCAN clustering changes, and visualizations. 




## Background
*Nester utilized the existing code:*

#### AVGN
[Animal Vocalization Generative Network (AVGN)](https://github.com/timsainb/avgn_paper?tab=readme-ov-file) is a
repository of python tools created by Tim Sainburg et al. (2020) centered around latent models used to generate, 
visualize, and characterize animal vocalizations. Nester pulls code from AVGN for spectrogram processing, UMAP and 
HDBSCAN methods, and visualizations.

#### Chipper output (GZIPs) to JSON
Nester utilizes modified code created by Kate Snyder to process chipper output files (GZIPs) to JSON as preprocessing
for Nester's algorithm. It takes chipper syllable onset and offset's for accuract syllable segmentation.


## Getting Started
To get Nester started on your device, follow the installation steps below and refer to the [Nester Manual](https://github.com/mrtnzram/Nester/blob/master/manual_images/Manual.md).

1) Open terminal/powershell.
2) Set path to directory where you want to clone the directory: `cd filepath`
3) Clone directory: `git clone https://github.com/username/repo-name.git`
4) Create conda environment: `conda create -n env_name python==3.10.16 
pip`
5) Activate environment: `conda activate env_name`
6) Install interactive widget dependencies: `conda install -c conda-forge ipympl jupyterlab`
7) If Jupyter is not already installed, install Jupyter: `pip install jupyter`
8) Run Jupyter lab: `jupyter lab`

## Contributing to Nester

Nester is currently in its early stages of development. In the near future, Nester would ideally become a software with a GUI. If you would like to contribute and create a GUI for Nester feel free to use the code. If you would like to suggest any changes, please feel free to submit a pull request.

If you find any bugs or would like to suggest changes or improvements to Nester, please create a new issue at [github/mrtnzram/Nester/issues](https://github.com/mrtnzram/Nester/issues).





