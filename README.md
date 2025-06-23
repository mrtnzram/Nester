# Nester
*created by Ramiel Martinez as an extension of [AVGN (Sainburg, et al., 2020)](https://github.com/timsainb/avgn_paper?tab=readme-ov-file) with the assistance of Ximena Leon, Kate Snyder, and Nicole Creanza*

Nester is a software compiled inside JupyterLab which utilizes machine-learning algorithms for semi-automated syntaxxing of birdsong syllables. It is the intended step after using [Chipper](https://github.com/CreanzaLab/chipper/tree/master).
It mainly utilizes UMAP and HDBSCAN algorithms curated by the avgn package for automatic clustering of bird song syllables using spectrogram features. This software extends the capabilities of the algorithm by introducing 
human supervision. It features real time dataframe corrections, UMAP and HDBSCAN clustering changes, and visualizations. 




## Background
*Nester utilized the existing code:*

### AVGN
[Animal Vocalization Generative Network (AVGN)](https://github.com/timsainb/avgn_paper?tab=readme-ov-file) is a
repository of python tools created by Tim Sainburg et al. (2020) centered around latent models used to generate, 
visualize, and characterize animal vocalizations. Nester pulls code from AVGN for spectrogram processing, UMAP and 
HDBSCAN methods, and visualizations.

### Chipper output (GZIPs) to JSON
Nester utilizes modified code created by Kate Snyder to process chipper output files (GZIPs) to JSON as preprocessing
for Nester's algorithm. It takes chipper syllable onset and offset's for accuract syllable segmentation.


## Getting Started
To get these started [Nester Manual]()






