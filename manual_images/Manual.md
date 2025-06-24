# Nester v1.0 Manual

![NesterImage0](https://github.com/mrtnzram/Nester/blob/master/manual_images/Nester.png)

##### Updated: June 2025

The repository contains a tutorial_birds_dataset that you can follow along with to get acquainted with Nester. As you use Nester and may need to jump between stages of processing, for ease of access the jupyter notebook has been sectioned accordingly.

![Tableofcontents](https://github.com/mrtnzram/Nester/blob/master/manual_images/tableofcontents.png)

## Table of Contents

- [Getting Started](#getting-started)
- [The Process](#the-process)
- [0.0 Segmenting Syllables](#00-segmenting-syllables)
- [0.5 Making the Initial Dataset with Spectrograms](#05-making-the-initial-dataset-with-spectrograms)
- [1.0 Nester](#10-nester)
- [1.5 Visualizations](#15-visualizations)

## Getting Started

To get Nester started on your device, follow the installation steps below:

1. Open terminal/powershell.
2. Set path to directory where you want to clone the directory: `cd filepath`
3. Clone directory: `git clone https://github.com/username/repo-name.git`
4. Create conda environment: `conda create -n env_name python==3.10.16 pip`
5. Activate environment: `conda activate env_name`
6. Install interactive widget dependencies: `conda install -c conda-forge ipympl jupyterlab`
7. If Jupyter is not already installed, install Jupyter: `pip install jupyter`
8. Run Jupyter lab: `jupyter lab`

### Folder Nesting and Naming Conventions

Folder nesting should be in the example format (this is for locating the wav files correctly):

```
ParentFolder:
  Species:
    Bird: (What's important is the nesting here, the rest above can be however you like)
      wavfiles
      GZIPfolder:
          gzips
```

GZIP file naming should follow the convention:

```
SegSyllsOutput_Species-species-BirdID_F1_boutnumber
```

This is because of how the code is pulling the species, birdid, and boutnumber for the dataframe. If your gzip file doesn't match this naming convention, either match our naming convention or modify `chipper_output_to_json.py` accordingly.

Alternatively, a preprocessing script has been made for Creanza Lab zebra finch data folder nesting and naming convention:

- Change:
  ```python
  from chipper_output_to_json import SongAnalysis
  ```
  To:
  ```python
  from chipper_output_to_jsonzf import SongAnalysis
  ```

This preprocessing script follows the following Folder Nesting and Naming Convention:

```
ParentFolder:
  Wavs:
  GZIPS:
    BirdId_boutnumber_x_xx_xx_xx_xx
```

### Configuring Python Version in JupyterLab

Nester requires that your environment’s Python version be 3.10.16. You can check your JupyterLab kernel's Python version with:

```python
import sys
print(sys.version)
```

If JupyterLab isn’t automatically using your environment’s Python version, run the following in your terminal/powershell:

```bash
python -m ipykernel install --user --name=env_name --display-name "Python (env_name)"
```

Then manually select:

Kernel -> Change Kernel -> `env_name`

### Dependencies

When opening Nester for the first time in your environment, run the first two cells in the notebook.

1. After running these cells, restart the kernel (Kernel → Restart Kernel). These cells only need to be run once during the initial installation of Nester to ensure all dependencies are installed.
![initialization1](https://github.com/mrtnzram/Nester/blob/master/manual_images/initialization1.png)

2. Every time you open Nester, run the following three cells to ensure all modules are imported and paths are configured correctly.
![initialization2](https://github.com/mrtnzram/Nester/blob/master/manual_images/initialization2.png)

These cells ensure that all of the necessary modules are imported and all of the paths are configured correctly. If you have a specific datetime identifier you already have in your folders that you would like to work on simply add dt_id = “YYYY-MM-DD_HH-MM-SS” inside initialize_paths like so

`Initialize_paths(dataset_id,dt_id = “YYYY-MM-DD_HH-MM-SS”)`

## The Process

Nester is a powerful tool designed to help researchers analyze bird vocalizations by clustering syllables based on their acoustic features. It begins with [Chipper](https://github.com/CreanzaLab/chipper/tree/master), a semi-automated tool that identifies the start and end times of each syllable, which are saved in a JSON file. These time points are then used by [AVGN](https://github.com/timsainb/avgn_paper/tree/V2) to generate spectrograms converted into numerical arrays that describe the properties of each syllable.

Because these arrays are high-dimensional, Nester applies UMAP (Uniform Manifold Approximation and Projection), a technique that reduces the data into two dimensions while preserving the relationships between similar syllables. This simplification makes it easier to visualize and interpret the patterns in the data. Next, HDBSCAN, a density-based clustering algorithm, groups similar syllables together and identifies any that don’t fit well into clusters.

While this automatic process reaches about 80% accuracy, Nester’s true strength lies in its flexibility: it allows users to adjust UMAP and HDBSCAN parameters in real time and manually correct any misclassified syllables. This combination of automation and human guidance enables more precise and meaningful analysis of birdsong data without compromising efficiency.


![nesterflowchart](https://github.com/mrtnzram/Nester/blob/master/manual_images/nesterflowchart.png)


## [0.0] Segmenting Syllables

In segmenting syllables, you have two options:

1. Using [Chipper](https://github.com/CreanzaLab/chipper/tree/master) to segment syllables and take the GZIP output to JSON for Nester.
2. Using [Vocalseg](https://github.com/timsainb/vocalization-segmentation/tree/master/vocalseg) or another automatic segmenting algorithm to segment syllables using dynamic thresholding (not in this repository).

We recommend using Chipper to ensure 100% accuracy in the onsets and offsets of the syllables.

### Steps:

1. Indicate the name of the folder containing all the `.gzip` files.

![segmenting1](https://github.com/mrtnzram/Nester/blob/master/manual_images/segmenting1.png)

2. Run the code and double-check that the number of gzip folders match. `Find_gzip_directories` walks through your root directory and finds the folders containing gzips.

![segmenting2](https://github.com/mrtnzram/Nester/blob/master/manual_images/segmenting2.png)

3. Process the GZIPs by running the cell. This step may take a few minutes depending on your dataset size. The tutorial birds data set only contains 4 birds with around 5 bouts each which takes around 3 to 5 minutes to run depending on your device.

![segmenting3](https://github.com/mrtnzram/Nester/blob/master/manual_images/segmenting3.png)

## [0.5] Making the Initial Dataset with Spectrograms

In this section, you will create your initial dataset for processing with Nester, given you have your JSON files at hand.

1. Use the JSON files created earlier to generate your initial dataframe.

![spectrograms1](https://github.com/mrtnzram/Nester/blob/master/manual_images/spectrograms1.png)

### Columns in the Dataframe:

- `Length_s`: Length of the `.wav` file in seconds.
- `Rate`: Sample rate of the `.wav` file.
- `Wav_location`: Correct file location of the `.wav` files.
- `Species`: Bird species.
- `Key`: Bout IDs.
- `Start_time`: Syllable onset in seconds.
- `End_time`: Syllable offset in seconds.
- `Audio`: Respective audio values for the syllable (on a scale from -1 to 1).

Note: the Audio column is grabbed by the librosa module, on a scale from -1 to 1 for amplitude. You can convert this into decibel units using the following command if youd like:

![spectrograms2](https://github.com/mrtnzram/Nester/blob/master/manual_images/spectrograms2.png)


`librosa.amplitude_to_db(S, ref=1.0, amin=1e-05, top_db=80.0)`

### Next Steps:

2. Use the next cell to check how many syllables you have in your dataset.

![spectrograms3](https://github.com/mrtnzram/Nester/blob/master/manual_images/spectrograms3.png)

3. Normalize the audio. Normalizing ensures that all spectrograms are clear for processing.

![spectrograms4](https://github.com/mrtnzram/Nester/blob/master/manual_images/spectrograms4.png)

4. With these data features, create spectrograms using AVGN. Adjust hyperparameters as needed for your dataset. Default settings usually suffice.

![spectrograms5](https://github.com/mrtnzram/Nester/blob/master/manual_images/spectrograms5.png)

5. Ensure the number of spectrograms matches the number of `.wav` files. If not, check your GZIP directories in the Segmenting Syllables section.

![spectrograms6](https://github.com/mrtnzram/Nester/blob/master/manual_images/spectrograms6.png)

6. Run these three cells to create your spectrograms. This uses AVGN’s `make_spec` script to create numerical arrays for processing. Default `n_jobs = -1` uses all CPU cores. Reduce to `n_jobs = 1 or 2` if needed.

![spectrograms7](https://github.com/mrtnzram/Nester/blob/master/manual_images/spectrograms7.png)

7. Verify all spectrograms were created correctly.

![spectrograms12](https://github.com/mrtnzram/Nester/blob/master/manual_images/spectrograms12.png)

(Note: If you are following along with the tutorial birds dataset your spectrograms may or may not look different, this is simply based on how your dataframe rows are ordered. Pandas normally arranges your dataset numerically according to BirdId, but Bird 79618 only has 5 digits compared (the rest have 6) which may be processed last or first. This is completely fine.)


8. (Optional) Log-transform spectrograms by adjusting the `log_scaling_factor`.

![spectrograms13](https://github.com/mrtnzram/Nester/blob/master/manual_images/spectrograms13.png)

9. Pad spectrograms so they all have the same size for processing.

![spectrograms8](https://github.com/mrtnzram/Nester/blob/master/manual_images/spectrograms8.png)

10. Save spectrograms in the dataframe.

![spectrograms9](https://github.com/mrtnzram/Nester/blob/master/manual_images/spectrograms9.png)

11. Grab sequence positions of syllables with the provided cell.

![spectrograms10](https://github.com/mrtnzram/Nester/blob/master/manual_images/spectrograms10.png)
   
Updated dataframe columns include:

- `Length_s`, `Rate`, `Wav_location`, `Species`, `Key`, `Start_time`, `End_time`, `Audio`
    - `Syllables_sequence_pos`: Sequence positions for each bout starting at 0.
    - `Syllables_sequence_id`: Unique sequence ID starting at 0 per bird.
 
12. We now have everything we need for Nester syntaxing. With these two cells you will be able to save your dataset as a pickle or csv for future Nesting.

![spectrograms11](https://github.com/mrtnzram/Nester/blob/master/manual_images/spectrograms11.png)

*Note: if you need to use the spectrogram values again, you will need to read the pickle file instead of the csv because of array handling.*

## [1.0] Nester

This section is the core of Nester, containing syntaxing functions after dataset generation. It includes features such as:

- Dynamically changing UMAP and HDBSCAN parameters.
- Correcting syntaxes.
- Saving figures.

Before using Nester, enable matplotlib interactive widgets and load your dataset. Activating Nester performs initial clustering, which may take a moment depending on dataset size.

![nester1](https://github.com/mrtnzram/Nester/blob/master/manual_images/nester1.png)
![nester0](https://github.com/mrtnzram/Nester/blob/master/manual_images/nester0.png)

### Overview

Nester’s default view plots all bouts by bird. This bird’s eye view shows syllable clustering. Logs and summaries appear at the bottom. shift through your birds by choosing from the bird drop down menu.

![nester2](https://github.com/mrtnzram/Nester/blob/master/manual_images/nester2.png)
![nester4](https://github.com/mrtnzram/Nester/blob/master/manual_images/nester4.png)

*to open the logs, press the log icon beside Python(env_name)*

![nester00](https://github.com/mrtnzram/Nester/blob/master/manual_images/nester00.png)

### Clustering

Nester performs clustering by bird. Everytime you activate Nester, it performs the clustering for every bird and stores these initial syntaxes (starts at 0) in the hdbscan_syntax_label column. As you change the parameters, these syntaxes will change accordingly.

#### Parameters:

You are able to dynamically change the parameters and update the clustering in real time to get more accurate syntaxes depending on the complexity of the birdsongs.

- **UMAP distance**: Controls how tightly UMAP clusters points. Lower values = clumpier embeddings. Higher values = broader structure preservation. Default = 0.25.

- **Cluster size**: HDBSCAN minimum cluster size. Higher values = fewer clusters, lower values = more clusters. Default = 5.

For parameter details, refer to the [UMAP documentation](https://umap-learn.readthedocs.io/en/latest/parameters.html) and [HDBSCAN documentation](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html).

*Note: If you notice Nester isn’t updating the plots, the parameters may be invalid (e.g,  too small, or 0 cluster size). Simply check the logs for errors.*

As you change the parameters for clustering, It stores the values in the columns hdbscan_mcs and umap_min_dist accordingly. This saves your settings as you may shift through the different birds you have. This may also be useful to you as you want to project them in the UMAP space.

### Correcting Syntaxes

After adjusting parameters, select specific bouts via the Bout dropdown. Increment or decrement syntax by clicking spectrograms. Toggle modes in the top-left corner.

!![nester3](https://github.com/mrtnzram/Nester/blob/master/manual_images/nester3.png)

### Saving Labels

After finalizing bout corrections, click “Save Labels to Dataframe.” This transfers all of the syntaxes to a new column named “final_syntax”. This ensures that when you want to pause your work and come back again, after running Nester it doesn’t overwrite your changes when it clusters all of the birds.

![nester4](https://github.com/mrtnzram/Nester/blob/master/manual_images/nester4.png)

### Saving Figures

Save figures using the “Save All Figures” button. Saved figures are located in `data/output_figures/corrected`.

### Finalizing Changes

After corrections, save progress to a pickle or CSV file named `dataset_id_corrected`. 

![nester5](https://github.com/mrtnzram/Nester/blob/master/manual_images/nester5.png)

Note: When returning to work later simply run the paths and modules cells and go to the Nester Section. Don’t forget to change the dataset you are working with from save_loc to pickle_corrected.

Finalize dataset changes with provided cells. This fills in all of the empty rows in your “final_syntax” column with the “hdbscan_syntax_labels”. This may be useful as some birds may not need corrections at all. Do not run these cells if the changes aren’t finalized yet as you will not be able to change your parameters anymore. 

![nester7](https://github.com/mrtnzram/Nester/blob/master/manual_images/nester7.png)


Final dataset includes:

- `Hdbscan_syntax_label`: Clusters created with `hdbscan_mcs` and `umap_min_dist` parameters.
- `Final_syntax`: Finalized clusters with corrections.
- `Hdbscan_mcs`: HDBSCAN minimum cluster size parameter.
- `Umap_min_dist`: UMAP minimum distance parameter.

![nester6](https://github.com/mrtnzram/Nester/blob/master/manual_images/nester6.png)

## [1.5] Visualizations

With the finalized dataset, Nester offers visualizations for projecting syllables in UMAP space using AVGN script:

1. **By Bird**: Plots syllables per bird, colored by cluster.

![bybird](https://github.com/mrtnzram/Nester/blob/master/data/output_figures/tutorial-birds/umap/indiv_mapping/Individual_mapping_Junco-hyemalis_139623_RST.png)

2. **By Species**: Plots syllables per species, colored by bird.

![byspecies](https://github.com/mrtnzram/Nester/blob/master/data/output_figures/tutorial-birds/umap/species_mapping/Species_mapping_Junco-hyemalis_RST.png)












