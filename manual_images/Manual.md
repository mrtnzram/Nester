# Nester v1.0 Manual
*updated: June 2025*

![Nester](https://github.com/mrtnzram/Nester/blob/master/manual_images/Nester.png)

The repository contains a tutorial_birds_dataset that you can follow along with to get acquainted with Nester. As you use Nester and may need to jump between stages of processing, for ease of access the jupyter notebook has been sectioned accordingly. 

![Tableofcontents](https://github.com/mrtnzram/Nester/blob/master/manual_images/tableofcontents.png)

## Table of Contents:

* [Getting Started]()
* [The Process]()
* [[0.0] Segmenting Syllables]()
* [[0.5] Making the Initial Dataset with Spectrograms]()
* [[1.0] Nester]()
* [[1.5] Visualizations]()


## Getting Started

To get Nester started on your device, follow the installation steps below
1. Open terminal/powershell.
2. Set path to directory where you want to clone the directory: `cd filepath`
3. Clone directory: `git clone https://github.com/mrtnzram/Nester.git`
4. Create conda environment: `conda create -n env_name python==3.10.16 pip`
5. Activate environment: `conda activate env_name`
6. Install interactive widget dependencies: `conda install -c conda-forge ipympl jupyterlab`
7. If Jupyter is not already installed, install Jupyter: `pip install jupyter`
8. Run Jupyter lab: `jupyter lab`


### Folder Nesting and Naming Conventions

* Folder nesting should be in the example format: (this is for locating the wav files correctly)
```
| ParentFolder:

  | Species:

    | Bird:  ( Whats important is the nesting here, the rest above can be however you like )
  
      | GZIPfolder:

      | wavfiles
```

* GZIP file naming should follow the convention:

``` SegSyllsOutput_Species-species-BirdID_F1_boutnumber ```

this is because of how the code is pulling the species, birdid, and boutnumber for the dataframe. If your gzip file doesn't match this naming convention either match our naming convention or go and modify chipper_output_to_json.py accordingly. 

*Alternatively, a preprocessing script has been made for Creanza Lab zebra finch data folder nesting and naming convention.* simply change this in env.py :

`from chipper_output_to_json import SongAnalysis`

to

`from chipper_output_to_jsonzf import SongAnalysis`

This preprocessing script follows the following Folder Nesting and Naming Convention:

```
| ParentFolder:

  | Wavs:

  | GZIPS:
```

```BirdId_boutnumber_x_xx_xx_xx_xx```


Nester requires that your environment’s python version be 3.10.16. 
You can check what python version your JupyterLab kernel is with:
