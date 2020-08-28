# CoRa

CoRa (COVID-19 Radiomics) is a simple CLI-tool which can create lung segmentations and subsequently extract radiomics features from chest CT scans. It was originally created for and used in my master's thesis, but it can be applied to the broader setting of radiomic analysis of lung lobes. It combines the functionality of the [lungmask](https://github.com/JoHof/lungmask) and [pyradiomics](https://github.com/Radiomics/pyradiomics) libraries. It attempts to simplify the workflow of extracting radiomic features.  



## Installation
To install CoRa (on Linux systems), clone the repository, then navigate to the source folder and run `pip install .`. This will install CoRa, and register it as a command line tool. Cora is a simple command line interface, tailored for use with specific datasets provided in the `data/` folder.

## Quickstart
To get started you will need chest CT scans in the NifTI file format. Additionally, a `params.yaml` file is required, which should contain the extraction parameters. An example parameter file is already included in the base directory. 

To create the image segmentations, write `cora masks` in a terminal, after installation. This will create lung segmentations based on the different lung lobes.

In the root folder, add a `cases.csv` file, with three columns, Image, Mask and Label. These columns should contain the paths to the image, the segmentation, and the mask label you want to analyze. A default casefile will be added as an example.

To extract the features using the segmentations we created, execute the `cora run` command. If you use default settings and file locations this should work, otherwise you should provide the modified file locations using the run command parameters (see below).

## Commands
To show options simply run `cora`. A short description for each command:
 - 'cora run': extracts features
 

## Versioning
Currently on version 0.1
We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/FrankTN/SW_COVID/tags). 

## Authors

* **Frank te Nijenhuis** - *Initial work* - [FrankTN](https://github.com/FrankTN)
