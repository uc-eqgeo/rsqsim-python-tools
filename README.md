# rnc2

Python code for RNC2 earthquakes theme

*Bruce Shaw and Andy Howell, October 2021*

If you use this code, please cite: 
[Shaw, et al, 2021] "An Earthquake Simulator for New Zealand"

## Repository contents

This repository consists of python tools designed for analysis of synthetic earthquake catalogues derived from the earthquake simulator RSQSim ([Richards-Dinger and Dieterich, 2012](https://pubs.geoscienceworld.org/ssa/srl/article/83/6/983/315277/RSQSim-Earthquake-Simulator)). In `examples/shaw2021`, you can find the script used to create the figures published in [Shaw, et al, 2021] (`readCatNZ.py`). Note that this script is written for a very specific machine setup; `shaw2021figures.py` in the same directory

These tools are a work in progress and the authors accept no liability for any bugs in the code. However, if you do find bugs, please get in touch and we will endeavour to fix them :). 


## Prerequisites

Installation instructions for Anaconda can be found here: <https://www.anaconda.com/products/individual#Downloads>

## Installation

   1. Open Anaconda Prompt
   2. Run `cd {PATH_TO_REPO}`
   3. Run `conda env create`
   4. Run `conda activate rnc2`
   
   ![image](https://user-images.githubusercontent.com/21334474/104807917-68da6e80-5847-11eb-8904-07e4da4f2b1d.png)
   
## Integration of the catalogue

V1.0 of a sysnthetic earthquake catalogue for New Zealand can be found at <INSERT LINK HERE>. From that Zenodo repository, download and unzip 


## Running Jupyter Notebooks
   
   1. Make sure the correct environment is activated (rnc2).
   2. Run `jupyter notebook`
   3. Navigate to `examples/rsqsim_api/`
   
   ![image](https://user-images.githubusercontent.com/21334474/105001742-1c846e00-5a95-11eb-8323-1d53ef98941b.png)
   
   4. Choose Notebook to run, e.g. `visualization/plot_event.ipynb`
   5. Run the whole notebook in a single step by clicking on the menu Cell -> Run All.
   
   ![image](https://user-images.githubusercontent.com/21334474/105001885-50f82a00-5a95-11eb-98a2-bca7760c656a.png)
   
 ## References
   
   
  


