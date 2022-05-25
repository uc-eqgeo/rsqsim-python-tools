# rsqsim-python-tools

Python code for analysis of RSQSim outputs.

*Andy Howell and Bruce Shaw, October 2021*

*(with earlier assistance from Samantha Woon and Charles Williams).*

If you use this code, please cite: 
[Shaw, et al, 2021] "An Earthquake Simulator for New Zealand", **Bulletin of the Seismological Society of America**, 2021. DOI: 
<https://doi.org/10.1785/0120210087>


## Repository contents

This repository consists of python tools designed for analysis of synthetic earthquake catalogues derived from the earthquake simulator RSQSim ([Richards-Dinger and Dieterich, 2012](https://pubs.geoscienceworld.org/ssa/srl/article/83/6/983/315277/RSQSim-Earthquake-Simulator)). 

In `examples/shaw2021`, you can find jupyter notebooks used to create some of the figures published in [Shaw, et al, 2021]. Notebooks to allow reproduction of the other figures remain a work in progress.

`src/rsqsim_api` is a python module that primarily contains code for filtering the NZ catalogue and plotting slip distributions.  Examples of how to use this code, in Jupyter Notebook format, can be found in `data/examples/rsqsim_api`. Note that at this stage (October 2021), plotting tools are designed to be used primarily with NZ data and use NZTM projection.

DISCLAIMER: These tools are a work in progress and the authors accept no liability for any bugs in the code. However, if you do find bugs, please get in touch and we will endeavour to fix them :). 


## Prerequisites

Installation instructions for Anaconda can be found here: <https://www.anaconda.com/products/individual#Downloads>

## Installation

   1. Open Anaconda Prompt
   2. Run `cd {PATH_TO_REPO}`
   3. Run `conda env create`
   4. Run `conda activate rsqsim-python-tools`
   
   
## Integration of the catalogue

V1.0 of a synthetic earthquake catalogue for New Zealand can be found at https://doi.org/10.5281/zenodo.5534462. Download the files from that Zenodo repository into `data/shaw2021` and all of the example Jupyter notebooks should work.


## Running Jupyter Notebooks
   
   1. Make sure the correct environment is activated (rsqsim-python-tools).
   2. Run `jupyter notebook`
   3. Navigate to `examples/rsqsim_api/`
   
   ![image](https://user-images.githubusercontent.com/21334474/105001742-1c846e00-5a95-11eb-8323-1d53ef98941b.png)
   
   4. Choose Notebook to run, e.g. `visualization/plot_event1837093.ipynb`
   5. Run the whole notebook in a single step by clicking on the menu Cell -> Run All.
   
   ![image](https://user-images.githubusercontent.com/21334474/105001885-50f82a00-5a95-11eb-98a2-bca7760c656a.png)
   
 ## References
   ???
   
  


