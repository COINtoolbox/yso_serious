# <img align="right" src="docs/images/SPICY_COIN Logo_small.png" width="350"> 

## COIN-Fink: YSO-Serious, a broker module for identifying YSO dippers

This is a Fink module, developed during the [COIN-Focus #4](https://cosmostatistics-initiative.org/focus/yso1), and aims to identify YSO-dippers within the Fink alert database.


- [yso_serious/utils.py](https://github.com/COINtoolbox/yso_serious/utils.py): 
    functions related to process the alerts taken from the stream as well as from parquet files.
    
- [yso_serious/scripts/train_iso_forest.py](https://github.com/COINtoolbox/yso_serious/scripts/train_iso_forest.py):  
    Train an isolation forest model and save results and model to file.
    
- [LICENSE](https://github.com/COINtoolbox/yso_serious/blob/master/LICENSE):
    MIT License
    
## Installation

Create a virtual environment following [these instructions](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/26/python-virtual-env/). Source it and install the other dependencies using pip:

```
python3 -m pip install -r requirements.txt
```

Then you can install the functionalities of this package.

```
python setup.py install 
```
