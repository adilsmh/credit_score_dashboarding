## Directory Structure

```
├── app
│   ├── workflows       # storing unit testing CI scripts   
├── data                
│   ├── interim         # storing intermediate results (mostly for debugging)
│   ├── processed       # storing transformed data used for reporting, modeling, etc
│   ├── raw             # storing raw data to use as inputs to rest of pipeline
│   ├── test            # storing (submission - Kaggle) test data for model testing / results comparing 
│   ├── README.md
├── models              # storing pickle / h5 format models 
├── notebooks
│    ├── notebook_eda.ipynb 
│    ├── notebook_model.ipynb  
│    ├── notebook_predict.ipynb 
├── reports
│    ├── eda            # storing EDA plots
│    ├── model          # storing models structure/performance plots
├── scripts
│   ├── eda
│   ├── model_perf
├── .gitignore
├── README.md

```
