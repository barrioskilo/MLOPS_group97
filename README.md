# MLOPS_group97

### Project structure

------------

    ├── LICENSE
    ├── .dvc                        <- DVC configurations.
    ├── .github                     <- Folder for the workflows working in GitHub.
    │   └── workflows               <- Contains the file used in the workflow.
    │   │   ├── tests.yml        
    │
    ├── .ruff_cache                 <- Good coding practices folder.
    ├── app                         <- FastAPI folder.
    │   └── pictachio_inference.py  
    │
    ├── outputs                
    │
    ├── Makefile                    <- Makefile with commands like `train_lightning` or `predict_lightning`.
    ├── README.md                   <- The top-level README for developers using this project.
    │   
    ├── docs                        <- A default Sphinx project; see sphinx-doc.org for details.
    │
    │
    ├── reports                     <- Generated analysis as HTML, JSON, etc.
    │   └── figures                 <- Generated graphics and figures to be used in reporting.
    │
    ├── requirements.txt            <- The requirements file for reproducing the analysis environment.
    ├── requirements_tests.txt      <- The requirements file for reproducing the test environment.
    │
    ├── setup.py                    <- Make this project pip installable with `pip install -e`.
    │
    ├── pistachio
    │   ├── src                     <- Source code for use in this project.
    │   │   ├── __init__.py         <- Makes src a Python module.
    │   │   │
    │   │   ├── data                <- Scripts to download or generate data.
    │   │   │   └── make_lightning_dataset.py
    │   │   │
    │   │   ├── features            <- Scripts to turn raw data into features for modeling.
    │   │   │   └── build_features.py
    │   │   │
    │   │   ├── models              <- Scripts to train models and then use trained models to make
    │   │   │   │                       predictions.
    │   │   │   ├── lightning_predict.py
    │   │   │   └── lightning_train.py
    │   │   │
    │   │   └── monitoring          <- Scripts to make the monitoring.
    │   │   │   └── monitor.py
    │   │   │
    │   │   └── visualization       <- Scripts to create exploratory and results oriented visualizations
    │   │       └── visualize.py
    │
    ├── tests                       <- The different created tests.
    │   └── test_data.py 
    │   └── test_model.py 
    │   └── test_model2.py 
    │   └── test_training.py  
    │
    ├── wandb                       <- Weight and bias runs information.
    ├── cloudbuild.yaml                       
    ├── config.yaml                
    ├── data.dvc
    ├── inference.dockerfile
    ├── pyproject.toml
    ├── test_environment.py    
    ├── training.dockerfile                                   
    └── tox.ini                     <- tox file with settings for running tox; see tox.readthedocs.io


------------
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

## Project Description

### Goal
The overall goal of the project consists of applying all the different material that has been taught in the "02476 Machine Learning Operations Jan 24" course of the DTU. These will be applied to an image classification problem related to pistachios by focusing on data augmentation tasks while using one of the given frameworks to do so. Finally, a presentation about our findings and the submission of the final report will be made.

### Framework
For the framework part, Torchvision framework for Computer Vision has been chosen. The main purpose of using this framework is to use some of the pre-trained models provided by the framewrok. The torchvision package consists of popular datasets, model architectures, and common image transformations for computer vision.

### Data
The selected data to work with in this project is a [Pistachio Image](https://www.kaggle.com/datasets/muratkokludataset/pistachio-image-dataset/code) dataset from Kaggle which includes a total of 2148 images. There are 2 types of pistachios: 1232 of Kirmizi pitachios type and 916 of Siirt pistachios type.

### Models
We will work on an image classification task on the pistachio data using some of these models: CNN, ResNet18 and ResNet50.
