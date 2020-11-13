lab-project-demo
==============================

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make virtual` or `make install`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- This folder is only for local developing, delete .gitkeep to not include
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │   ├── references     <- Data dictionaries, manuals, and all other explanatory materials.
    │   ├── reports        <- Generated analysis as HTML, PDF, LaTeX, etc.
    │      └── figures     <- Generated graphics and figures to be used in reporting
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-az-initial-data-exploration`.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so project can be imported
    │
    ├── lab-project-demo  <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    ├── .env               <- Environment variables go here, can be read by `python-dotenv` package
    │
    ├── sandbox            <- Help to create a sanbdox env to test the project on local.
    │   │                     Please change only the runtime_requirements.txt file 
    │   │                     to specify the requirments needed during production runtime
    │   ├── create_cluster
    │   ├── run_now
    │   ├── run_pipeline
    │   └── runtime_requirements.txt
    │
    ├── deployment              <- Scripts to test and deploy in databricks
    │   ├── deployment.yml      <- Deployment configuration
    │   │
    │   ├── cicd-databricks     <- Python package de deploy on databricks
    │   │
    │   ├── dev-tests           <- Here goes the dev pipelines test. Excuted when merging to dev
    │   │   ├── pipeline1
    │   │   │   ├── job_spec_aws.json
    │   │   │   └── pipeline_runner.py
    │   │   │
    │   ├── integration-tests   <- Here goes the integration pipelines test. Excuted when merging to main
    │   │   ├── pipeline1
    │   │   │   ├── job_spec_aws.json
    │   │   │   └── pipeline_runner.py
    │   │   │
    │   └── pipelines           <- Here goes the production job pipelines. Excuted after release and deployment
    │       └── pipeline1
    │           ├── job_spec_aws.json
    │           └── pipeline_runner.py
    │
    │ job_spec_aws.json     : Contain model name, cluster configuration, and the job scheduling
    │ pipeline_runner.py    : The main pipeline script 
    └──
