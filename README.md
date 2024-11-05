# GRU_D_EWS

## Overview
Code of Re-implementing the methodology in the paper:

A Machine Learning Approach to Incorporate Time for Early Warning Systems

This respository contains synethetically generated vital sign data and the pipeline to implement a GRU-D EWS system.  

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [License](#license)
- [Contact](#contact)

## Installation

This repository was developed in a Python 3.10.5 environment

To set up the project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/GassyAllan/GRU_D_EWS.git

2. **Navigate to the project directory**:
    ```bash
    cd GRU_D_EWS

3. **Install depenencies**:
    ```bash
    pip install -r requirements.txt

## Usage

The notebook highlights the pre-processing pipeline and provides analysis and examples of instances.  This is found here: [example notebook](GRU_D_Model/GRU_D_Demo.ipynb).

## Features

Synethetic Data can be found here:[synethetic_data](Data/ews_synthetic_data.csv).  This data was generated using Hidden Markov Models using parameters that are clinically plausable using four possible states (Stable, Unstable, Peri-arrest, Dead).  It contains 1000 examples of vital sign sequences within an in-patient environment, of which 200 result in an adverse outcome (episode_id: 2000-2199) and 800 result in hospital survival/discharge (episode_id: 2200-2999).  This data is to demonstrate the pipeline used to develop the GRU-D model described in the paper and is not reflective of real-world clinical data, which is **NOT** publically availble for patient confidentiality reasons.

Helper functions are found in the folder [GRU_D_Model](GRU_D_Model).  

These are divided into:

1. Pre-Preprocessing
  - GRU-D pre-processing functions [pre_processing_func](GRU_D_Model/pre_processing_func.py)
  - Baseline EWS Calculations [baseline_ews_calc](GRU_D_Model/baseline_ews_calc.py)

2. Modelling Training and Analysis
  - GRU-D Training [train_test_func](GRU_D_Model/train_test_func.py)
  - GRU-D and EWS Analysis [train_test_func](GRU_D_Model/train_test_func.py)

3. Plotting Functions
  - Vitals and Model outputs [plotting_func](GRU_D_Model/plotting_func.py)

## License
This project is licensed under the Apache License Version 2.0 - see the [LICENSE](LICENSE) file for details.

## Contact

Please contact the corresonding author Allan Pang for any queries. 

allan.pang@nhs.net
