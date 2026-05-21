# AI-Based Prediction of Specific Capacitance in Carbon-Based Supercapacitors

## Overview

This project develops a machine learning pipeline to predict the **specific capacitance (F/g)** of carbon-based supercapacitor electrodes using experimentally reported literature data. The workflow includes data collection, preprocessing, feature engineering, model development, evaluation, and interpretability analysis.

The objective is to accelerate materials screening and provide insights into the key factors influencing supercapacitor performance.

---

## Features

- Data-driven prediction of supercapacitor capacitance
- Comprehensive data cleaning and feature engineering
- Comparison of multiple machine learning models
- Explainable AI using feature importance and permutation analysis
- Visualization of model performance and physical insights
- Evaluation across multiple independent datasets

---

## Machine Learning Models

- Linear Regression
- Random Forest Regression
- Gradient Boosting Regression
- XGBoost Regression

---

## Datasets

### Dataset 1: Biomass-Activated Carbons
- 73 samples
- Biomass-derived activated carbon materials
- Best Model: Random Forest / Gradient Boosting
- Best R²: 0.234

### Dataset 2: Carbon-Based Supercapacitors
- 556 samples
- Diverse carbon electrode materials
- Best Model: Gradient Boosting
- Best R²: 0.637

### Dataset 3: Doped Porous Carbons
- 19 samples
- Heteroatom-doped porous carbon materials
- Best Model: Random Forest
- Best R²: 0.13

### Dataset 4: Large-Scale Supercapacitor Dataset
- 4,794 samples
- Large literature-derived supercapacitor dataset
- Best Model: XGBoost
- Best R²: 0.9546

### Dataset 5: Carbon Composite Electrodes
- 38 samples
- CNF, rGO, MWCNT, and GQD-based electrodes
- Best Model: Random Forest
- Best R²: 0.023

---

## Key Findings

- Specific Surface Area (SSA) is the most influential predictor across most datasets.
- Current Density, Electrolyte Type, and Heteroatom Doping significantly affect capacitance.
- Ensemble methods consistently outperform linear models.
- Model performance improves substantially with larger and more consistent datasets.

---
## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Future Work

- Expand dataset size through automated literature mining
- Integrate advanced explainability techniques (SHAP)
- Extend the framework to other energy storage materials such as MXenes and MOFs
- Explore automated materials discovery workflows

---

## Disclaimer

This project is intended for academic and research purposes. All datasets were compiled from publicly available literature sources.
