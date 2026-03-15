# AgriGuide Crop Recommendation System

## Project Overview

AgriGuide is a machine learning based crop recommendation system designed to help farmers identify the most suitable crop based on soil nutrients and environmental conditions. The model analyzes agricultural data and predicts the best crop that can be cultivated under given conditions.

## Problem Statement

Farmers often struggle to decide which crop is most suitable for their soil and environmental conditions. Incorrect crop selection can lead to low yield and financial losses. This project uses machine learning techniques to recommend the optimal crop for cultivation.

## Dataset Description

The dataset contains agricultural parameters including:

* Nitrogen (N)
* Phosphorus (P)
* Potassium (K)
* Temperature
* Humidity
* pH Value
* Rainfall
* Crop Label

These features are used to train a machine learning model that predicts the most suitable crop.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Jupyter Notebook

## Machine Learning Workflow

1. Data collection and preprocessing
2. Exploratory Data Analysis (EDA)
3. Feature scaling using StandardScaler
4. Model training using classification algorithms
5. Model evaluation and selection
6. Model serialization using Pickle

## Model Files

* best_model.pkl – Trained machine learning model
* le.pkl – Label encoder
* sc.pkl – Feature scaler

## Project Files

* AgriGuide.ipynb – Data analysis and model training notebook
* AgriGuide_model.py – Model implementation script
* AgriGuide_CropRec.csv – Dataset
* Image-Crop-Recommendation.png – Project illustration

## Key Outcome

The trained model predicts the most suitable crop based on environmental conditions, helping improve agricultural productivity and decision making.

## Author

Anamika K
