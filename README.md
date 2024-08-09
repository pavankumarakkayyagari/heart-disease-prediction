# Heart Disease Prediction
# Project Overview
This project aims to predict the likelihood of heart disease in individuals using machine learning models. The dataset used includes various health indicators such as age, gender, cholesterol levels, blood pressure, and more. The goal is to build a reliable model that can assist in early detection and prompt treatment of heart disease.

# Table of Contents
Project Overview
Dataset
Installation
Project Structure
Data Preprocessing
Exploratory Data Analysis (EDA)
Machine Learning Models
Results
Conclusion
Future Work
Contributing
License
Contact

# Dataset
The dataset used in this project is stored in the data/ directory as cardio_train.csv. It includes the following features:

age: Age of the patient (in days)
gender: Gender of the patient (1 = female, 2 = male)
height: Height of the patient (in cm)
weight: Weight of the patient (in kg)
ap_hi: Systolic blood pressure
ap_lo: Diastolic blood pressure
cholesterol: Cholesterol levels (1 = normal, 2 = above normal, 3 = well above normal)
gluc: Glucose levels (1 = normal, 2 = above normal, 3 = well above normal)
smoke: Smoking status (0 = no, 1 = yes)
alco: Alcohol intake (0 = no, 1 = yes)
active: Physical activity (0 = no, 1 = yes)
cardio: Presence of cardiovascular disease (0 = no, 1 = yes, target variable)

# Installation
To run this project locally, you'll need to have Python installed. You can install the necessary packages using the following command:

bash
Copy code
pip install -r requirements.txt
Requirements:
pandas
numpy
seaborn
matplotlib
scikit-learn



