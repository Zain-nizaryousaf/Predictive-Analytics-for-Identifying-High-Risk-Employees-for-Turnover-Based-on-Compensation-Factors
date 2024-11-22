# Predictive Analytics for Identifying High-Risk Employees for Turnover Based on Compensation Factors

## Table of Contents
1. [Project Overview](#project-overview)
2. [Introduction](#introduction)
3. [Goal of the Project](#goal-of-the-project)
4. [Dataset](#dataset)
    - [Dataset Characteristics](#dataset-characteristics)
5. [Data Preprocessing](#data-preprocessing)
    - [Handling Missing Data](#handling-missing-data)
    - [Categorical Encoding](#categorical-encoding)
    - [Feature Scaling](#feature-scaling)
6. [Exploratory Data Analysis](#exploratory-data-analysis)
    - [Visualizations](#visualizations)
    - [Correlation Analysis](#correlation-analysis)
7. [Feature Engineering](#feature-engineering)
8. [Model Implementation](#model-implementation)
    - [Logistic Regression](#logistic-regression)
    - [Decision Tree Classifier](#decision-tree-classifier)
    - [Support Vector Machine (SVM)](#support-vector-machine-svm)
    - [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
9. [Hyperparameter Tuning](#hyperparameter-tuning)
10. [Cross-Validation](#cross-validation)
11. [Model Evaluation](#model-evaluation)
12. [Conclusion](#conclusion)
13. [Limitations and Future Work](#limitations-and-future-work)

---

## Project Overview

The objective of this project is to predict employee turnover based on compensation-related factors, using machine learning models. By analyzing attributes like gross wages, overtime pay, longevity pay, and other compensation-related variables, the goal is to identify employees at high risk of leaving the organization.

This project explores the following steps:
1. Data Preprocessing
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Model Training with multiple classifiers
5. Model Evaluation and Comparison

---

## Introduction

Employee retention is a significant concern for organizations. The ability to predict employee turnover enables businesses to take proactive steps in retaining their best talent. In this project, we will apply predictive analytics to identify high-risk employees based on their compensation factors.

We will use multiple machine learning models to predict whether an employee will stay or leave the organization, using a dataset that includes various compensation-related factors.

---

## Goal of the Project

The goal of this project is to develop a machine learning model that can predict high-risk employees for turnover based on compensation factors. This will help HR and management to take proactive measures, improving employee retention rates.

### Objectives:
- Preprocess the data by handling missing values, encoding categorical variables, and scaling numerical features.
- Train multiple models (Logistic Regression, Decision Trees, SVM, and KNN) to predict employee turnover.
- Evaluate the models using accuracy, precision, recall, ROC-AUC, and confusion matrices.
- Tune the models using hyperparameter tuning and cross-validation.

---

## Dataset

The dataset contains information about employees' compensation and other work-related factors. Below are the key columns in the dataset:

- **Person Name**: The name of the employee.
- **Organization**: The organization to which the employee belongs.
- **Job**: The job title or role of the employee.
- **Gross Wages**: The total wages paid to the employee.
- **Base Salary**: The base salary of the employee.
- **Longevity Pay**: Additional pay based on the employee's years of service.
- **Overtime**: Additional pay for overtime worked.
- **Other Benefits**: Other benefits provided to the employee.
- **Separation Pay**: The pay received by the employee upon leaving the company.
- **PERS Contributions**: Contributions made to the employee's retirement plan.
- **ER Paid**: Employer-paid contributions for retirement plans.
- **Annual Buybacks**: Compensation for unused benefits.
- **Year Ending**: The date the data pertains to.

---

### Dataset Characteristics

- **Data Types**: The dataset includes numerical columns (e.g., Gross Wages, Base Salary) and categorical columns (e.g., Job, Organization).
- **Target Variable**: The target variable is likely whether the employee has left the organization or not (Turnover).
- **Missing Values**: Some columns may have missing values, which will be handled during preprocessing.

---

## Data Preprocessing

Data preprocessing is essential for building accurate machine learning models. It involves the following steps:

### Handling Missing Data
Missing data is imputed using statistical techniques such as mean imputation for numerical columns.

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
df['column_name'] = imputer.fit_transform(df[['column_name']])


