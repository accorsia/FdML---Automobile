# Machine Learning Project for Automotive Data Analysis

## Overview

This project leverages machine learning techniques to analyze and predict automotive data. The dataset includes various vehicle details such as price, technical specifications, performance metrics, and more. The primary goals are to build regression models to predict vehicle prices and classification models to categorize vehicles into distinct segments.

## Features

- **Data Pre-processing:**  
  Cleans the dataset by handling missing values and converting categorical variables into numerical ones, ensuring data consistency.

- **Feature Engineering and Selection:**  
  Analyzes feature correlations to identify the most relevant variables for both regression and classification models. Visualizations such as correlation matrices support the selection process.

- **Regression Models:**  
  Implements multiple regression algorithms (e.g., Support Vector Regression, Random Forest Regression, Gradient Boosting Regression) with hyperparameter tuning to accurately predict vehicle prices.

- **Classification Models:**  
  Employs various classification algorithms (e.g., Support Vector Machine, Random Forest, Gradient Boosting) to classify vehicles into different categories.

- **Model Evaluation:**  
  Uses metrics like Mean Squared Error (MSE), Mean Absolute Error (MAE), R-squared for regression, and Accuracy, F1-score, Precision, and Recall for classification. Evaluation results are visualized and saved as `.npz` files for further analysis.

## Project Structure

- **`main.py`**:  
  The main script that orchestrates data processing, model training, evaluation, and serialization of results.

- **`dataset.py`**:  
  Handles dataset loading, pre-processing, and augmentation (including techniques like oversampling).

- **`graph.py`**:  
  Contains functions for creating visualizations, including correlation matrices and feature importance graphs.

- **`reg.py`**:  
  Responsible for building and evaluating regression models, and calculating related metrics.

- **`clf.py`**:  
  Manages the creation and evaluation of classification models, along with performance metrics.

- **`utils.py`**:  
  Provides utility functions for additional data analysis and visualization tasks.

- **`arguments.py`**:  
  Reads and processes command-line arguments to configure the project execution.

- **`npz/`**:  
  Directory used for storing serialized evaluation metrics and results in `.npz` format.

## Workflow

1. **Data Loading and Pre-processing:**  
   The dataset is loaded and cleaned by removing null values and converting categorical data into numerical formats. Oversampling is applied to balance underrepresented classes.

2. **Feature Selection and Visualization:**  
   A correlation analysis is performed to select the best features. Visual tools like correlation matrices and feature plots aid in understanding the relationships between variables.

3. **Model Building:**  
   - **Regression:**  
     Multiple regression models are created using algorithms such as SVR, Random Forest, and Gradient Boosting. Hyperparameter tuning is conducted to determine the optimal configuration.
   - **Classification:**  
     Classification models are developed using SVM, Random Forest, and Gradient Boosting. The models are refined through extensive hyperparameter experimentation.

4. **Model Evaluation:**  
   Both model types are evaluated using cross-validation. Custom metrics are computed to compare performance, and standard evaluation metrics (MSE, MAE, R² for regression; accuracy, F1-score, precision, and recall for classification) are visualized and serialized.

5. **Results and Future Directions:**  
   Final results are summarized and saved for further analysis. Insights on model performance are discussed, along with potential future improvements and research directions.

## Contributing

Contributions to enhance the project are welcome. Please submit issues or pull requests for bug fixes and feature enhancements.
