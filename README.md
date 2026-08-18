# Diabetes Risk Prediction

A machine learning project that predicts diabetes risk from health indicators and compares the performance of several classification models.

## Project Overview

This project uses survey-based health indicators to explore the relationship between lifestyle, general health, and diabetes. The analysis is implemented in a Jupyter Notebook and includes exploratory data analysis, model training, model evaluation, hyperparameter tuning, and data visualization.

The project currently includes:

- Correlation analysis and a feature correlation heatmap
- Logistic Regression
- Random Forest
- XGBoost
- Artificial Neural Network (ANN)
- Random Forest hyperparameter tuning with `GridSearchCV`
- ROC curve and AUC calculation for the Random Forest model
- Random Forest feature importance visualization
- Neural network training and validation curves
- A simple neural-network-based diabetes risk prediction function

## Dataset

The project uses three CSV files containing health indicators from the 2015 Behavioral Risk Factor Surveillance System (BRFSS):

- `diabetes_012_health_indicators_BRFSS2015.csv`
- `diabetes_binary_health_indicators_BRFSS2015.csv`
- `diabetes_binary_5050split_health_indicators_BRFSS2015.csv`

The current modeling workflow uses the balanced 50/50 binary dataset. It contains:

- 70,692 observations
- 21 input features
- 1 target variable: `Diabetes_binary`
- No missing values in the dataset loaded by the notebook

The target variable is binary:

- `0`: No diabetes
- `1`: Prediabetes or diabetes

The input features include high blood pressure, high cholesterol, BMI, smoking status, stroke history, heart disease, physical activity, diet, general health, mental health, physical health, age, education, and income.

## Data Split

The balanced dataset is divided into training and testing sets using an 80/20 split. Stratified sampling is used so that both target classes remain balanced.

The saved notebook output contains 14,139 test observations.

## Model Results

The following results are taken directly from the current saved outputs in `diabetes.ipynb`:

| Model | Test Accuracy | Diabetes Precision | Diabetes Recall | Diabetes F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 74.78% | 0.74 | 0.77 | 0.75 |
| Random Forest | 74.59% | 0.73 | 0.79 | 0.76 |
| XGBoost | 74.79% | 0.73 | 0.79 | 0.76 |
| Artificial Neural Network | 74.53% | — | — | — |

The models achieved similar accuracy of approximately 75%. Among the saved results, XGBoost produced the highest accuracy, while Random Forest and XGBoost both achieved a recall of 0.79 for the diabetes class.

Random Forest grid search tested combinations of:

- `n_estimators`: 100, 125, and 150
- `max_depth`: None, 10, and 20
- 5-fold cross-validation
- Accuracy as the scoring metric

The best saved parameters were:

```python
{"max_depth": 10, "n_estimators": 100}
```

## Neural Network Architecture

The neural network contains:

- A dense layer with 32 ReLU units
- A dropout layer with a rate of 0.2
- A dense layer with 16 ReLU units
- A sigmoid output layer for binary classification

It was trained for 50 epochs with a batch size of 32 using the Adam optimizer and binary cross-entropy loss.

## Visualizations

The notebook generates:

- A correlation heatmap for the balanced dataset
- A Random Forest ROC curve
- A Random Forest feature importance chart
- Neural network training and validation loss curves
- Neural network training and validation accuracy curves

## Technologies

- Python
- Jupyter Notebook
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- XGBoost
- TensorFlow / Keras

## Repository Structure

```text
Diabetes/
├── Data/
│   ├── diabetes_012_health_indicators_BRFSS2015.csv
│   ├── diabetes_binary_5050split_health_indicators_BRFSS2015.csv
│   └── diabetes_binary_health_indicators_BRFSS2015.csv
├── diabetes.ipynb
└── README.md
```

## Running the Project

1. Clone the repository:

```bash
git clone https://github.com/reeta99/Diabetes.git
cd Diabetes
```

2. Install the required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow jupyter
```

3. Update the CSV paths in `diabetes.ipynb` to point to the files inside the local `Data` directory.

4. Start Jupyter Notebook:

```bash
jupyter notebook
```

5. Open and run `diabetes.ipynb`.

## Current Limitations

- The notebook currently loads the CSV files using absolute paths from the original local computer, so the paths must be updated after cloning.
- The train/test split does not currently specify a fixed `random_state`, so rerunning the notebook may produce slightly different results.
- The neural network section currently fits `StandardScaler` separately to the test data instead of only transforming it with the scaler fitted on the training data.
- The same test set is used as validation data during neural network training.
- The risk prediction function is a demonstration and has not been validated for clinical use.
- The balanced dataset does not represent the real prevalence of diabetes in the general population.

## Disclaimer

This project is for educational and research purposes only. Its predictions are not medical diagnoses and should not replace advice, testing, or treatment from qualified healthcare professionals.
