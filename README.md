# PEM Fuel Cell Voltage Prediction and Power-Density Optimization

This project applies machine learning to electrochemical performance data from a Proton Exchange Membrane (PEM) fuel cell. The goal is to predict the cell voltage from operating conditions and then estimate the operating region that gives the highest power density.

The workflow uses a Random Forest regression model trained on fuel-cell polarization data. The model learns the relationship between current density, pressure, relative humidity, membrane compression, and cell voltage. After training, the model is interpreted using permutation importance and SHAP analysis, and then used to search for the condition that maximizes predicted power density.

---

## Project Objective

The main objectives of this project are:

1. Predict PEM fuel-cell cell voltage using operating-condition data.
2. Evaluate model performance using grouped cross-validation.
3. Identify which operating parameters have the strongest influence on voltage prediction.
4. Use the trained model to estimate maximum power density.
5. Visualize voltage and power-density trends across the polarization curve.

---

## Why This Project Matters

Fuel-cell performance depends strongly on operating parameters such as humidity, pressure, compression, and current density. In experimental electrochemistry, testing every possible combination of conditions can be time-consuming and expensive.

Machine learning can help by:

- learning trends from existing experimental data,
- predicting voltage under new operating conditions,
- identifying the most important parameters,
- supporting faster optimization of fuel-cell performance,
- reducing the number of physical experiments required.

This makes the project relevant to electrochemistry, materials science, energy systems, and data-driven optimization.

---

## Dataset

The project uses PEM fuel-cell experimental data from CSV files. The script reads data from two files:

```python
df1 = pd.read_csv('/content/pem-dataset1/Standard Test of Nafion Membrane 112/1.csv')
df2 = pd.read_csv('/content/pem-dataset1/Standard Test of Nafion Membrane 112/2.csv')
```

Expected columns include:

| Column | Meaning |
|---|---|
| `current_density` | Current density of the fuel cell, typically in mA/cm² |
| `pressure` | Operating pressure |
| `relative_humidity` | Relative humidity condition |
| `membrane_compression` | Compression applied to the membrane/electrode assembly |
| `cell_voltage` | Measured fuel-cell voltage |

The target variable is:

```python
cell_voltage
```

The input features are:

```python
current_density
pressure
relative_humidity
membrane_compression
```

---

## Methodology

### 1. Data Preparation

The two CSV files are loaded and should be combined into a single dataframe before model training:

```python
df = pd.concat([df1, df2], ignore_index=True)
```

The feature matrix and target vector are then defined:

```python
X = df[['current_density', 'pressure', 'relative_humidity', 'membrane_compression']]
y = df['cell_voltage']
```

---

### 2. Grouped Cross-Validation

Fuel-cell polarization data usually contains multiple points from the same operating condition. If the data is split randomly, points from the same curve may appear in both training and validation sets, causing data leakage.

To reduce this risk, the project groups data by operating condition:

```python
groups = (
    df['pressure'].astype(str) + '_' +
    df['relative_humidity'].astype(str) + '_' +
    df['membrane_compression'].astype(str)
)
```

Then `GroupKFold` is used for cross-validation. This ensures that entire operating-condition groups are kept together during training and validation.

---

### 3. Model Training

The model used is a Random Forest Regressor:

```python
RandomForestRegressor(random_state=42)
```

Hyperparameter tuning is performed using `GridSearchCV` with grouped cross-validation.

Example parameter grid:

```python
param_grid = {
    'n_estimators': [300, 600],
    'max_depth': [None, 15],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 1.0],
}
```

The best model is selected based on cross-validated R² score.

---

### 4. Model Interpretation

Two interpretation methods are used:

#### Permutation Importance

Permutation importance measures how much the model performance drops when each feature is randomly shuffled.

This helps answer questions such as:

- Is current density the dominant factor?
- Does relative humidity strongly affect predicted voltage?
- How important are pressure and membrane compression?

#### SHAP Analysis

SHAP values are used to understand how each feature contributes to individual model predictions.

SHAP can show:

- which features increase or decrease predicted voltage,
- how the effect changes across the polarization curve,
- whether operating conditions interact with current density.

---

### 5. Power-Density Optimization

Power density is calculated as:

```python
power_density = current_density * voltage
```

If current density is in mA/cm² and voltage is in V, then power density is in mW/cm².

The trained Random Forest model is used to predict voltage across possible operating conditions. The project then searches for the condition that maximizes predicted power density.

---

## Outputs

The project can generate the following outputs:

| Output | Description |
|---|---|
| Best model | Trained Random Forest model saved as `.joblib` |
| Cross-validation score | GroupKFold R² score |
| Feature importance plot | Shows which variables most affect voltage prediction |
| SHAP summary plot | Explains global feature effects |
| SHAP dependence plots | Shows feature effects across current density |
| Optimum power curve | Shows predicted voltage and power density vs current density |


--
