# BigMart Sales Prediction

An end-to-end machine learning regression pipeline to predict **BigMart product sales** (`Item_Outlet_Sales`) using item + outlet attributes. The project covers **data inspection, missing-value handling, categorical normalization, one-hot encoding, feature scaling**, and training/evaluating tree-based regressors. A final **submission file** is generated as `bestcheck.csv`.

---

## Summary

This project predicts `Item_Outlet_Sales` using features such as:
- **Item-level:** item weight, fat content, visibility, MRP, item type/category
- **Outlet-level:** outlet identifier/type, outlet size, outlet location type, establishment year, etc.

The workflow is implemented in:
- `Attempt 4 - DT properly.ipynb`

Final output:
- `bestcheck.csv` (5681 rows) with columns:
  - `Item_Identifier`, `Outlet_Identifier`, `Item_Outlet_Sales`

---

## What I did

1. Loaded and inspected the training data (`training.csv`) to understand feature distributions, data types, and missing values.
2. Handled missing data:
   - Imputed `Outlet_Size` based on `Outlet_Type`
   - Filled missing `Item_Weight` values using mean imputation
3. Cleaned and standardized categorical features (e.g., normalized `Item_Fat_Content` labels).
4. Prepared the dataset for modeling:
   - Dropped identifier columns
   - Applied one-hot encoding to categorical variables
   - Scaled features using MinMaxScaler
   - Split data into training and validation sets
5. Trained and evaluated regression models (Decision Tree and Random Forest), including basic hyperparameter sweeps.
6. Applied the finalized Random Forest model to the test dataset and generated predictions.
7. Exported the final predictions to `bestcheck.csv` in submission-ready format.

---

## ML models used / trained

### 1) Decision Tree Regressor
- `sklearn.tree.DecisionTreeRegressor`
- Baseline tree model
- Includes a **max_depth sweep** (1–19) to visualize overfitting vs generalization using train/test scores

### 2) Random Forest Regressor (final predictor)
- `sklearn.ensemble.RandomForestRegressor`
- Used with:
  - `n_estimators=100`
  - `max_depth=6`
- Validation checked using **Mean Squared Error (MSE)**
- Includes an exploratory sweep over `min_samples_leaf`


---

## Tech stack
	•	Python
	•	NumPy, pandas
	•	matplotlib
	•	scikit-learn (DecisionTreeRegressor, RandomForestRegressor, MinMaxScaler, train_test_split, MSE)
