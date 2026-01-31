# Chennai House Price Sales Prediction

A machine-learning regression project that predicts **house/property sale prices in Chennai** using structured housing + locality features. The work includes **EDA**, **data cleaning**, **feature normalization/encoding**, and training/evaluating multiple regression models (Linear Regression, KNN Regressor, Decision Tree Regressor).

---

## Summary

This repository builds a supervised ML pipeline to estimate `SALES_PRICE` from housing attributes such as **area/locality, interior square footage, distance to main road, bedroom/bathroom/room counts, sale condition, parking facility, build type, utility availability, street type, municipal zone, and quality scores**.

The solution is implemented end-to-end inside a single notebook:
- `chennaiHouseSalesPrediction.ipynb`

A cleaned + encoded dataset is used to train and validate multiple models and compare performance (notably using **MSLE/RMSLE** for Linear Regression and **MSE/MSLE** for KNN).

---

## Dataset

File included in the repo:
- `chennai_house_price_prediction.csv` (**7109 rows × 19 columns**)

**Target column**
- `SALES_PRICE`

**Key input columns**
- `AREA`, `INT_SQFT`, `DIST_MAINROAD`
- `N_BEDROOM`, `N_BATHROOM`, `N_ROOM`
- `SALE_COND`, `PARK_FACIL`, `BUILDTYPE`
- `UTILITY_AVAIL`, `STREET`, `MZZONE`
- `QS_ROOMS`, `QS_BATHROOM`, `QS_BEDROOM`, `QS_OVERALL`
- `COMMIS`
- `PRT_ID` (identifier; dropped before training)

---

## What I did

1. **Loaded the dataset** (`chennai_house_price_prediction.csv`) and performed initial exploration:
   - summary stats (`describe`), missing values (`isnull`), data types, value counts
2. **Performed EDA**
   - univariate analysis (histograms / distributions / bar plots)
   - bivariate analysis (group-bys, median price comparisons, scatter plots)
3. **Cleaned and prepared the data**
   - checked duplicates
   - handled missing values:
     - filled `N_BEDROOM` and `N_BATHROOM` using **mode**
   - corrected inconsistent category labels using targeted replacements (see details below)
   - cast selected numerical count columns (`N_BEDROOM`, `N_BATHROOM`, `N_ROOM`) to **categorical/object** for proper encoding
4. **Engineered a model-ready dataset**
   - dropped non-feature identifier (`PRT_ID`)
   - applied **one-hot encoding** using `pd.get_dummies(...)`
5. **Trained and evaluated multiple regression models**
   - **Multiple Linear Regression** (baseline) using train/validation split and **MSLE/RMSLE**
   - **KNN Regressor** with **MinMax scaling**, then K-sweep (elbow) to pick a better K
   - **Decision Tree Regressor** baseline with train/validation scoring
6. **Compared model behavior** using appropriate metrics and validation scores.

---

## ML models used / trained

The notebook explicitly trains and evaluates:

1. **Multiple Linear Regression**
   - `sklearn.linear_model.LinearRegression`
   - Evaluation includes **Mean Squared Log Error (MSLE)** and **RMSLE** (via sqrt(MSLE))

2. **k-Nearest Neighbors Regressor (KNN)**
   - `sklearn.neighbors.KNeighborsRegressor`
   - Uses **MinMaxScaler** because KNN is distance-based
   - Evaluation includes **MSE** and **MSLE**
   - Includes an **elbow curve** over K values (1–24) to select a better neighbor count

3. **Decision Tree Regressor**
   - `sklearn.tree.DecisionTreeRegressor`
   - Evaluated using train/validation score (`.score()`)

---

## Data cleaning details (exact replacements performed)

To standardize inconsistent categorical values, the notebook applies these corrections:

### Parking facility
- `Noo` → `No`

### Area name normalization
- `TNagar` → `T Nagar`
- `Adyr` → `Adyar`
- `KKNagar` → `KK Nagar`
- `Chrompt`, `Chormpet`, `Chrmpet` → `Chrompet`
- `Ana Nagar`, `Ann Nagar` → `Anna Nagar`
- `Karapakam` → `Karapakkam`
- `Velchery` → `Velachery`

### Sale condition normalization
- `PartiaLl`, `Partiall` → `Partial`
- `Adj Land` → `AdjLand`
- `Ab Normal` → `AbNormal`

### Build type / utility / street normalization
- `Comercial` → `Commercial`
- `Other` → `Others`
- `All Pub` → `AllPub`
- `NoAccess` → `No Access`
- `Pavd` → `Paved`

---

## Tech stack
	•	Python
	•	pandas / numpy
	•	matplotlib
	•	scikit-learn (LinearRegression, KNN Regressor, Decision Tree, scaling, train-test split, metrics)
