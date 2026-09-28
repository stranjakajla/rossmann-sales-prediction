# Rossmann Store Sales Prediction

A machine learning project that predicts daily sales for Rossmann drug stores. It covers the full pipeline: data cleaning, exploratory data analysis, feature engineering, and training and comparing **Linear Regression** and **Random Forest** regression models.

Built as a seminar project for the course **Machine Learning** at the Faculty of Information Technologies, Džemal Bijedić University of Mostar.

## Dataset

[Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales/data) from Kaggle:

- `train.csv`: 1,017,209 rows of daily sales for 1,115 stores
- `store.csv`: store details (store type, assortment, competition, Promo2 program)

**Target variable:** `Sales` (continuous, so this is a regression problem).

## Pipeline

1. **Data loading and merging:** `train.csv` and `store.csv` joined on `Store`
2. **Feature engineering:** `Year`, `Month`, `Day` and `WeekOfYear` extracted from the date; days when stores were closed were removed, since their sales are always 0
3. **Missing values:** `CompetitionDistance` filled with the median per store type (group imputation); missing Promo2 and competition dates filled based on their meaning (the store never had Promo2, or the data is unknown)
4. **Exploratory data analysis:** descriptive statistics, skewness and kurtosis, distributions, outliers, and relationships between features and sales
5. **Encoding:** label encoding of categorical variables
6. **Correlation analysis:** correlation with the target, and detection of multicollinear feature pairs (|r| > 0.75)
7. **Modeling:** six iterations comparing algorithms, feature sets (all 18 features vs. a reduced set without multicollinear features) and train/test splits (80:20 vs. 70:30)
8. **Evaluation:** RMSE, MAE and R², actual vs. predicted plots, and feature importance

## Results

| # | Model | Features | Split | RMSE | R² |
|---|---|---|---|---|---|
| 1 | Linear Regression | all (18) | 80:20 | 1520.80 | 0.7578 |
| 2 | Linear Regression | reduced (13) | 80:20 | 1530.42 | 0.7547 |
| 3 | Linear Regression | all (18) | 70:30 | 1511.09 | 0.7609 |
| 4 | **Random Forest** | **all (18)** | **80:20** | **693.93** | **0.9496** |
| 5 | Random Forest | reduced (13) | 80:20 | 720.86 | 0.9456 |
| 6 | Random Forest | all (18) | 70:30 | 697.80 | 0.9490 |

Best model: **Random Forest, all features, 80:20 split** with RMSE = 693.93, MAE = 478.75 and R² = 0.9496.

**Key findings:**

- **Random Forest wins clearly.** It outperforms Linear Regression in every configuration, with R² ≈ 0.95 vs. ≈ 0.76 and less than half the RMSE, which shows that the relationships in the data are non-linear.
- **Removing multicollinear features doesn't help.** Performance drops slightly for both models, so these features still carry some useful information.
- **The split ratio barely matters.** Both models are stable between the 80:20 and 70:30 splits.
- **`Customers` is by far the strongest predictor.** This is also a limitation, because the number of customers is not known in advance when forecasting future sales (data leakage).

### Actual vs. predicted

![Actual vs predicted](images/09_actual_vs_predicted.png)

### Model comparison across iterations

![Model comparison](images/10_usporedba_iteracija.png)

### Feature importance (Random Forest)

![Feature importance](images/11_vaznost_varijabli.png)

## Technologies

`Python` · `pandas` · `NumPy` · `scikit-learn` · `Matplotlib` · `seaborn` · `SciPy`

## How to run

1. Download `train.csv` and `store.csv` from [Kaggle](https://www.kaggle.com/c/rossmann-store-sales/data) and put them in the `data/` folder. The dataset is not included in this repository because of its size.
2. Install the dependencies and run the script:

```bash
pip install -r requirements.txt
python rossmann_ml.py
```

All charts and the results table (`rezultati_iteracija.csv`) are saved to the `outputs/` folder. Training Random Forest on the full dataset can take a while. To train on a sample, set `SAMPLE_SIZE` in the script (e.g. `200000`).

## Project structure

```
rossmann-sales-prediction/
├── rossmann_ml.py          # Full pipeline: EDA, preprocessing, modeling, evaluation
├── requirements.txt
├── data/                   # Put train.csv and store.csv here
├── images/                 # Selected result charts
└── docs/
    └── Rossmann-Seminarski-rad.docx   # Full seminar paper (Bosnian)
```

> Code comments and chart labels are in Bosnian.

## Author

Ajla Stranjak
