# Food Delivery Time Prediction

https://food-delivery-time-predictor1.streamlit.app/

Predict how many **minutes** a food delivery will take from the order details: the delivery person, distance, weather, traffic, vehicle, and more. This is a **supervised regression** project, built end to end from raw, messy data to a saved model and a small web app.

---

## Results

Four models were trained on the same 80/20 split (see the notebook). XGBoost and Random Forest were tuned with `RandomizedSearchCV`.

| Model | Test R² | Test RMSE (min) | Train R² | Notes |
|---|---|---|---|---|
| Linear Regression | 0.57 | 6.22 | 0.58 | Baseline: relationships are not linear |
| Decision Tree | 0.68 | 5.40 | 1.00 | Overfits (memorises the training set) |
| Random Forest (untuned) | 0.82 | 4.05 | 0.97 | Big gain: averaging many trees reduces overfitting |
| **Random Forest (tuned)** | **0.82** | **3.97** | n/a | Best test score in the notebook |
| XGBoost (tuned) | 0.81 | 4.17 | 0.82 | Smallest train/test gap |

*RMSE is in minutes, so the best models are off by roughly 4 minutes on average.*

**What I learned from the experiments**
- Tree-based models clearly beat Linear Regression, which suggests delivery time depends on feature interactions (for example traffic and distance together).
- A single Decision Tree memorises the training data (train R² of 1.00). Ensembles fixed most of that.
- Adding a "meal period" feature (lunch / dinner / normal) did **not** help (adjusted R² went from 0.805 to 0.804), so it was left out of the final features.

### Improved training script

`train.py` rebuilds the pipeline as a single scikit-learn `Pipeline` with two fixes over the notebook: the train/test split now happens **before** imputation (so no test-set information leaks in), and the text `"NaN"` in the weather column is treated as a missing value. With these changes the final saved model scores:

| Model | Test R² | Test RMSE (min) |
|---|---|---|
| Random Forest (tuned) | 0.829 | 3.92 |
| XGBoost (tuned) | 0.820 | 4.01 |

---

## Approach

1. **Load and inspect** the raw data
2. **Clean** messy text and missing values (for example `"conditions Sunny"` becomes `Sunny`, `"(min) 24"` becomes `24`)
3. **EDA** to see which features move delivery time
4. **Feature engineering**: `distance_km` from restaurant and customer coordinates using the **Haversine formula**; rows with impossible distances (over 100 km) removed
5. **Encode** traffic density as an ordered number and one-hot encode the other categories
6. **Train and compare** Linear Regression, Decision Tree, Random Forest and XGBoost
7. **Tune** with `RandomizedSearchCV`, **evaluate** (RMSE, R², Adjusted R²) and **save** the model

The reasoning behind each step is explained inside the notebook.

---

## Dataset

Historical food-delivery orders (about 45,000 rows) with the delivery person's age and rating, restaurant and delivery coordinates, weather, road traffic density, vehicle type and condition, order type, festival flag and city type. The target is `Time_taken(min)`. The CSV is included in this repo as `Food delivery.csv`.

---

## Tech Stack

Python, pandas, NumPy, Matplotlib, seaborn, scikit-learn, XGBoost, joblib, Streamlit

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/shivrajvarale/Food-Delivery-Prediction.git
cd Food-Delivery-Prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the web app (the model trains automatically on first run, ~10 seconds)
streamlit run app.py

# Optional: train the model on its own and print the scores
python train.py
```

To explore the full analysis, open `Food_Delivery_Time_Prediction.ipynb` in Jupyter, or use Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shivrajvarale/Food-Delivery-Prediction/blob/main/Food_Delivery_Time_Prediction.ipynb)

In Colab, upload `Food delivery.csv` and check the path in the first data-loading cell.

---

## Repository Structure

```
Food-Delivery-Prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── Food_Delivery_Time_Prediction.ipynb   # full, documented analysis
├── Food delivery.csv                     # dataset
├── train.py                              # clean, train, compare, save the model
└── app.py                                # Streamlit app for predictions
```

Running `train.py` or the app creates a `model/` folder with the saved pipeline (`food_delivery_model.joblib`) and `metrics.json`.

---

## Possible Improvements

- Add feature importance / SHAP plots and a residual analysis
- Report cross-validated scores instead of a single split
- Add a day-of-week feature from `Order_Date`
- Try LightGBM or CatBoost
- Deploy the app online (for example Streamlit Community Cloud)

---

## Author

**Shivraj Varale**
B.Tech CSE (AI & ML), VIT Bhopal University


