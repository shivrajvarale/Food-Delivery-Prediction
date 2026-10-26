# Food Delivery Time Prediction

Predict how many **minutes** a food delivery will take, given the order details —
who is delivering it, how far it has to travel, the weather, traffic, time of day,
and more. Framed as a **supervised regression** problem and solved end-to-end, from
raw messy data to a saved model ready for inference.

---

## Results

Four models were trained and compared. XGBoost (with tuned hyperparameters) came out on top.

| Model | Test R² | Test RMSE (min) | Train R² | Notes |
|---|---|---|---|---|
| Linear Regression | 0.57 | 6.21 | 0.58 | Baseline — relationships aren't linear |
| Decision Tree | 0.67 | 5.46 | 1.00 | Overfits (memorises the training set) |
| Random Forest | 0.82 | 4.03 | 0.97 | Big gain — averaging tames overfitting |
| **XGBoost (tuned)** | **0.83** | **3.91** | 0.85 | Best score *and* healthiest train/test gap |

*RMSE is in minutes, so the final model is off by roughly 4 minutes on average.*

---

## Approach

1. **Load & inspect** the raw data
2. **Clean** messy text and missing values (e.g. `"conditions Sunny"` → `Sunny`, `"(min) 24"` → `24`)
3. **EDA** — find which features actually move delivery time
4. **Feature engineering**
   - `distance_km` from restaurant/customer coordinates via the **Haversine formula**
   - `order_hour` extracted from the order timestamp
5. **Encode** categoricals (ordinal for traffic density, one-hot for the rest) and **scale** features
6. **Train & compare** Linear Regression, Decision Tree, Random Forest, XGBoost
7. **Tune** the best model (RandomizedSearchCV), **evaluate** (RMSE, R², Adjusted R²), and **save** it for inference

The maths and reasoning behind each step are explained inline in the notebook.

---

## Dataset

The dataset contains historical food-delivery orders with features such as the
delivery person's age and rating, restaurant and delivery coordinates, weather,
road-traffic density, vehicle type, order type, festival flag, and city type.
The target is `Time_taken(min)`.

---

## Tech stack

Python · pandas · NumPy · Matplotlib · seaborn · scikit-learn · XGBoost · joblib

---

## How to run

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/food-delivery-time-prediction.git
cd food-delivery-time-prediction

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the notebook
jupyter notebook Food_Delivery_Time_Prediction.ipynb
```

Then update the dataset path in the first load cell to point at your local CSV.

---

## Repository structure

```
food-delivery-time-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── Food_Delivery_Time_Prediction.ipynb   # the full, documented notebook
├── food_delivery.csv                      # dataset (or link it in the README)
├── xgb_best.pkl                           # saved best model
└── standard_scaler.pkl                    # saved scaler
```
