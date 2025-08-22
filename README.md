# Housing-Price-Prediction


This project uses machine learning regression models to predict housing prices in California based on the [California Housing dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html). The goal is to compare multiple models and evaluate their performance on various metrics.

---

## 📊 Dataset

- **Source**: `sklearn.datasets.fetch_california_housing`
- **Shape**: 20,640 rows × 8 features
- **Features**:
  - `MedInc`: Median income in block group
  - `HouseAge`: Median house age in block group
  - `AveRooms`: Average number of rooms
  - `AveBedrms`: Average number of bedrooms
  - `Population`: Block group population
  - `AveOccup`: Average house occupancy
  - `Latitude`: Latitude coordinate
  - `Longitude`: Longitude coordinate
- **Target**: Median house value (in 100,000 USD)

---

## 🔍 Exploratory Data Analysis (EDA)

- Checked for nulls and data types.
- Generated summary statistics (`df.describe()`).
- Correlation heatmap (`sns.heatmap`) showed a strong correlation between `Latitude` and `Longitude`.
- Pairplot created for visualizing relationships between features.

---

## 🧪 Model Training & Evaluation

**Train-Test Split**: 80% train / 20% test  
**Target Variable**: `dataset.target` (Median house value)

### ✅ Models Used:

| Model               | MSE       | MAE      | R² Score |
|--------------------|-----------|----------|----------|
| Linear Regression   | 0.5559    | 0.5332   | 0.5758   |
| Ridge Regression    | 0.5558    | 0.5332   | 0.5759   |
| Lasso Regression    | 0.5539    | 0.5333   | 0.5773   |
| Random Forest       | 0.2540    | 0.3268   | 0.8062   |
| AdaBoost            | 0.6145    | 0.6498   | 0.5311   |
| Gradient Boosting   | 0.2615    | 0.3483   | 0.8004   |
| XGBoost             | 0.2611    | 0.3456   | 0.8007   |
| LightGBM            | **0.2023**| **0.2966**| **0.8456** |
| CatBoost            |0.2429     | 0.3322    | 0.8146  |

📝 *Note: LightGBM performed the best among all models.*

---
- Saved using `pickle`:

  ```python
  with open("best_model.pkl", "rb") as f:
    cat_loaded = pickle.load(f)

## 💻 Run the app locally:

Install required packages:

```bash
pip install -r requirements.txt
```

### Run the app:

```bash
streamlit run app.py
```
## Screenshots
<img width="1043" height="827" alt="Screenshot 2025-08-22 123159" src="https://github.com/user-attachments/assets/034afad1-277d-4967-9d3b-c38651e3c564" />

