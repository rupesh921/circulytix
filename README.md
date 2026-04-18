# CircuLytix ♻️ : AI-Driven LCA Platform for Metallurgy


CircuLytix is an AI/ML-driven Life Cycle Assessment (LCA) platform designed specifically for the metallurgy and mining sectors. The tool automates LCI data estimation, enables dynamic flow visualization, and provides decision-support for advancing circular economy strategies. By offering quantitative comparisons between primary and recycled metal production routes, it empowers industries to cut emissions, reduce costs, and build self-reliance

---

## 💻 Technologies & Frameworks Used

This project utilizes a robust Python ecosystem, combining advanced machine learning libraries with a lightweight REST API backend to deliver real-time environmental impact predictions

### Core Data & Feature Engineering
Raw process, material, energy, and transport inputs are mathematically transformed (e.g., logarithmic transformations) to derive meaningful circularity features.
* **Pandas & NumPy:** For structural data manipulation and high-performance numerical preprocessing.
* **Standard Scaler:** To normalize input features.

### Machine Learning Framework
The prediction engine operates via a dual-phase ML pipeline built to estimate missing parameters and predict both resource consumption and CO2 emissions
* **Framework:** Scikit-Learn
* **Primary Models:** `HistGradientBoostingRegressor` and `MultiOutputRegressor` (for simultaneous Energy and Water prediction)
* **Ensemble Modeling:** Random Forest
* **Optimization:** `RandomizedSearchCV` combined with SciPy distributions for efficient hyperparameter tuning and cross-validation.

### Model Persistence & Deployment
To ensure fast and consistent backend deployment without the need for constant retraining, the models and their feature mappings are serialized
* **Joblib:** Used for bundling the trained models into lightweight, portable files

### Backend & API Architecture
The application interface is served via a REST API that handles JSON inputs, runs the two-stage predictions, and scales results instantly
* **Framework:** Flask49
* **Cross-Origin Support:** Flask-CORS51

---

## 📊 Model Performance Metrics
The AI models were evaluated on rigorous metrics to ensure the reliability of the Life Cycle Assessment estimations
* **Evaluation Metrics Used:** Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)
* **Combined Overall Accuracy:** **85.97%**
    * *MAE-based Accuracy:* 88.98%
    * *R²-based Accuracy:* 82.95%

---

