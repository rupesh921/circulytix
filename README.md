# CircuLytix ♻️ : AI-Driven LCA Platform for Metallurgy

[cite_start]**Smart India Hackathon 2025** [cite: 1] | [cite_start]**Problem Statement ID:** 25069 [cite: 2] | [cite_start]**Team:** Maverick Minds [cite: 7]

[cite_start]CircuLytix is an AI/ML-driven Life Cycle Assessment (LCA) platform designed specifically for the metallurgy and mining sectors[cite: 3, 20]. [cite_start]The tool automates LCI data estimation, enables dynamic flow visualization, and provides decision-support for advancing circular economy strategies[cite: 20]. [cite_start]By offering quantitative comparisons between primary and recycled metal production routes, it empowers industries to cut emissions, reduce costs, and build self-reliance[cite: 24, 34].

---

## 💻 Technologies & Frameworks Used

[cite_start]This project utilizes a robust Python ecosystem, combining advanced machine learning libraries with a lightweight REST API backend to deliver real-time environmental impact predictions[cite: 49, 51].

### Core Data & Feature Engineering
[cite_start]Raw process, material, energy, and transport inputs are mathematically transformed (e.g., logarithmic transformations) to derive meaningful circularity features[cite: 90, 108].
* [cite_start]**Pandas & NumPy:** For structural data manipulation and high-performance numerical preprocessing[cite: 42, 124].
* [cite_start]**Standard Scaler:** To normalize input features[cite: 136].

### Machine Learning Framework
[cite_start]The prediction engine operates via a dual-phase ML pipeline built to estimate missing parameters and predict both resource consumption and CO2 emissions[cite: 35, 100, 102].
* [cite_start]**Framework:** Scikit-Learn[cite: 132].
* [cite_start]**Primary Models:** `HistGradientBoostingRegressor` and `MultiOutputRegressor` (for simultaneous Energy and Water prediction)[cite: 36, 100].
* [cite_start]**Ensemble Modeling:** Random Forest[cite: 133].
* [cite_start]**Optimization:** `RandomizedSearchCV` combined with SciPy distributions for efficient hyperparameter tuning and cross-validation[cite: 44, 135].

### Model Persistence & Deployment
[cite_start]To ensure fast and consistent backend deployment without the need for constant retraining, the models and their feature mappings are serialized[cite: 110, 111].
* [cite_start]**Joblib:** Used for bundling the trained models into lightweight, portable files[cite: 111, 139].

### Backend & API Architecture
[cite_start]The application interface is served via a REST API that handles JSON inputs, runs the two-stage predictions, and scales results instantly[cite: 104, 105].
* [cite_start]**Framework:** Flask[cite: 49, 142].
* [cite_start]**Cross-Origin Support:** Flask-CORS[cite: 51, 142].

---

## 📊 Model Performance Metrics
[cite_start]The AI models were evaluated on rigorous metrics to ensure the reliability of the Life Cycle Assessment estimations[cite: 110].
* [cite_start]**Evaluation Metrics Used:** Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)[cite: 143].
* [cite_start]**Combined Overall Accuracy:** **85.97%**[cite: 257].
    * [cite_start]*MAE-based Accuracy:* 88.98%[cite: 255].
    * [cite_start]*R²-based Accuracy:* 82.95%[cite: 256].

---

## 🌍 Core Impact
[cite_start]By estimating missing parameters and automating complex LCA workflows, this tool helps industries increase recycled content by 20-30%[cite: 218]. [cite_start]It significantly reduces the manual effort required by analysts, saving roughly 30-40% of the time usually spent imputing missing LCI data, and provides direct, AI-driven recommendations to enhance sustainability branding[cite: 162, 166]. 
