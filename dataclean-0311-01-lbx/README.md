# Industrial Feature Engineering Pipeline: Home Credit Default Risk

## Project Overview
This repository delivers a robust, production-standard data preprocessing and feature engineering pipeline for the **Home Credit Default Risk** competition. 

The core mission of this project is to solve the **1:N relational data challenge**. By transforming 7 distinct, high-volume tables into a single "Industrial Wide-Table," we provide a high-signal dataset optimized for Gradient Boosting Decision Trees (GBDTs) such as **LightGBM** and **XGBoost**.



---

## Key Engineering Strategies

### 1. High-Performance Aggregation Engine
We tackle the complexity of many-to-one relationships (e.g., `Bureau`, `Previous Applications`, `Installments`) by flattening them into static client-level profiles. 
* **Statistical Depth**: Beyond basic averages, we utilize `max`, `min`, `sum`, and `variance` to capture the volatility of historical financial behavior.
* **Feature Traceability**: A strict naming convention is enforced (e.g., `BU_`, `PREV_`, `INS_` prefixes) to ensure data lineage is clear during model interpretation.

### 2. Domain-Driven Feature Derivation
Rather than relying on brute-force polynomial crossings, we prioritize **Financial Proxy Indicators**:
* **Repayment Capacity**: Engineered ratios such as *Annuity-to-Credit* and *Annuity-to-Income*.
* **Behavioral Latency**: Precise calculation of **DPD (Days Past Due)** and **DBD (Days Before Due)** to isolate risk signals from trustworthy payment patterns.
* **Anomaly Masking**: Transformed system-specific outliers (e.g., the `365243` value in `DAYS_EMPLOYED`) into meaningful binary flags.

### 3. Aggressive Memory Optimization
Engineered specifically for memory-constrained environments (e.g., Kaggle Kernels or 16GB RAM workstations):
* **Type Downcasting**: Dynamically scanning value ranges to convert `float64` → `float32` and `int64` → `int8/16/32`.
* **Footprint Reduction**: Achieved a **70-80% reduction** in memory usage.
* **De-fragmentation**: Utilizing `.copy()` and explicit `gc.collect()` to maintain a contiguous memory layout and stable runtime.

---

## Tech Stack
* **Language**: Python 3.8+
* **Core Libraries**: Pandas, NumPy
* **Preprocessing**: Scikit-Learn (LabelEncoding, One-Hot Encoding)
* **Optimization**: Custom Memory Downcasting Script

## Pipeline Deliverables
* **Data Integration**: Successfully flattened 7 relational tables into 1 master matrix.
* **Dimensionality**: Expanded from 122 raw variables to **500+ high-signal features**.
* **Model Readiness**: 100% numerical format, fully encoded, and optimized for immediate training.

