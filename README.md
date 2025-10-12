<h1 align="center">🌊 Industry-Standard Flood Prediction Model</h1>
<h3 align="center">An End-to-End Machine Learning Pipeline for Predicting Flood Risk in Pakistan</h3>

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/2/29/2022_Pakistan_Floods_Satellite_Image.jpg" width="700" alt="Satellite view of flooded river delta">
</p>

---

## 📘 Overview
This project provides a **robust, end-to-end machine learning pipeline** for predicting flood risk in Pakistan.  
It leverages **historical weather, terrain, and hydrological data** to train a powerful **XGBoost model**, moving beyond simple thresholds to learn the **complex environmental patterns** that precede flood events.  

The project is designed with **scalability**, **maintainability**, and **clear modular structure**, adhering to **industry best practices**.

---

## 🚀 Key Features

- **🧭 User-Friendly Interface** – A simple menu-driven script (`main.py`) acts as the single entry point to control the entire pipeline.  
- **📑 Phased Execution** – Three distinct, runnable phases:  
  1. Data collection & processing  
  2. Model training  
  3. Prediction demo  
- **🌦 Intelligent Data Collection** – Automatically fetches and updates historical weather & terrain data, downloading only new entries on subsequent runs.  
- **🛰 Ground Truth Integration** – Trains on **real historical flood events**, ensuring the model learns from actual precursors, not assumptions.  
- **🧮 Advanced Feature Engineering** – Incorporates **time-series**, **topographical**, and **hydrological** features (e.g., rainfall averages, elevation, slope, glacial melt proxies).  
- **⚙️ Powerful ML Model** – Uses **XGBoost Classifier**, renowned for its accuracy on complex tabular datasets.  
- **📁 Modular Structure** – Organized into a clean `src/` directory for each part of the pipeline.  
- **🪵 Centralized Logging** – Structured logs saved to `logs/app.log` for easy debugging and reproducibility.  

---

## ⚙️ Installation

Clone the repository and set up a virtual environment:

```bash
# Clone the project
git clone <your-repository-url>
cd flood-prediction-project

# Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install all required dependencies
pip install -r requirements.txt
