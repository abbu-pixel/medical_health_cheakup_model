🧠 Medical Health Checkup Prediction – README.md
# 🧠 Medical Health Checkup Prediction (MLOps End-to-End)

An end-to-end **MLOps project** that predicts a person’s medical health status using physiological data.  
The project integrates **Machine Learning**, **MLflow tracking**, **Data Versioning (DVC)**, **Drift Monitoring (Evidently)**, and **Automated CI/CD (GitHub Actions + Render)**.

---

## 🚀 Project Overview

This system predicts whether a patient is **Healthy** or **Needs Attention** based on their medical measurements.  
It demonstrates the **complete MLOps lifecycle** — from data processing and model training to deployment, monitoring, and auto-retraining.

---

## 🧩 Architecture Overview



Raw Data → DVC Tracking → Model Training (MLflow)
↓
Drift Monitoring (Evidently)
↓
CI/CD Pipeline (GitHub Actions)
↓
Auto Retrain → Push Models + Reports
↓
Flask API Deployment (Render)
↓
🌐 Web App for Prediction + Monitoring


---

## ⚙️ Tech Stack

| Category | Tools & Frameworks |
|-----------|--------------------|
| **Programming** | Python 3.12 |
| **ML Frameworks** | Scikit-learn, XGBoost, TensorFlow |
| **Experiment Tracking** | MLflow |
| **Data Versioning** | DVC |
| **Monitoring** | Evidently AI |
| **Automation (CI/CD)** | GitHub Actions |
| **Deployment** | Flask + Gunicorn on Render Cloud |
| **Frontend** | HTML, CSS, JavaScript |

---

## 📁 Project Structure



medical_health_cheakup_model/
│
├── .github/workflows/
│ └── mlops_pipeline.yml ← CI/CD workflow
│
├── data/ ← Raw & processed data
│ └── processed/
│ ├── X_train.csv
│ └── X_test.csv
│
├── models/ ← Trained models (.pkl)
│ └── RandomForest_model.pkl
│
├── reports/ ← Evaluation & drift reports
│ ├── confusion_matrix.png
│ ├── metrics.json
│ └── evidently_drift_report.html
│
├── src/ ← Core source code
│ ├── app.py ← Flask API (predict + monitor)
│ ├── train_with_mlflow.py ← Model training + MLflow logging
│ └── monitor.py ← Evidently drift monitoring
│
├── static/ ← Web UI
│ └── index.html
│
├── requirements.txt
├── Dockerfile
├── dvc.yaml
└── README.md


---

## 🔄 CI/CD Workflow

Your entire pipeline is automated through **GitHub Actions**.

**Trigger:** Every push to `main` branch

**Steps in pipeline:**
1. Checkout repository  
2. Set up Python environment  
3. Install dependencies  
4. Pull data from DVC  
5. Train models and log metrics to MLflow  
6. Generate Evidently drift report  
7. Commit trained model and reports back to repo  
8. Trigger Render deployment via Deploy Hook  

✅ *Every push = auto retrain, re-monitor, and redeploy.*

---

## 📊 Model Monitoring

The pipeline uses **Evidently AI** to monitor:
- **Data Drift**
- **Data Quality**
- **Feature Distribution changes**

### Reports Generated:
- `reports/evidently_drift_report.html` → Full drift report  
- `reports/metrics.json` → Accuracy & performance metrics  
- `/monitor` endpoint → Live dashboard view  

---

## 🌐 Deployment (Render)

The Flask app is deployed on Render and includes:
- `/` → Web interface (user form for prediction)  
- `/predict` → POST endpoint for model predictions  
- `/monitor` → Shows live Evidently drift report  

🔗 **Live Demo:** [https://medical-cheakup.onrender.com](https://medical-cheakup.onrender.com)

---

## 🧠 Example Input Features

| Feature | Description |
|----------|-------------|
| Age | Age in years |
| Gender | Male / Female |
| Heart Rate | Beats per minute |
| Temperature | Body temperature (°C) |
| Oxygen Level | SpO₂ percentage |
| Glucose Level | mg/dL |
| Cholesterol | mg/dL |
| Systolic BP | mmHg |
| Diastolic BP | mmHg |

---

## 🩺 Example Output

```json
{
  "prediction": "Healthy ✅"
}


or

{
  "prediction": "Needs Attention ⚠️"
}

🧾 MLflow Experiment Tracking

All training runs are logged to local MLflow:

Parameters (e.g. n_estimators, depth)

Metrics (accuracy, loss)

Models (stored in mlruns/ directory)

Best model automatically exported to /models

🧠 Monitoring with Evidently
python src/monitor.py


Generates reports/evidently_drift_report.html

Saves drift summary as JSON

Integrated into CI/CD workflow automatically

🧰 Local Setup (Run Manually)
# Clone repo
git clone https://github.com/abbu-pixel/medical_health_cheakup_model.git
cd medical_health_cheakup_model

# Install dependencies
pip install -r requirements.txt

# Run training
python src/train_with_mlflow.py

# Run drift monitoring
python src/monitor.py

# Start Flask app
python src/app.py


Access the app at: http://localhost:5000

🧩 Deployment Automation

Render automatically redeploys when:

A new model is pushed

CI/CD workflow triggers curl $RENDER_DEPLOY_HOOK

🏁 Key Achievements

✅ End-to-end ML lifecycle automation
✅ Continuous Integration & Deployment
✅ Data drift detection & monitoring
✅ Model retraining with version control
✅ Live deployed health prediction system

👨‍💻 Author

Abbu Rahman
MLOps Engineer | ML Developer | Cloud Enthusiast
📧 abbura*****@gmail.com
🌐 GitHub Profile

🏆 Summary

This project automates the entire ML workflow — from data versioning and training to model monitoring and deployment — using MLflow, DVC, Evidently, GitHub Actions, and Render Cloud.


---

Would you like me to **add visuals (architecture diagram + pipeline image links)** at the top of this README (I can generate and provide them so you can upload to GitHub and link)?  
It makes it look **industry-grade**, like a professional portfolio project.
