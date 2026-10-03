# Attendance and Achievement Analysis

**Machine Learning Case Study 49**

Perni Bharath Raghavendra · Mark Zuckerberg Cohort · Roll number: 150096724139

[GitHub repository](https://github.com/bharath-541/Attendance-and-Exam-Score-Prediction-Using-Linear-Regression) | [Live Streamlit application](https://bharath-attendance-exam-score.streamlit.app/)

Held-out test results: MAE **1.016**, RMSE **1.876**, R² **0.734**. The report explains the metrics and limitations.

Open analysis.ipynb in Jupyter or Google Colab and run the cells in order.
The notebook uses ten inputs and multiple linear regression. It includes categorical encoding, evaluation, two plots, one example prediction and model export.

Dataset: https://www.kaggle.com/datasets/lainguyn123/student-performance-factors
The records are synthetic. The notebook reads the included CSV or downloads the public dataset when the CSV is absent.

Run the Streamlit application:

```sh
pip install -r requirements.txt
streamlit run app.py
```

Files: analysis.ipynb, data/StudentPerformanceFactors.csv, data/source.json, report.docx, app.py, model.joblib, results.json, requirements.txt and this README.
