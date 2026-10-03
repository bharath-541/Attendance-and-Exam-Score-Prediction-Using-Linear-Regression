from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Attendance and Achievement', page_icon='📚')
st.title('Attendance and Achievement')
st.caption('ML Case Study 49 · Perni Bharath Raghavendra · Mark Zuckerberg Cohort')
st.markdown('[GitHub repository](https://github.com/bharath-541/Attendance-and-Exam-Score-Prediction-Using-Linear-Regression)')
st.write('Estimate an exam score using attendance and nine other inputs.')
st.caption('Demonstration using synthetic student records. Predictions are estimates.')

@st.cache_resource
def load_model():
    return joblib.load(ROOT / 'model.joblib')

bundle = load_model()
with st.form('student'):
    left, right = st.columns(2)
    with left:
        attendance = st.slider('Attendance (%)', 60, 100, 80)
        hours = st.number_input('Hours studied', 1, 44, 20)
        previous = st.number_input('Previous score', 50, 100, 75)
        sleep = st.number_input('Sleep hours', 4, 10, 7)
        tutoring = st.number_input('Tutoring sessions', 0, 8, 1)
    with right:
        physical = st.number_input('Physical activity', 0, 6, 3)
        resources = st.selectbox('Access to resources', ['Low', 'Medium', 'High'], index=1)
        motivation = st.selectbox('Motivation level', ['Low', 'Medium', 'High'], index=1)
        internet = st.selectbox('Internet access', ['Yes', 'No'])
        extra = st.selectbox('Extracurricular activities', ['No', 'Yes'])
    submitted = st.form_submit_button('Predict exam score')

if submitted:
    student = pd.DataFrame([{
        'Attendance': attendance, 'Hours_Studied': hours, 'Previous_Scores': previous,
        'Sleep_Hours': sleep, 'Tutoring_Sessions': tutoring, 'Physical_Activity': physical,
        'Access_to_Resources': resources, 'Motivation_Level': motivation,
        'Internet_Access': internet, 'Extracurricular_Activities': extra
    }])
    score = bundle['model'].predict(bundle['encoder'].transform(student))[0]
    st.metric('Estimated exam score', f'{score:.1f}')
    if not 0 <= score <= 100:
        st.warning('This prediction is outside the assumed score range of 0–100.')

st.divider()
st.write('Linear regression performance on the held-out test set')
results = json.loads((ROOT / 'results.json').read_text())
columns = st.columns(3)
for column, name in zip(columns, ['MAE', 'RMSE', 'R2']):
    column.metric(name, f'{results[name]:.3f}')
st.caption('Lower MAE and RMSE are better; higher R² is better. Unusual high scores may have larger errors.')
