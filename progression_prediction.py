"""
Created on Wed Jan 18 2023

@author: k
"""
# import packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
import shap
import lightgbm
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import scikitplot as skplt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, \
    confusion_matrix, accuracy_score, roc_auc_score, auc, brier_score_loss
import streamlit as st
import warnings
warnings.filterwarnings("ignore")  # ignore warnings
pd.options.display.max_columns = None
pd.options.display.max_rows = None
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams.update({'font.size': 16})

st.set_page_config(
    page_title="Prediction model of CRC tumor progression",
    layout='wide',
    page_icon=':male-doctor:️'
)
# dashboard title
# st.title("Real-Time Fraud Detection Dashboard")
st.markdown("<h1 style='text-align:center; color: black'>ML："
            "Prediction of CRC tumor progression <br> in 18 months</h1>", unsafe_allow_html=True)


# side-bar
def user_input_features():
    st.sidebar.header('Make a prediction')
    st.sidebar.write('User input parameters  below :arrow_down: ')
    a1 = st.sidebar.selectbox('Tumor stage', (0, 1, 2, 3, 4, 5, 6, 7), help=(
'''for 0= stage I ;
    1= stage II ;
    2= stage IIIa ;
    3= stage IIIb ;
    4= stage IIIc ;
    5= stage IVa ;
    6= stage IVb ;
    7= stage IVc '''))
    a2 = st.sidebar.selectbox('Completion of chemotherapy', (1, 0), help='1=completed ; 0=uncompleted')
    a3 = st.sidebar.number_input('CEA (ng/ml)', min_value=0.00)
    a4 = st.sidebar.number_input('CA125 (kU/L)', min_value=0.00)
    a5 = st.sidebar.number_input('CA50 (μg/L)', min_value=0.00)
    a6 = st.sidebar.number_input('FER (μg/L)', min_value=0.00)
    a7 = st.sidebar.number_input('hsCRP (mg/L)', min_value=0.00)
    a8 = st.sidebar.number_input('EOS (×10^9/L)', min_value=0.00)
    a9 = st.sidebar.number_input('PLT (×10^9/L)', min_value=0.00)
    a10 = st.sidebar.number_input('AFU (U/L)', min_value=0.00)
    a11 = st.sidebar.number_input('TG (mmol/L)', min_value=0.00)
    a12 = st.sidebar.number_input('HDL-c (mmol/L)', min_value=0.00)
    a13 = st.sidebar.number_input('CK (U/L)', min_value=0.00)
    a14 = st.sidebar.number_input('Cl (mmol/L)', min_value=0.00)
    a15 = st.sidebar.number_input('Na (mmol/L)', min_value=0.00)

    output = [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15]
    return output


outputdf = user_input_features()

GSXGB = xgb.XGBClassifier()
GSXGB.load_model('GSXGB_crc')

st.title("**Make predictions** in :blue[_real time_] :male-doctor:")
' '
shapdatadf = pd.read_excel("crcdataf.xlsx")
outputdf = pd.DataFrame([outputdf], columns=shapdatadf.columns)


p1 = GSXGB.predict(outputdf)
p2 = GSXGB.predict_proba(outputdf)

placeholder1 = st.empty()
st.subheader(':blue[Step 1.] User input parameters in the sidebar :arrow_left: or below ⬇️')
st.write(outputdf)

st.subheader(':blue[Step 2.] Here is the outcome')
placeholder2 = st.empty()
with placeholder2.container():
    f1, f2 = st.columns([1, 1.5], gap='small')
    with f1:
        st.write(f':red_circle: Predicted class: :green[**_{p1}_**]')
        ':zero: means the tumor :blue[**won\'t  progress**] in ***_18 months_***,'
        ':one: means the tumor :red[will progress] in ***_18 months_***'
        st.write(':red_circle: Predicted class Probability : ')
        st.write(p2)
        st.bar_chart(p2, width=150, use_container_width=False)
    with f2:
        ':red_circle: Feature importance plot'
        explainer = shap.Explainer(GSXGB)
        shap_values = explainer(outputdf)

        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.plots.waterfall(shap_values[0])
        st.pyplot(bbox_inches='tight')
        ':red[**Note**]: The bars are the SHAP values for each feature. The feature values are show in gray ' \
        'to the left of the feature names. For bars in :red[red] might :red[_contribute to tumor progression_] ' \
        '，while bars in :blue[blue] might :blue[_not_].'
