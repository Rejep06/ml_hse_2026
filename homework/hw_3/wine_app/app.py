import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

ROOT = Path(__file__).parent
st.set_page_config(page_title='Wine Quality', page_icon='🍷', layout='wide')


@st.cache_resource
def load_model():
    return joblib.load(ROOT / 'model.joblib')


@st.cache_data
def load_data():
    return pd.read_csv(ROOT / 'test.csv')


@st.cache_data
def test_proba(X, _model):
    return _model.predict_proba(X)[:, 1]


model = load_model()
data = load_data()
X = data.drop(columns='target')
y = data['target']
p = test_proba(X, model)
default = json.loads((ROOT / 'config.json').read_text())['threshold']

st.title('🍷 Рекомендация вин')
st.write('Предсказываем вероятность оценки вина не ниже 7.')
threshold = st.sidebar.slider('Порог рекомендации', 0.0, 1.0, float(default), 0.001)
tab1, tab2, tab3, tab4 = st.tabs(['Одно вино', 'Метрики', 'CSV', 'Коэффициенты'])

with tab1:
    st.subheader('Оценить одно вино')
    values = {}
    cols = st.columns(3)
    for i, name in enumerate(X.columns.drop('is_red')):
        column = X[name]
        step = 0.0001 if column.max() - column.min() < 0.2 else 0.01
        with cols[i % 3]:
            values[name] = st.number_input(name, float(column.min()), float(column.max()),
                                           float(column.median()), step=step)
    values['is_red'] = int(st.toggle('Красное вино', value=False))
    wine = pd.DataFrame([values])[X.columns]
    prob = model.predict_proba(wine)[0, 1]
    st.metric('Вероятность хорошего вина', f'{prob:.1%}')
    st.write('**Рекомендовать**' if prob >= threshold else '**Не рекомендовать**')

with tab2:
    pred = (p >= threshold).astype(int)
    precision = metrics.precision_score(y, pred, zero_division=0)
    recall = metrics.recall_score(y, pred, zero_division=0)
    f1 = metrics.f1_score(y, pred, zero_division=0)
    a, b, c = st.columns(3)
    a.metric('Precision', f'{precision:.3f}')
    b.metric('Recall', f'{recall:.3f}')
    c.metric('F1', f'{f1:.3f}')

    cm = metrics.confusion_matrix(y, pred)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    metrics.ConfusionMatrixDisplay(cm).plot(ax=ax[0], colorbar=False)
    fpr, tpr, _ = metrics.roc_curve(y, p)
    ax[1].plot(fpr, tpr)
    ax[1].scatter(cm[0, 1] / cm[0].sum(), recall, color='black')
    ax[1].set(xlabel='FPR', ylabel='TPR', title='ROC')
    pr, re, _ = metrics.precision_recall_curve(y, p)
    ax[2].plot(re, pr)
    ax[2].axhline(y.mean(), ls='--', color='gray')
    ax[2].scatter(recall, precision, color='black')
    ax[2].set(xlabel='Recall', ylabel='Precision', title='PR')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.info('Выше порог строже отбор и больше пропусков хороших вин. Ниже порог больше рекомендаций, но и больше ложных срабатываний.')

with tab3:
    uploaded = st.file_uploader('Загрузите CSV без target', type='csv')
    if uploaded is not None:
        batch = pd.read_csv(uploaded)
        missing = X.columns.difference(batch.columns)
        if len(missing):
            st.error('Не хватает столбцов: ' + ', '.join(missing))
        else:
            prob = model.predict_proba(batch[X.columns])[:, 1]
            batch['probability_good'] = prob
            batch['recommend'] = (prob >= threshold).astype(int)
            st.dataframe(batch)
            st.download_button('Скачать результат', batch.to_csv(index=False),
                               'wine_predictions.csv', 'text/csv')

with tab4:
    coefs = pd.Series(model[-1].coef_[0], index=X.columns).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    coefs.plot.barh(ax=ax)
    ax.set(xlabel='Вес после стандартизации', title='Коэффициенты логистической регрессии')
    st.pyplot(fig)
    plt.close(fig)
    st.write('За хорошее вино:', ', '.join(coefs.tail(3).index))
    st.write('Против:', ', '.join(coefs.head(3).index))
