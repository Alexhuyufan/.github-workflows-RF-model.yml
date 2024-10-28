from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.externals import joblib

app = Flask(__name__)

# 加载你的模型文件
model = joblib.load('Simplified_RF.pkl')  # 确保这里是正确的模型文件路径

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 获取表单数据并进行预处理
    age = request.form.get('age', type=float)
    cci = request.form.get('cci', type=float)
    sapsii = request.form.get('sapsii', type=float)
    gcs_min = request.form.get('gcs_min', type=float)
    rdw_max = request.form.get('rdw_max', type=float)
    bilirubin_total_max = request.form.get('bilirubin_total_max', type=float)
    spo2_min = request.form.get('spo2_min', type=float)
    weight = request.form.get('weight', type=float)
    temperature_max = request.form.get('temperature_max', type=float)
    platelets_min = request.form.get('platelets_min', type=float)
    septic_shock = request.form.get('septic_shock', type=int)
    surgical = request.form.get('surgical', type=int)
    inr_max = request.form.get('inr_max', type=float)
    first_careunit_2 = request.form.get('first_careunit_2', type=int)
    chloride_max = request.form.get('chloride_max', type=float)
    
    # 构建 DataFrame
    data = pd.DataFrame([
        [age, cci, sapsii, gcs_min, rdw_max, bilirubin_total_max,
         spo2_min, weight, temperature_max, platelets_min,
         septic_shock, surgical, inr_max, first_careunit_2,
         chloride_max]
    ], columns=[
        'age', 'cci', 'sapsii', 'gcs_min', 'rdw_max', 'bilirubin_total_max',
        'spo2_min', 'weight', 'temperature_max', 'platelets_min',
        'septic_shock', 'surgical', 'inr_max', 'first_careunit_2',
        'chloride_max'
    ])
    
    # 使用模型进行预测
    prediction = model.predict(data)
    
    # 将预测结果传递给模板
    return render_template('index.html', prediction_text=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)