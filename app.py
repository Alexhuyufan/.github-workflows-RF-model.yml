from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib
import os  # 添加这行代码来导入 os 模块

app = Flask(__name__)

# 加载模型文件
model = joblib.load('Simplified_RF.pkl')  # 请确保模型文件路径正确

# 设置 host 和 port
host = os.getenv('HOST', '0.0.0.0')  # 默认是 0.0.0.0
port = int(os.getenv('PORT', 5005))   # 默认是 5000


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 从表单中获取输入值
    inputs = {
        'age': request.form.get('age', type=int),
        'cci': request.form.get('cci', type=int),
        'sapsii': request.form.get('sapsii', type=int),
        'gcs_min': request.form.get('gcs_min', type=int),
        'rdw_max': request.form.get('rdw_max', type=float),
        'bilirubin_total_max': request.form.get('bilirubin_total_max', type=float),
        'spo2_min': request.form.get('spo2_min', type=float),
        'weight': request.form.get('weight', type=float),
        'temperature_max': request.form.get('temperature_max', type=float),
        'platelets_min': request.form.get('platelets_min', type=float),
        'septic_shock': request.form.get('septic_shock', type=int),
        'surgical': request.form.get('surgical', type=int),
        'inr_max': request.form.get('inr_max', type=float),
        'first_careunit_2': request.form.get('first_careunit_2', type=int),
        'chloride_max': request.form.get('chloride_max', type=float)
    }

    # 将输入数据转换为模型所需的格式
    features = [[inputs['age'], inputs['cci'], inputs['sapsii'], inputs['gcs_min'], inputs['rdw_max'],
                 inputs['bilirubin_total_max'], inputs['spo2_min'], inputs['weight'], inputs['temperature_max'],
                 inputs['platelets_min'], inputs['septic_shock'], inputs['surgical'], inputs['inr_max'],
                 inputs['first_careunit_2'], inputs['chloride_max']]]

    # 进行预测
    prediction = model.predict_proba(features)[0][1] * 100  # 预测死亡概率

    return render_template('index.html', prediction={'death_probability': round(prediction, 2)})
    
if __name__ == '__main__':
    app.run(port=5005, debug=True)
