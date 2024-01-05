# -*- coding: utf-8 -*-
# time:2024/1/5 20:57
# file app.py.py
# outhor:czy
# email:1060324818@qq.com

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# 加载模型
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 获取请求中的数据
        data = request.get_json(force=True)
        # 将数据转换为numpy数组
        input_data = np.array(data['input']).reshape(1, -1)
        # 使用模型进行预测
        prediction = model.predict(input_data)
        # 将预测结果转换为列表（因为numpy数组不能被序列化为JSON）
        prediction = prediction.tolist()
        # 返回预测结果
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)

