# -*- coding: utf-8 -*-
# time:2024/1/5 20:57
# file app.py
# author: czy
# email: 1060324818@qq.com

from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import numpy as np

app = Flask(__name__)

# 实例化模型
input_size = 13  # 请替换成实际的输入大小

@app.before_request
def before_request():
    global model
    model = torch.nn.Linear(input_size,1)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

@app.after_request
def after_request(response):
    global model
    del model
    return response

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 获取请求中的数据
        data = request.get_json(force=True)
        # 将数据转换为numpy数组
        input_data = np.array(data['input'], dtype=np.float32).reshape(1, -1)
        # 将numpy数组转换为PyTorch张量
        input_tensor = torch.tensor(input_data)
        # 使用模型进行预测
        with torch.no_grad():
            prediction = model(input_tensor)
        # 将预测结果转换为列表（因为numpy数组不能被序列化为JSON）
        prediction = prediction.numpy().tolist()
        # 返回预测结果
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
