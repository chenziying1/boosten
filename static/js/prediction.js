// prediction.js
async function submitForm(event) {
    event.preventDefault();

    const inputElements = document.getElementById('predictionForm').elements;
    const inputData = Array.from(inputElements).reduce((data, input) => {
        if (input.id) {
            data[input.id.toLowerCase()] = parseFloat(input.value);
        }
        return data;
    }, {});

    // 发送POST请求给Flask后端
    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ input: Object.values(inputData) }),
    });

    // 解析并显示结果
    const resultElement = document.getElementById('result');
    const result = await response.json();
    resultElement.innerText = `Predicted Price: $${result.prediction[0].toFixed(2)}`;
}
