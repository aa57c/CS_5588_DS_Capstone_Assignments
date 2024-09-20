from flask import Flask, request, jsonify

from DS_Capstone_Hands_On_Assignment_9_19.DS_Capstone_Hands_On_Assignment_9_19 import combine_models

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Assume data is received as JSON (stock data, Reddit comment)
    data = request.json
    prediction = combine_models(data['stock'], data['reddit'])
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)