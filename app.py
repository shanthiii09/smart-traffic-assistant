from flask import Flask, render_template, request
from smart_traffic_assistant import predict_traffic   # ✅ direct import

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    source = request.form['source']
    destination = request.form['destination']

    try:
        result = predict_traffic(source, destination)  # ✅ call function directly
        return render_template('index.html', result=result)
    except Exception as e:
        return render_template('index.html', result=f"❌ Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
