from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load model
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pkl')
model = pickle.load(open(model_path, 'rb'))

# Load processed dataset
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed_data.csv')
df = pd.read_csv(data_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        flight_no = int(request.form['flight_no'])

        # Check column
        if 'FlightNum' not in df.columns:
            return render_template('index.html', prediction_text="❌ FlightNum column missing!")

        # Find flight row
        row = df[df['FlightNum'] == flight_no]

        if row.empty:
            return render_template('index.html', prediction_text="❌ Flight not found!")

        # 🔥 IMPORTANT FIX
        # Drop both target + FlightNum (ID should not go into model)
        X = row.drop(['Delayed', 'FlightNum'], axis=1, errors='ignore')

        prediction = model.predict(X)

        result = "✈️ Delayed" if prediction[0] == 1 else "✅ On Time"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)