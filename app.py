from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

app = Flask(__name__)

# =========================
# PATH FIX (IMPORTANT for Render)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "models", "model.pkl")
data_path = os.path.join(BASE_DIR, "data", "processed_data.csv")

# =========================
# LOAD MODEL & DATA
# =========================
model = pickle.load(open(model_path, 'rb'))
df = pd.read_csv(data_path)

# =========================
# ROUTES
# =========================
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

        # Drop unnecessary columns
        X = row.drop(['Delayed', 'FlightNum'], axis=1, errors='ignore')

        prediction = model.predict(X)

        result = "✈️ Delayed" if prediction[0] == 1 else "✅ On Time"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
