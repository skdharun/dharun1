from flask import Flask, request, render_template_string
import joblib
import pandas as pd
import os

app = Flask(__name__)
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1.0")
model = joblib.load("model.pkl")

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Student Result Predictor</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f2f5fa;
            margin: 0;
            padding: 30px 15px;
        }
        .card {
            max-width: 480px;
            margin: 30px auto;
            background: white;
            padding: 28px;
            border-radius: 14px;
            box-shadow: 0 4px 20px #0001;
        }
        h1 { color: #173b70; }
        label { display: block; margin-top: 16px; }
        input {
            box-sizing: border-box;
            width: 100%;
            padding: 11px;
            margin-top: 6px;
        }
        button {
            margin-top: 22px;
            padding: 12px;
            width: 100%;
            border: 0;
            border-radius: 6px;
            background: #1769d2;
            color: white;
            font-size: 16px;
            cursor: pointer;
        }
        .result {
            margin-top: 20px;
            padding: 14px;
            background: #e9f5e9;
            border-radius: 6px;
            font-weight: bold;
        }
        .version { color: #555; font-size: 14px; }
        .error { color: #b91c1c; }
    </style>
</head>
<body>
<div class="card">
    <h1>Student Result Predictor</h1>
    <p class="version">Model Version: {{ version }}</p>

    <form method="POST">
        <label>Study Hours (0–24)</label>
        <input type="number" name="study_hours"
               min="0" max="24" step="0.1"
               value="{{ hours }}" required>

        <label>Attendance (0–100%)</label>
        <input type="number" name="attendance"
               min="0" max="100" step="0.1"
               value="{{ attendance }}" required>

        <label>Internal Marks (0–100)</label>
        <input type="number" name="internal_marks"
               min="0" max="100" step="0.1"
               value="{{ marks }}" required>

        <button type="submit">Predict Result</button>
    </form>

    {% if result %}
    <div class="result">Predicted Result: {{ result }}</div>
    {% endif %}

    {% if error %}
    <p class="error">{{ error }}</p>
    {% endif %}
</div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None
    hours, attendance, marks = 5, 80, 70

    if request.method == "POST":
        try:
            hours = float(request.form["study_hours"])
            attendance = float(request.form["attendance"])
            marks = float(request.form["internal_marks"])

            if not (0 <= hours <= 24):
                raise ValueError("Study hours must be between 0 and 24.")
            if not (0 <= attendance <= 100):
                raise ValueError("Attendance must be between 0 and 100.")
            if not (0 <= marks <= 100):
                raise ValueError("Internal marks must be between 0 and 100.")

            features = pd.DataFrame([{
                "study_hours": hours,
                "attendance": attendance,
                "internal_marks": marks
            }])

            prediction = model.predict(features)[0]
            result = "Pass" if prediction == 1 else "Fail"

        except (ValueError, KeyError) as exc:
            error = str(exc) or "Please enter valid numbers."

    return render_template_string(
        HTML, version=MODEL_VERSION, result=result,
        error=error, hours=hours,
        attendance=attendance, marks=marks
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
