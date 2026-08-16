# from flask import Flask, render_template, request
# import numpy as np
# import joblib
# import os

# app = Flask(__name__)

# # ---------------------------------------------------------
# # Load trained model and scaler
# # ---------------------------------------------------------

# # MODEL_PATH = "heart_model.pkl"
# # SCALER_PATH = "scaler.pkl"
# MODEL_PATH = "models/heart_model.pkl"
# SCALER_PATH = "models/scaler.pkl"
# model = joblib.load(MODEL_PATH)
# scaler = joblib.load(SCALER_PATH)


# # ---------------------------------------------------------
# # Home / Prediction Route
# # ---------------------------------------------------------

# @app.route("/", methods=["GET", "POST"])
# def home():

#     prediction = None
#     probability = None

#     if request.method == "POST":

#         try:

#             # -------------------------------------------------
#             # Get 13 inputs from HTML form
#             # -------------------------------------------------

#             age = float(request.form["age"])
#             sex = float(request.form["sex"])
#             cp = float(request.form["cp"])
#             trestbps = float(request.form["trestbps"])
#             chol = float(request.form["chol"])
#             fbs = float(request.form["fbs"])
#             restecg = float(request.form["restecg"])
#             thalach = float(request.form["thalach"])
#             exang = float(request.form["exang"])
#             oldpeak = float(request.form["oldpeak"])
#             slope = float(request.form["slope"])
#             ca = float(request.form["ca"])
#             thal = float(request.form["thal"])


#             # -------------------------------------------------
#             # Keep EXACT same feature order as training
#             # -------------------------------------------------

#             features = np.array([[
#                 age,
#                 sex,
#                 cp,
#                 trestbps,
#                 chol,
#                 fbs,
#                 restecg,
#                 thalach,
#                 exang,
#                 oldpeak,
#                 slope,
#                 ca,
#                 thal
#             ]])


#             # -------------------------------------------------
#             # Apply saved scaler
#             # -------------------------------------------------

#             scaled_features = scaler.transform(features)


#             # -------------------------------------------------
#             # Prediction
#             # -------------------------------------------------

#             prediction = int(model.predict(scaled_features)[0])


#             # -------------------------------------------------
#             # Prediction probability
#             # -------------------------------------------------

#             if hasattr(model, "predict_proba"):

#                 probabilities = model.predict_proba(
#                     scaled_features
#                 )[0]

#                 probability = round(
#                     float(probabilities[1]) * 100,
#                     1
#                 )

#             else:

#                 probability = 50.0


#         except Exception as e:

#             print("Prediction Error:", e)

#             return render_template(
#                 "index.html",
#                 error="Unable to process the provided information. Please check your inputs."
#             )


#     return render_template(
#         "index.html",
#         prediction=prediction,
#         probability=probability
#     )


# # ---------------------------------------------------------
# # Run Flask
# # ---------------------------------------------------------

# if __name__ == "__main__":

#     app.run(
#         debug=True,
#         host="127.0.0.1",
#         port=5000
#     )

from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import joblib
import sqlite3
from datetime import datetime
import os

app = Flask(__name__)

MODEL_PATH = "models/heart_model.pkl"
SCALER_PATH = "models/scaler.pkl"
DATABASE = "heart_history.db"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# =========================================================
# DATABASE
# =========================================================

def init_db():

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date_time TEXT,
            age INTEGER,
            sex INTEGER,
            prediction INTEGER,
            probability REAL
        )
    """)

    conn.commit()
    conn.close()


init_db()


# =========================================================
# SAVE PREDICTION
# =========================================================

def save_prediction(age, sex, prediction, probability):

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions
        (date_time, age, sex, prediction, probability)
        VALUES (?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        age,
        sex,
        prediction,
        probability
    ))

    conn.commit()
    conn.close()


# =========================================================
# GET HISTORY
# =========================================================

def get_history():

    conn = sqlite3.connect(DATABASE)

    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT *
        FROM predictions
        ORDER BY id DESC
    """).fetchall()

    conn.close()

    return rows


# =========================================================
# DASHBOARD
# =========================================================

@app.route("/")
def dashboard():

    history = get_history()

    total = len(history)

    high_risk = sum(
        1 for row in history
        if row["prediction"] == 1
    )

    low_risk = total - high_risk

    return render_template(
        "index.html",
        page="dashboard",
        history=history,
        total=total,
        high_risk=high_risk,
        low_risk=low_risk,
        prediction=None,
        probability=None
    )


# =========================================================
# PREDICTION
# =========================================================

@app.route("/prediction", methods=["GET", "POST"])
def prediction_page():

    prediction = None
    probability = None

    if request.method == "POST":

        try:

            age = float(request.form["age"])
            sex = float(request.form["sex"])
            cp = float(request.form["cp"])
            trestbps = float(request.form["trestbps"])
            chol = float(request.form["chol"])
            fbs = float(request.form["fbs"])
            restecg = float(request.form["restecg"])
            thalach = float(request.form["thalach"])
            exang = float(request.form["exang"])
            oldpeak = float(request.form["oldpeak"])
            slope = float(request.form["slope"])
            ca = float(request.form["ca"])
            thal = float(request.form["thal"])

            features = np.array([[
                age,
                sex,
                cp,
                trestbps,
                chol,
                fbs,
                restecg,
                thalach,
                exang,
                oldpeak,
                slope,
                ca,
                thal
            ]])

            scaled_features = scaler.transform(features)

            prediction = int(
                model.predict(scaled_features)[0]
            )

            if hasattr(model, "predict_proba"):

                probability = round(
                    float(
                        model.predict_proba(
                            scaled_features
                        )[0][1]
                    ) * 100,
                    1
                )

            else:
                probability = 50.0

            save_prediction(
                age,
                sex,
                prediction,
                probability
            )

        except Exception as e:

            print("Prediction Error:", e)

            return render_template(
                "index.html",
                page="prediction",
                error=str(e),
                prediction=None,
                probability=None
            )

    return render_template(
        "index.html",
        page="prediction",
        prediction=prediction,
        probability=probability,
        history=get_history()
    )


# =========================================================
# HISTORY
# =========================================================

@app.route("/history")
def history_page():

    history = get_history()

    return render_template(
        "index.html",
        page="history",
        history=history,
        prediction=None,
        probability=None
    )


# =========================================================
# STATISTICS
# =========================================================

@app.route("/statistics")
def statistics():

    history = get_history()

    total = len(history)

    high = sum(
        1 for row in history
        if row["prediction"] == 1
    )

    low = total - high

    high_percentage = (
        round((high / total) * 100, 1)
        if total else 0
    )

    low_percentage = (
        round((low / total) * 100, 1)
        if total else 0
    )

    return render_template(
        "index.html",
        page="statistics",
        total=total,
        high_risk=high,
        low_risk=low,
        high_percentage=high_percentage,
        low_percentage=low_percentage,
        history=history,
        prediction=None,
        probability=None
    )


# =========================================================
# REPORTS
# =========================================================

@app.route("/reports")
def reports():

    history = get_history()

    return render_template(
        "index.html",
        page="reports",
        history=history,
        prediction=None,
        probability=None
    )


# =========================================================
# PROFILE
# =========================================================

@app.route("/profile")
def profile():

    return render_template(
        "index.html",
        page="profile",
        history=get_history()
    )


# =========================================================
# SETTINGS
# =========================================================

@app.route("/settings")
def settings():

    return render_template(
        "index.html",
        page="settings",
        history=get_history()
    )


# =========================================================
# HELP
# =========================================================

@app.route("/help")
def help_page():

    return render_template(
        "index.html",
        page="help",
        history=get_history()
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )