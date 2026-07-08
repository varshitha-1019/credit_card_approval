from flask import Flask, render_template, request
import pandas as pd
import mysql.connector
import joblib
model = joblib.load("trained_models/fraud_model.pkl")

app = Flask(__name__)

try:
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Bhavani@123",
        database="fraudshield"
    )
    print("✅ MySQL Connected")
except Exception as e:
    print("⚠ MySQL Not Connected:", e)
    db = None
    
@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/analytics")
def analytics():
    return render_template("analytics.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/approval", methods=["GET", "POST"])
def approval():

    if request.method == "POST":

        try:

            name = request.form["name"]
            age = int(request.form["age"])
            pan = request.form["pan"]
            income = float(request.form["income"])
            score = int(request.form["score"])
            employment = request.form["employment"]
            loan = request.form["loan"]
            limit = float(request.form["limit"])

            statement = request.files["statement"]
            statement_name = statement.filename

            # Approval Logic
            if score >= 750 and income >= 500000:
                status = "Approved"
            else:
                status = "Rejected"

            # Save to database only if MySQL is connected
            if db is not None:

                cursor = db.cursor()

                query = """
                INSERT INTO applications
                (
                    name,
                    age,
                    pan,
                    income,
                    credit_score,
                    employment_status,
                    existing_loan,
                    credit_limit,
                    status
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """

                values = (
                    name,
                    age,
                    pan,
                    income,
                    score,
                    employment,
                    loan,
                    limit,
                    status
                )

                cursor.execute(query, values)
                db.commit()
                cursor.close()

            else:
                print("MySQL not connected. Skipping database insert.")

            return render_template(
                "result.html",
                status=status,
                name=name,
                pan=pan,
                score=score,
                income=income
            )

        except Exception as e:
            return f"Error: {e}"

    return render_template("approval.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():

    result = None

    if request.method == "POST":

        try:

            file = request.files["file"]

            df = pd.read_csv(file)

            total = len(df)

            if "Class" in df.columns:

                fraud = len(df[df["Class"] == 1])
                normal = len(df[df["Class"] == 0])

                result = f"Total Transactions: {total} | Fraud Transactions: {fraud} | Normal Transactions: {normal}"

            else:

                result = f"Dataset Loaded Successfully ({total} rows)"

        except Exception as e:

            result = str(e)

    return render_template("upload.html", result=result)


@app.route("/predict", methods=["GET", "POST"])
def predict():

    prediction = None
    confidence = None

    if request.method == "POST":

        try:

            # Get input from form
            time = float(request.form["time"])
            amount = float(request.form["amount"])

            # Create DataFrame with correct feature names
            data = pd.DataFrame(
                [[time, amount]],
                columns=["Time", "Amount"]
            )

            # Predict
            prediction_value = model.predict(data)[0]

            # Prediction probability
            probability = model.predict_proba(data)[0]
            confidence = round(max(probability) * 100, 2)

            # Display Result
            if prediction_value == 1:
                prediction = "🚨 Fraud Transaction Detected"
            else:
                prediction = "✅ Normal Transaction"

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template(
        "predict.html",
        prediction=prediction,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True)