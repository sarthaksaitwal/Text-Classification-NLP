from flask import Flask, render_template, request
from src.inference.predict import NewsClassifier

app = Flask(__name__)

# Load model once at startup
classifier = NewsClassifier()


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    text = ""

    if request.method == "POST":
        text = request.form["news_text"]

        if text.strip():
            result = classifier.predict([text])[0]
            prediction = result["label"]
            confidence = result["confidence"]

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        text=text
    )


if __name__ == "__main__":
    app.run(debug=True)
