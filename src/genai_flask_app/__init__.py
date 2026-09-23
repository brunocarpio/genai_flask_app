import time  # noqa: I001

from flask import Flask, jsonify, render_template, request
from genai_flask_app.model import AppState

app = Flask(__name__)
model_app = AppState("You are a funny comedian who always provides fun answers.")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    selected_model = data.get("model")
    user_message = data.get("message")

    if not user_message or not selected_model:
        return jsonify({"error": "Missing message or model selection"}), 400

    start_time = time.time()

    print("request start_time", start_time)

    try:
        if selected_model == "llama":
            result = model_app.llama_response(user_message)
        elif selected_model == "granite":
            result = model_app.granite_response(user_message)
        elif selected_model == "mistral":
            result = model_app.mistral_response(user_message)
        else:
            return jsonify({"error": "Invalid model selection"}), 400

        return jsonify({"response": result, "duration": time.time() - start_time})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def main():
    app.run(debug=True)
