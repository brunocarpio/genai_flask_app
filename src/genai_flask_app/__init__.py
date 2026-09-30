import time  # noqa: I001

from flask import Flask, jsonify, redirect, render_template, request, url_for
from genai_flask_app.model import AppState
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import Mapped, mapped_column

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

loaded_agents = {}


class Agent(db.Model):
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(nullable=False)
    system: Mapped[str] = mapped_column(nullable=False)


with app.app_context():
    db.create_all()


@app.route("/", methods=["GET"])
def index():
    agents = Agent.query.all()
    return render_template("index.html", agents=agents)


@app.route("/add", methods=["GET", "POST"])
def add_agent():
    if request.method == "POST":
        name = request.form["name"].strip()
        system = request.form["system"].strip()
        print(name, system)
        new_agent = Agent(name=name, system=system)
        try:
            db.session.add(new_agent)
            db.session.commit()
            return redirect(url_for("index"))
        except Exception as e:
            return jsonify({"error": e}), 500
    return render_template("add_agent.html")


@app.route("/edit/<int:id>", methods=["GET", "POST"])
def edit_agent(id):
    try:
        agent = Agent.query.get(id)
        if request.method == "POST":
            agent.name = request.form["name"].strip()
            agent.system = request.form["system"].strip()
            db.session.commit()
            return redirect(url_for("index"))
        return render_template("edit_agent.html", agent=agent)
    except Exception as e:
        return jsonify({"error": e}), 500


@app.route("/delete/<int:id>")
def delete_agent(id):
    try:
        agent = Agent.query.get(id)
        db.session.delete(agent)
        db.session.commit()
        return redirect(url_for("index"))
    except Exception as e:
        return jsonify({"error": e}), 500


@app.route("/chat_agent/<int:id>", methods=["GET"])
def chat_agent(id):
    agent = Agent.query.get(id)
    return render_template("chat_agent.html", agent=agent, page_name="chat")


@app.route("/generate/<int:id>", methods=["POST"])
def generate(id):
    data = request.json

    selected_model = data.get("model")
    user_message = data.get("message")

    if not user_message or not selected_model:
        return jsonify({"error": "Missing message or model selection"}), 400

    start_time = time.time()

    print("request start_time", start_time)

    try:
        agent = Agent.query.get(id)

        loaded_agents.setdefault(id, AppState(agent.system))

        model_app = loaded_agents.get(id)

        print(model_app)

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
