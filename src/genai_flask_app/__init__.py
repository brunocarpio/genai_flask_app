import time

from flask import Flask, jsonify, redirect, render_template, request, url_for
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from genai_flask_app.model import WModel

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

loaded_agents = {}


def load_model(id, system):
    if id not in loaded_agents:
        model = WModel()
        model.system = system
        loaded_agents[id] = model
    else:
        model = loaded_agents[id]
    return model


class Agents(db.Model):
    __tablename__ = "agents"

    _id: Mapped[int] = mapped_column(primary_key=True)
    _name: Mapped[str] = mapped_column(nullable=False)
    _system: Mapped[str] = mapped_column(nullable=False)

    _prompts: Mapped[list["Prompts"]] = relationship(
        back_populates="_agent", cascade="all, delete"
    )

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def system(self):
        return self._system

    @system.setter
    def system(self, value):
        self._system = value

    def __str__(self) -> str:
        return f"Agent(id={self._id}, name={self._name}, system={self._system}, prompts={self._prompts})"


class Prompts(db.Model):
    __tablename__ = "prompts"

    _id: Mapped[int] = mapped_column(primary_key=True)
    _agent_id: Mapped[int] = mapped_column(ForeignKey("agents._id"))
    _value: Mapped[str] = mapped_column(nullable=False)

    _agent: Mapped["Agents"] = relationship(back_populates="_prompts")

    def __init__(self, prompt_value):
        self.value = prompt_value

    def __str__(self) -> str:
        return f"Prompt({self._id}, {self._agent_id}, {self._value})"


with app.app_context():
    db.create_all()


@app.route("/", methods=["GET"])
def index():
    agents = Agents.query.all()
    return render_template("index.html", agents=agents)


@app.route("/add", methods=["GET", "POST"])
def add_agent():
    if request.method == "POST":
        name = request.form["name"].strip()
        system = request.form["system"].strip()

        new_agent = Agents()
        new_agent.name = name
        new_agent.system = system
        print("new agent", new_agent)

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
        agent: Agents = Agents.query.get(id)
        if request.method == "POST":
            name = request.form["name"].strip()
            system = request.form["system"].strip()

            agent.name = name.strip()
            agent.system = system.strip()

            model = load_model(id, agent.system)
            model.system = system

            db.session.commit()

            return redirect(url_for("index"))
        return render_template("edit_agent.html", agent=agent)
    except Exception as e:
        return jsonify({"error": e}), 500


@app.route("/delete/<int:id>")
def delete_agent(id):
    try:
        agent = Agents.query.get(id)

        loaded_agents.pop(id, None)

        db.session.delete(agent)
        db.session.commit()
        return redirect(url_for("index"))
    except Exception as e:
        return jsonify({"error": e}), 500


@app.route("/chat_agent/<int:id>", methods=["GET"])
def chat_agent(id):
    agent: Agents = Agents.query.get(id)

    system = agent.system
    model = load_model(id, system)

    chat_history = model.get_chat_history()

    return render_template(
        "chat_agent.html", agent=agent, page_name="chat", chat_history=chat_history
    )


@app.route("/generate/<int:id>", methods=["POST"])
def generate(id):
    data = request.json

    selected_model = data.get("model")
    user_message = data.get("message")

    if not user_message or not selected_model:
        return jsonify({"error": "Missing message or model selection"}), 400

    start_time = time.time()

    agent: Agents = Agents.query.get(id)
    system = agent.system

    model = load_model(id, system)

    if selected_model == "llama":
        result = model.llama_response(user_message)
    elif selected_model == "granite":
        result = model.granite_response(user_message)
    elif selected_model == "mistral":
        result = model.mistral_response(user_message)
    else:
        return jsonify({"error": "Invalid model selection"}), 400

    duration = time.time() - start_time
    print("request duration seconds", duration)

    response = {"content": result, "duration": duration}
    return response


def main():
    app.run(debug=True)


if __name__ == "__main__":
    main()
