from flask import Flask

from app.routes import init_app


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    init_app(app)
    return app
