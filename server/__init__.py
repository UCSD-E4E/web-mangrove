from flask import Flask
from flask_socketio import SocketIO
from secrets import token_urlsafe

socketio = SocketIO()

def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config['SECRET_KEY'] = token_urlsafe(16)

    from . import fileutils
    app.register_blueprint(fileutils.bp)

    socketio.init_app(app)
    return app