from flask import Flask
from flask_socketio import SocketIO
from secrets import token_urlsafe

socketio = SocketIO()

def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config['SECRET_KEY'] = token_urlsafe(16)

    @app.after_request
    def add_header(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Origin, X-Requested-With, Content-Type, Accept'
        return response

    from . import fileutils
    app.register_blueprint(fileutils.bp)

    socketio.init_app(app, cors_allowed_origins='*')
    return app
