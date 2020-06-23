import os
import sys
from flask import Flask, render_template, flash, request, redirect, url_for, send_from_directory
from wtforms import Form, StringField, PasswordField, BooleanField, SubmitField,validators
from wtforms.validators import DataRequired
from retile import Retile
from celery import Celery

app = Flask(__name__)
app.config['SECRET_KEY'] = "it is a secret"
training_file = ""

'''
app.config.update(
    CELERY_BROKER_URL='redis://localhost:6379',
    CELERY_RESULT_BACKEND='redis://localhost:6379'
)
celery = make_celery(app)

def make_celery(app):
    celery = Celery(
        app.import_name,
        backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL']
    )
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery
'''

def file_exists():
    path = request.form.get('file_path')
    if os.path.isfile(path):
        print("File exists!",file=sys.stderr)
        return True
    print("File not exists!",file=sys.stderr)
    return False


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html',title='Home')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    global training_file
    if request.method == 'POST' and file_exists():  #this block is only entered when the form is submitted
        training_file = request.form.get('file_path')
        print(training_file,file=sys.stderr)
        return render_template('classify.html',file_path = training_file)

    return render_template('upload.html', title='Sign In')

@app.route('/classify', methods=['POST'])
def classify():
    global training_file
    #Retile.test(Retile)
    Retile.run(Retile, "256", training_file, "output")
    
    print(training_file,file=sys.stderr)
    return render_template('classify.html',file_path = training_file)


if __name__ == '__main__':
    app.run(debug=True)


