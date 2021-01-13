import os
from io import BytesIO
import zipfile
import json
from time import time
from flask import Blueprint, request, Response

from . import constants
from .. import socketio

bp = Blueprint('fileutils', __name__, url_prefix='/files')

def create_folders():
    if not os.path.exists(constants.UPLOADED_FOLDER):
        os.makedirs(constants.UPLOADED_FOLDER)
    if not os.path.exists(constants.PROCESSED_FOLDER):
        os.makedirs(constants.PROCESSED_FOLDER)

@bp.route('upload', methods=['POST'])
def upload_files():
    create_folders()
    files = request.data
    content_type = request.mimetype
    if content_type == 'application/zip':
        with zipfile.ZipFile(BytesIO(files), 'r') as zip_ref:
            zip_ref.extractall(constants.UPLOADED_FOLDER)

        return 'Done Uploading!'
    elif content_type == 'image/tiff':
        filename = 'image-' + str(round(time() * 1000))  + '.tif'
        query_filename = request.args.get('filename')
        if query_filename is not None:
            filename = query_filename
        uploaded_path = os.path.join(constants.UPLOADED_FOLDER, filename)
        with open(uploaded_path, 'wb') as tif:
            tif.write(files)

        return 'Done Uploading!'
    else:
        return Response(json.dumps({
            'message': 'Content-Type invalid. Only .zip and .tif files are supported.',
            'status': 400,
        }), status=400, mimetype='application/json')

@bp.route('retile', methods=['POST'])
def retile_files():
    create_folders()
    total_files = sum(1 for files in os.walk(constants.UPLOADED_FOLDER) for _ in files[2])
    files_counter = 0
    content_type = request.mimetype
    room = None
    if content_type == 'application/json':
        data = request.get_json()
        if 'room' not in data:
            return Response(json.dumps({
                'message': 'Body provided by request but room not found.',
                'status': 400,
            }), status=400, mimetype='application/json')
        room = data['room']
    for dirpath, _, filenames in os.walk(constants.UPLOADED_FOLDER):
        for filename in filenames:
            os.system('gdal2tiles.py ' + os.path.join(dirpath, filename) + ' ' + os.path.join(constants.PROCESSED_FOLDER, filename))
            files_counter += 1
            socketio.emit('message', str(total_files) + '/' + str(files_counter), room=room)
    return 'Done Retiling!'

@bp.route('list', methods=['GET'])
def list_files():
    create_folders()
    files = []
    for dirpath, _, filenames in os.walk(constants.UPLOADED_FOLDER):
        files.extend(os.path.join(dirpath, filename) for filename in filenames)
    return Response(json.dumps(files), mimetype='application/json')