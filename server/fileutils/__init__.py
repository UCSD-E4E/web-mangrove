import os
from io import BytesIO
import zipfile
import json
from time import time
from flask import Blueprint, request, Response

from . import constants

bp = Blueprint('fileutils', __name__, url_prefix='/files')

@bp.route('', methods=('GET', 'POST'))
def get_files():
    if not os.path.exists(constants.UPLOADED_FOLDER):
        os.makedirs(constants.UPLOADED_FOLDER)
    if not os.path.exists(constants.PROCESSED_FOLDER):
        os.makedirs(constants.PROCESSED_FOLDER)
    if request.method == 'GET':
        files = []
        for dirpath, _, filenames in os.walk(constants.UPLOADED_FOLDER):
            files.extend(os.path.join(dirpath, file) for file in filenames)
        return Response(json.dumps(files), mimetype='application/json')
    elif request.method == 'POST':
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
            with open(os.path.join(constants.UPLOADED_FOLDER, filename), 'wb') as tif:
                tif.write(files)
            return 'Done Uploading!'
        else:
            return Response(json.dumps({
                'message': 'Content-Type invalid. Only .zip and .tif files are supported.',
                'status': 400,
            }), status=400, mimetype='application/json')

    return 'Hello World!'