import os

def retile(uploaded_path, processed_path):
    os.system('gdal2tiles.py ' + uploaded_path + ' ' + processed_path)