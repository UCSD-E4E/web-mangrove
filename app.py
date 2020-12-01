import azure_blob
import string, random, requests
import flash
from zipfile import ZipFile

from celery import Celery, Task

import os
from os import path
import json
import sys
from flask import Flask, render_template, make_response, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

from tqdm.autonotebook import tqdm
import tensorflow as tf
import pandas as pd
import numpy as np
import rasterio
import subprocess

import visualize
import classify_mod
import base64

from rasterio.plot import show
import matplotlib as mpl
import requests

import rasterio.features
from geojson import Point, Feature, FeatureCollection, dump


from PIL import Image
import PIL
PIL.Image.MAX_IMAGE_PIXELS = None

# dash 
import dash
import dash_core_components as dcc
import dash_html_components as html

from dotenv import load_dotenv

load_dotenv(verbose=True)


# color the pics in visualization
green_hue = (180-78)/360.0
red_hue = (180-180)/360.0


# TO DO: ADD OS FUNCTIONS TO MAKE IT RUN ON ANYONE'S COMPUTER
print('MAIN_DIRECTORY from OS: ', os.path.dirname(os.path.realpath(__file__)))

MAPBOX_APIKEY = "pk.eyJ1Ijoibm1laXN0ZXIiLCJhIjoiY2tjODZya3VnMHU0cjJ2bGpxanh0eW9idiJ9._DNCB5IcbFoGl7AIm0vVlA"

# overall directory where all files and folders are stored 
MAIN_DIRECTORY = os.path.dirname(os.path.realpath(__file__)) + "/"

# model path 
MODEL_PATH = MAIN_DIRECTORY + "mvnmv4_merced_bright.zip"
 
# image directory. images/images contains the tif files
IMAGE_DIRECTORY = MAIN_DIRECTORY + "images"

# upload path 
UPLOAD_FOLDER = IMAGE_DIRECTORY + '/images/'

ALLOWED_EXTENSIONS = {'zip', 'tif'}

sys.path.append(MAIN_DIRECTORY)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("TF version:", tf.__version__)
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")
server = Flask(__name__)

server.config['SECRET_KEY'] = "it is a secret" # old code idk if I need this
server.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


m_filename = 'mangrove'
nm_filename = 'nonmangrove'


CONNECTION_STRING = os.getenv('CONNECTION_STRING')

def make_celery(app):
    celery = Celery(server.name, broker=server.config['CELERY_BROKER_URL'])
    celery.conf.update(server.config)

    TaskBase = Task
    class ContextTask(TaskBase):
        abstract = True
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return TaskBase.__call__(self, *args, **kwargs)
    celery.Task = ContextTask
    return celery

# server.config['CELERY_BROKER_URL'] = 'redis://127.0.0.1:6379/0'
server.config['CELERY_BROKER_URL'] = 'redis://h:pb2bfc095a54282dafcd0a69dbd48562726bf133eb775122633cd320211f73c12@ec2-3-224-237-146.compute-1.amazonaws.com:18009'
celery = make_celery(server)

# check for allowed file extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def id_generator(size=32, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))



@server.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@server.route('/index')
def index():
    return render_template('index.html')

@server.route('/about')
def about():
    return render_template('about.html')

# post so user can send login credentials to the login endpoint
@server.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    # Handle post request
    if request.method == 'POST':
        # test input values to see if they're correct 
        if (request.form['username'] != 'sioadmin') or (request.form['password'] != 'sioadmin'):
            error = 'Invalid credentials. Please try again.'
        # if info is correct, redirect to main page
        else:
            # maybe later change this so that it goes to "/" and /login isn't in the url, lowkey dont really know how to do that
            html = render_template('index.html')
            response = make_response(html)
            return response

    return render_template('login.html', error=error)

@server.route('/download', methods=['GET'])
def download():

    filelist= list([m_filename+'.prj', nm_filename+'.prj', m_filename+'.tif', nm_filename+'.tif', m_filename+'.shx', nm_filename+'.shx', m_filename+'.shp', nm_filename+'.shp', nm_filename+'.dbf', m_filename+'.dbf'])
    filename = 'all.zip'
    createZip(filename, filelist)
    # Delete all the files uploaded + created 
    os.system('rm -rf ' + MAIN_DIRECTORY + 'images/*')
    print('mkdir ' + IMAGE_DIRECTORY + '/images/')
    # recreate images/images subfolder
    os.system('mkdir ' + IMAGE_DIRECTORY + '/images/')
    print('mkdir ' + IMAGE_DIRECTORY + '/images/')
    
    '''# list of files to delete so each user can start with same file structure
    files_to_del = filelist
    for f in files_to_del:
        os.system('rm -rf ' + MAIN_DIRECTORY + f)
        print('rm -rf ' + MAIN_DIRECTORY + f)
    
    # delete the files in static images
    for f in os.listdir('static/images/'):
        os.system('rm -rf ' + MAIN_DIRECTORY + 'static/images/' + f)
        print('rm -rf ' + MAIN_DIRECTORY + 'static/images/' + f)

    os.system('rm -rf ' + MAIN_DIRECTORY + 'image_m_green.png')
    print('rm -rf ' + MAIN_DIRECTORY + 'image_m_green.png')
    os.system('rm -rf ' + MAIN_DIRECTORY + 'image_nm_red.png')
    print('rm -rf ' + MAIN_DIRECTORY + 'image_nm_red.png')
    
    os.system('rm -rf ' + MAIN_DIRECTORY + 'mngrv.geojson')
    print('rm -rf ' + MAIN_DIRECTORY + 'mngrv.geojson')
    os.system('rm -rf ' + MAIN_DIRECTORY + 'n-mngrv.geojson')
    print('rm -rf ' + MAIN_DIRECTORY + 'n-mngrv.geojson')
    
    '''
    return redirect(url_for('uploaded_file', filename=filename))

@server.route('/classificationfin',  methods=['GET'])
def classificationfin():
    print('in classification fin')
    
    container_name = 'prediction-results'
    client = azure_blob.DirectoryClient(CONNECTION_STRING, container_name)
    list_of_files = list(client.ls_files('', recursive=False))
    print(list_of_files)
    if list_of_files != []:
        if list_of_files[0] == 'content.csv':
            html='Classification finished.'
    else:
        html = ''

    response = make_response(html)
    return response
    

@server.route('/searchresults', methods=['GET'])
def searchResults():
    
    print('search results is called')
    output_container_name = 'output-files'
    client = azure_blob.DirectoryClient(CONNECTION_STRING, output_container_name)
    list_of_files = list(client.ls_files('', recursive=False))

    html = ''
    for filename in list_of_files:
        html += filename+ '<br>'

    response = make_response(html)
    return response


@server.route('/upload', methods=['POST'])
def upload():

    print('in upload')
    # download() delete all the images from the prev instance of running the website
    # get the function running
    msg = requests.get('https://azunzipmangroves.azurewebsites.net/')
    if msg.status_code == 200:
        # check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            html = render_template('index.html')
            response = make_response(html)
            return response
            
        elif file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print('filename:', filename)
            input_container = 'input-files'
            client = azure_blob.DirectoryClient(CONNECTION_STRING, input_container)
            print('created client')
            client.create_blob_from_stream(blob_name=filename, stream=file)
            print('completed file upload')
            # file.save(os.path.join(server.config['UPLOAD_FOLDER'], filename))
    else: 
        print('error occured HANDLE THIS!')

    html = render_template('index.html')
    response = make_response(html)
    return response

@server.route('/visualization', methods=['GET'])
def visualization():
    return redirect(url_for('/visualization/_dash-update-component'))

# create a zip file with name zip_name (1.zip) 
# file list contains all the file going into the zip
# delete is an option boolean that determines ifyou should delete the files or not
def createZip(zip_name, filelist, delete=False):
    
    existing_filelist = []
    for filename in filelist:
        if os.path.exists(filename):
            existing_filelist.append(filename)

    with ZipFile(zip_name, 'w') as zipObj:
        # Add multiple files to the zip
        for filename in existing_filelist:
            zipObj.write(filename)
    if delete: 
        for filename in filelist:
            os.remove(MAIN_DIRECTORY+filename)
    return


@server.route('/download_m', methods=['GET'])
def download_m():

    filelist=[m_filename+'.dbf', m_filename+'.prj', m_filename+'.shp', m_filename+'.shx', m_filename+'.tif']
    filename = m_filename+'.zip'
    createZip(filename, filelist)
    return redirect(url_for('uploaded_file', filename=filename))

@server.route('/download_nm', methods=['GET'])
def download_nm():

    filelist=[nm_filename+'.dbf', nm_filename+'.prj', nm_filename+'.shp', nm_filename+'.shx', nm_filename+'.tif']
    filename = nm_filename+'.zip'
    createZip(filename, filelist)
    return redirect(url_for('uploaded_file', filename=filename))

@server.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(MAIN_DIRECTORY,
                               filename)

@celery.task()
def classify_celery():
    print('this CELERY IS RUNNING')
    classify_mod.classify()
    return 

@server.route('/prep_classification', methods=['GET'])
def prep_classification():
    classify_mod.post_classify()
    # not sure if this does anything?
    html = render_template('index.html')
    response = make_response(html)
    return response

@server.route('/classify', methods=['GET'])
def classify():
    print('in classify')

    # classify_celery.apply_async() # run on heroku

    classify_mod.classify() # run locally 
    html = "Performing classification... "
    response = make_response(html)
    return response
    
'''@server.before_request
def require_login():

    allowed_routes = ['login']
    if request.endpoint not in allowed_routes:
        return redirect('/login')
'''

def get_fig(version, mngrv_geojson, n_mngrv_geojson, sample=False):

    mangrove_exists = True
    nonmangrove_exists = True

    if mngrv_geojson == {}:
        mangrove_exists = False
        mngrv_geojson = {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[0, 0]]}}]}
    if n_mngrv_geojson == {}:
        nonmangrove_exists = False
        n_mngrv_geojson = {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[0, 0]]}}]}


    if sample:
        image_filename = "image_m_green-perm.png"
        image_m_green = base64.b64encode(open(image_filename, 'rb').read())

        image_filename = "image_nm_red-perm.png"
        image_nm_red = base64.b64encode(open(image_filename, 'rb').read())
    else: 
        if mangrove_exists:
            image_filename = "image_m_green.png"
            image_m_green = base64.b64encode(open(image_filename, 'rb').read())

        if nonmangrove_exists:
            image_filename = "image_nm_red.png"
            image_nm_red = base64.b64encode(open(image_filename, 'rb').read())


    if mangrove_exists:
        mngrv_tiles = len(mngrv_geojson['features'])
    if nonmangrove_exists:
        n_mngrv_tiles = len(n_mngrv_geojson['features'])

    # if only one exists 
    if mangrove_exists and nonmangrove_exists:
        haveborder = True
    else: 
        haveborder = False

    if mangrove_exists:
        sources = visualize.make_sources(mngrv_geojson) # dictionary to create the borders
    elif nonmangrove_exists:
        sources = visualize.make_sources(n_mngrv_geojson) # dictionary to create the borders
    print('got sources')

    hasScatter = True
    if mangrove_exists:
        lons, lats = visualize.get_centers(mngrv_geojson) # lat lon of all the centers of the scatter plot (to mimic hover text effect)
        hasScatter = True
    elif nonmangrove_exists:
        lons, lats = visualize.get_centers(n_mngrv_geojson) # lat lon of all the centers of the scatter plot (to mimic hover text effect)
        hasScatter = True
    else:
        hasScatter = False
        lons, lats = [0], [0]

    
    print('got lats lons')
    if mangrove_exists:
        latmin_m, latmax_m, lonmin_m, lonmax_m = visualize.get_latlonminmax(mngrv_geojson)
    if nonmangrove_exists:
        print('got min max lats lons')
        latmin_nm, latmax_nm, lonmin_nm, lonmax_nm = visualize.get_latlonminmax(n_mngrv_geojson)
    if not mangrove_exists and not nonmangrove_exists:
        latmin_m, latmax_m, lonmin_m, lonmax_m = (0, 0, 0, 0)


    avg_lon = np.average(lons)
    avg_lat = np.average(lats)
    # use a scatter map box to create the hover text
    if hasScatter:
        data = dict(type='scattermapbox',
                    lat=lats,
                    lon=lons,
                    opacity = 0,
                    mode='markers',
                    text='mangrove',
                    showlegend=False,
                    hoverinfo='text'
                    )
    else:
        data = dict(type='scattermapbox',
                    lat=[0],
                    lon=[0],
                    opacity = 0,
                    mode='markers',
                    text='mangrove',
                    showlegend=False,
                    hoverinfo='text'
                    )

    if haveborder:
        if mangrove_exists:
            border = [dict(sourcetype = 'geojson',
                    source =sources[k],
                    below="",
                    type = 'line',    # the borders
                    line = dict(width = 2),
                    color = 'black',
                    ) for k in range(mngrv_tiles)
                ]
        elif nonmangrove_exists:
            border = [dict(sourcetype = 'geojson',
                    source =sources[k],
                    below="",
                    type = 'line',    # the borders
                    line = dict(width = 2),
                    color = 'black',
                    ) for k in range(n_mngrv_tiles)
                ]
        else:
            border = [{}]
    else: 
        border = {}

    if mangrove_exists:
        mangrove = [dict(below ='',
                            opacity=1,
                        source = 'data:image/png;base64,{}'.format(image_m_green.decode()),
                        sourcetype= "image",
                        coordinates =  [
                                [lonmin_m, latmax_m], [lonmax_m, latmax_m], [lonmax_m, latmin_m], [lonmin_m, latmin_m]
                                        ])]
    else: 
        mangrove = [dict(below ='',
                            opacity=1,
                        # source = 'data:image/png;base64,{}'.format(image_m_green.decode()),
                        # sourcetype= "image",
                        coordinates =  [
                                [0, 0], [0, 0], [0, 0], [0, 0]
                                        ])]

    if nonmangrove_exists:
        non_mangrove = [dict(below ='',
                            opacity=1,
                        source = 'data:image/png;base64,{}'.format(image_nm_red.decode()),
                        sourcetype= "image",
                        coordinates =  [
                                [lonmin_nm, latmax_nm], [lonmax_nm, latmax_nm], [lonmax_nm, latmin_nm], [lonmin_nm, latmin_nm]
                                        ])]
    print('version:', version)

    if len(version) == 0 and haveborder:
        layers=(border)
    elif len(version) == 0 and not haveborder:
        layers = ({})
    elif len(version) == 1 and haveborder:
        if (version[0] == 'mangrove') and mangrove_exists:
            layers=(border+mangrove)
        elif (version[0] == 'mangrove') and not mangrove_exists:
            layers=(border)
        # gotta add probability
        # elif (version[0] == 'non-mangrove'):
        elif (version[0] == 'non-mangrove') and nonmangrove_exists:
            layers=(border+non_mangrove)
        elif (version[0] == 'non-mangrove') and not nonmangrove_exists:
            layers=(border)
        else:
            layers=(border)
    elif len(version) == 1 and not haveborder:
        if (version[0] == 'mangrove') and mangrove_exists:
            layers=(mangrove)
        elif (version[0] == 'mangrove') and not mangrove_exists:
            layers=({})
        # gotta add probability
        # elif (version[0] == 'non-mangrove'):
        elif (version[0] == 'non-mangrove') and nonmangrove_exists:
            layers=(non_mangrove)
        elif (version[0] == 'non-mangrove') and not nonmangrove_exists:
            layers=({})
        else:
            layers = ({})
    # everything
    elif len(version) == 2 and haveborder:
        if mangrove_exists and nonmangrove_exists:
            layers=(border+mangrove+non_mangrove)
        if mangrove_exists and not nonmangrove_exists:
            layers=(border+mangrove)
        if not mangrove_exists and nonmangrove_exists:
            layers=(border+non_mangrove)
        if not mangrove_exists and not nonmangrove_exists:
            layers=(border)
    elif len(version) == 2 and not haveborder:
        if mangrove_exists and nonmangrove_exists:
            layers=(mangrove+non_mangrove)
        if mangrove_exists and not nonmangrove_exists:
            layers=(mangrove)
        if not mangrove_exists and nonmangrove_exists:
            layers=(non_mangrove)
        if not mangrove_exists and not nonmangrove_exists:
            layers=({})

    if layers == ({}):
        layers = ([dict(below ='',
                            opacity=1,
                        # source = 'data:image/png;base64,{}'.format(image_m_green.decode()),
                        # sourcetype= "image",
                        coordinates =  [
                                [0, 0], [0, 0], [0, 0], [0, 0]
                                        ])])
    # version =='prob'

    '''if sample:
        title = "Sample Visualization of Mangrove CNN"
    else:
        title = "Visualization of Mangrove CNN"
    layout = dict(title=title,
                autosize=False,
                width=1400,
                height=800,
                hovermode='closest',
                hoverdistance = 30,
                mapbox=dict(accesstoken=MAPBOX_APIKEY,
                            layers=layers,
                            bearing=0,
                            center=dict(
                                        lat=avg_lat,  # the center of this regions
                                        lon=avg_lon),
                            pitch=0,
                            zoom=16,
                            style = 'mapbox://styles/mapbox/satellite-v8'

                            )
                )'''

    if mangrove_exists or nonmangrove_exists:
        zoom = 16
        layout = dict(
                    autosize=False,
                    width=1400,
                    height=800,
                    margin={'t': 0, 'l':0, 'r':0, 'b':0},
                    hovermode='closest',
                    hoverdistance = 30,
                    mapbox=dict(accesstoken=MAPBOX_APIKEY,
                                layers=layers,
                                bearing=0,
                                center=dict(
                                            lat=avg_lat,  # the center of this regions
                                            lon=avg_lon),
                                pitch=0,
                                zoom=zoom, 
                                # style = 'mapbox://styles/mapbox/satellite-v8'
                                style = 'mapbox://styles/mapbox/streets-v11'
                                )
                    )
    else:
        zoom = 2
        layout = dict(
                    autosize=False,
                    width=1400,
                    height=800,
                    margin={'t': 0, 'l':0, 'r':0, 'b':0},
                    hovermode='closest',
                    hoverdistance = 30,
                    mapbox=dict(accesstoken=MAPBOX_APIKEY,
                                layers=[],
                                bearing=0,
                                center=dict(
                                            lat=0,  # the center of this regions
                                            lon=0),
                                pitch=0,
                                zoom=zoom, 
                                style = 'mapbox://styles/mapbox/streets-v11'
                                )
                    )
        

    dict_of_fig = dict(data=[data], layout=layout)
    return dict_of_fig

def start_dash():
    # open the permanent tif image and create geojson file
    '''final_filename = 'mngrv-perm.geojson'
    with open(final_filename) as f:
        mngrv_geojson = json.load(f)

    # open the tif image and create geojson file
    final_filename = 'n-mngrv-perm.geojson'
    with open(final_filename) as f:
        n_mngrv_geojson = json.load(f)'''

    mngrv_geojson = {}
    n_mngrv_geojson = {}

    version = ['mangrove', 'non-mangrove']
    dict_of_fig = get_fig(version, mngrv_geojson, n_mngrv_geojson)

    app.layout = html.Div([html.Button('View Sample', id='view-sample', n_clicks=0), 
        html.Button('Update', id='view-mine', n_clicks=0), 
        dcc.Checklist(inputStyle={'-webkit-appearance': 'checkbox'}, id='radiobtn', 
        options=[
            {'label': 'Mangrove', 'value': 'mangrove'},
            {'label': 'Non-Mangrove', 'value': 'non-mangrove'},
            # {'label': 'Everything', 'value': 'everything'},
            # {'label': 'Probability', 'value': 'prob'}
        ],
        value=['mangrove', 'non-mangrove'], 
        labelStyle={'display': 'inline-block', 'textAlign': 'center', 'cursor': 'pointer'})  , 
        dcc.Graph(id='viz', figure=dict_of_fig)
        ])
    return

def start_dash_prev():
    # open the permanent tif image and create geojson file
    final_filename = 'mngrv-perm.geojson'
    with open(final_filename) as f:
        mngrv_geojson = json.load(f)

    # open the tif image and create geojson file
    final_filename = 'n-mngrv-perm.geojson'
    with open(final_filename) as f:
        n_mngrv_geojson = json.load(f)
    
    version = ['mangrove', 'non-mangrove']
    if mngrv_geojson != None and n_mngrv_geojson != None:
        dict_of_fig = get_fig(version, mngrv_geojson, n_mngrv_geojson)

        app.layout = html.Div([html.Button('View Sample', id='view-sample', n_clicks=0), 
            html.Button('Update', id='view-mine', n_clicks=0), 
            dcc.Checklist(inputStyle={'-webkit-appearance': 'checkbox'}, id='radiobtn', 
            options=[
                {'label': 'Mangrove', 'value': 'mangrove'},
                {'label': 'Non-Mangrove', 'value': 'non-mangrove'},
                # {'label': 'Everything', 'value': 'everything'},
                # {'label': 'Probability', 'value': 'prob'}
            ],
            value=['mangrove', 'non-mangrove'], 
            labelStyle={'display': 'inline-block', 'textAlign': 'center', 'cursor': 'pointer'})  , 
            dcc.Graph(id='viz', figure=dict_of_fig), 
            # Hidden div inside the app that stores the intermediate value
            html.Div(id='intermediate-value', style={'display': 'none'}),
            ])
    return

app = dash.Dash(__name__, server=server, routes_pathname_prefix='/visualization/', show_undo_redo=True)

start_dash()


@app.callback(dash.dependencies.Output('viz', 'figure'),
[dash.dependencies.Input(component_id='view-mine', component_property='n_clicks'), 
dash.dependencies.Input(component_id='view-sample', component_property='n_clicks'), 
dash.dependencies.Input('radiobtn', 'value')])
def update_figure(n_clicks_mine, n_clicks_sample, version):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    print('callback context: ', dash.callback_context.triggered)
    print('version in app callback: ', version) #  items checked in checkboxes. len: 0: nothing checked, 1: 1 item checked 2: 2 items checked. the values of the list are the values of items checked
    print('call back: ', dash.callback_context)
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if changed_id == 'view-sample.n_clicks' or changed_id == 'view-mine.n_clicks':
        btn_changed_id = changed_id
    else: 
        btn_changed_id = ''
    print(changed_id)
    print('n_clicks mine', n_clicks_mine)
    print('n_clicks sample', n_clicks_sample)

    # generate sample
    sample = False

    _mngrv_geojson = None
    _n_mngrv_geojson = None

    # if View Sample is the most recently clicked button 
    if (btn_changed_id == 'view-sample.n_clicks'):
        final_filename = 'mngrv-perm.geojson'
        with open(final_filename) as f:
            _mngrv_geojson = json.load(f)

        # open the tif image and create geojson file
        final_filename = 'n-mngrv-perm.geojson'
        with open(final_filename) as f:
            _n_mngrv_geojson = json.load(f)

        sample = True

    # if 'update' is the most recent button clicked, render the classfication
    elif (btn_changed_id == 'view-mine.n_clicks'):
        print('directory: ', os.listdir())

        nonmangrove_exists = False
        mangrove_exists = False

        m_tif_filename = m_filename+'.tif'
        nm_tif_filename = nm_filename+'.tif'

        
        if (path.exists(m_tif_filename)):
            mangrove_exists = True
        
        if (path.exists(nm_tif_filename)):
            nonmangrove_exists = True


        if mangrove_exists:
            final_filename = 'mngrv.geojson'
            if path.exists(final_filename):
                print("'mngrv.geojson' exists")
                with open(final_filename) as f:
                    _mngrv_geojson = json.load(f)
        
            else:
                print("'mngrv.geojson' does not exist yet")
                _mngrv_geojson = visualize.create_geojson(m_tif_filename, final_filename)
        else: 
            _mngrv_geojson={}

        if nonmangrove_exists:
            final_filename = 'n-mngrv.geojson'
            if path.exists(final_filename):
                print("'n-mngrv.geojson' exists")
                with open(final_filename) as f:
                    _n_mngrv_geojson = json.load(f)
        
            else:
                print("'n-mngrv.geojson' does not exist yet")
                _n_mngrv_geojson = visualize.create_geojson(nm_tif_filename, final_filename)
        else: 
            _n_mngrv_geojson={}

        # display the images
        # DO SOME ELSE STATEMENT!
        if mangrove_exists:
            saved_img = "image_m_green.png"
            if not path.exists(saved_img):
                image_m_green = visualize.get_im(m_tif_filename, green_hue)
                image_m_green.save("image_m_green.png","PNG")
                print("green m image saved")


        if nonmangrove_exists:
            saved_img = "image_nm_red.png"
            if not path.exists(saved_img):
                image_nm_red = visualize.get_im(nm_tif_filename, red_hue)
                image_nm_red.save("image_nm_red.png","PNG")
                print("red nm image saved")
            print('hi here')
        # n_clicks_mine = 0 # reset the btn

    '''    # # neither button is clicked           
    else: 
        _mngrv_geojson = {}
        _n_mngrv_geojson = {}'''
    

    dict_of_fig = get_fig(version, _mngrv_geojson, _n_mngrv_geojson, sample=sample)
    # return the figure to the graph and n_clicks to both the n_clicks of the buttons to reset them
    return dict_of_fig

'''
@app.callback(dash.dependencies.Output('viz', 'figure'),
[dash.dependencies.Input(component_id='view-mine', component_property='n_clicks'), 
dash.dependencies.Input('radiobtn', 'value')])
def update_figure(n_clicks, version):
    print('version in app callback: ', version) #  items checked in checkboxes. len: 0: nothing checked, 1: 1 item checked 2: 2 items checked. the values of the list are the values of items checked
    print('call back: ', dash.callback_context.triggered)
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    print(changed_id)
    print('n_clicks', n_clicks)

    # if you click the view my classification button, render the classfication
    if (int(n_clicks)> 0):
        print('directory: ', os.listdir())

        nonmangrove_exists = False
        mangrove_exists = False

        m_tif_filename = m_filename+'.tif'
        nm_tif_filename = nm_filename+'.tif'

        
        if (path.exists(m_tif_filename)):
            mangrove_exists = True
        
        if (path.exists(nm_tif_filename)):
            nonmangrove_exists = True


        if mangrove_exists:
            final_filename = 'mngrv.geojson'
            if path.exists(final_filename):
                print("'mngrv.geojson' exists")
                with open(final_filename) as f:
                    _mngrv_geojson = json.load(f)
        
            else:
                print("'mngrv.geojson' does not exist yet")
                _mngrv_geojson = visualize.create_geojson(m_tif_filename, final_filename)
        else: 
            _mngrv_geojson={}

        if nonmangrove_exists:
            final_filename = 'n-mngrv.geojson'
            if path.exists(final_filename):
                print("'n-mngrv.geojson' exists")
                with open(final_filename) as f:
                    _n_mngrv_geojson = json.load(f)
        
            else:
                print("'n-mngrv.geojson' does not exist yet")
                _n_mngrv_geojson = visualize.create_geojson(nm_tif_filename, final_filename)
        else: 
            _n_mngrv_geojson={}

        # display the images
        # DO SOME ELSE STATEMENT!
        if mangrove_exists:
            saved_img = "image_m_green.png"
            if not path.exists(saved_img):
                image_m_green = visualize.get_im(m_tif_filename, green_hue)
                image_m_green.save("image_m_green.png","PNG")
                print("green m image saved")


        if nonmangrove_exists:
            saved_img = "image_nm_red.png"
            if not path.exists(saved_img):
                image_nm_red = visualize.get_im(nm_tif_filename, red_hue)
                image_nm_red.save("image_nm_red.png","PNG")
                print("red nm image saved")

    # FIX THIS            
    else: 
        _mngrv_geojson = {}
        _n_mngrv_geojson = {}

    dict_of_fig = get_fig(version, _mngrv_geojson, _n_mngrv_geojson)
    print(dict_of_fig)
    return dict_of_fig
'''

if __name__ == '__main__':
    app.run_server(debug=False)
    server.run(debug=True)



