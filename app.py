import os
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
import base64

from descartes import PolygonPatch
from rasterio.plot import show
import matplotlib as mpl
import geopandas
import fiona

import rasterio.features
from geojson import Point, Feature, FeatureCollection, dump

import geopandas

from PIL import Image
import PIL
PIL.Image.MAX_IMAGE_PIXELS = None

from tensorflow.keras.models import load_model

# dash 
import dash
import dash_core_components as dcc
import dash_html_components as html


#Set model location
output_file = "results.csv"
#store the model

# TO DO: ADD OS FUNCTIONS TO MAKE IT RUN ON ANYONE'S COMPUTER
print('MAIN_DIRECTORY from OS: ', os.path.dirname(os.path.realpath(__file__)))

MAPBOX_APIKEY = "pk.eyJ1Ijoibm1laXN0ZXIiLCJhIjoiY2tjODZya3VnMHU0cjJ2bGpxanh0eW9idiJ9._DNCB5IcbFoGl7AIm0vVlA"

# overall directory where all files and folders are stored 
MAIN_DIRECTORY = os.path.dirname(os.path.realpath(__file__)) + "/"

# model path 
# MODEL_PATH = MAIN_DIRECTORY + "mangrove_model.h5"
MODEL_PATH = MAIN_DIRECTORY + "mvnmv4_merced_bright.zip"
 
# image directory. images/images contains the tif files
IMAGE_DIRECTORY = MAIN_DIRECTORY + "images"

# upload path 
UPLOAD_FOLDER = IMAGE_DIRECTORY + '/images/'

ALLOWED_EXTENSIONS = {'zip', 'tif'}

sys.path.append(MAIN_DIRECTORY)
import gdal_merge as gm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("TF version:", tf.__version__)
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")
server = Flask(__name__)

server.config['SECRET_KEY'] = "it is a secret" # old code idk if I need this
server.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# unzip model zip file
os.system("unzip -n " + MODEL_PATH)
model = MAIN_DIRECTORY + "mvnmv4_merced"

model = load_model(model)

#model = keras.models.load_model(MAIN_DIRECTORY + 'mvnmv4_merced_model.h5')
#model.summary()

# old code idk if i need this
'''def file_exists():
    path = request.form.get('file_path')
    if os.path.isfile(path):
        print("File exists!",file=sys.stderr)
        return True
    print("File not exists!",file=sys.stderr)
    return False'''

# check for allowed file extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#function for fixing shapefiles to only create polygons around the the specified class
def fix_shp(filename):
    shp = geopandas.read_file(filename)
    for index, feature in tqdm(shp.iterrows()):
        if feature["DN"] == 0:
            shp.drop(index, inplace=True)
    shp.to_file(filename)
    return shp

# pass in image path (file), read and save jpg in the static/images directory
def tif_to_jpg(file):
	with Image.open(file) as im:
		new_im = im.convert("RGB")
		new_file = file.rstrip(".tif")
		new_im.save(MAIN_DIRECTORY + "static/images/" + str(new_file)[-1] + ".jpg", "JPEG")

# add str to the begining of every element in list
def prepend(list, str): 
    # Using format() 
    str += '{0}'
    list = [str.format(i) for i in list] 
    return(list) 

#Since the original model outputs the values from the last dense layer (no final activation), we need to definte the sigmoid function for predicted class conditional probabilities
def sigmoid(x):
    return 1/(1 + np.exp(-x)) 


@server.route('/')
def home():
    return render_template('index.html')
@server.route('/index')
def index():
    return render_template('index.html')

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

    # TO DO: Allow user to download the .tif (maybe .shp) files?

    # Delete all the files uploaded + created 
    os.system('rm -rf ' + MAIN_DIRECTORY + 'images/*')
    print('mkdir ' + IMAGE_DIRECTORY + '/images/')
    # recreate images/images subfolder
    os.system('mkdir ' + IMAGE_DIRECTORY + '/images/')
    print('mkdir ' + IMAGE_DIRECTORY + '/images/')
    
    # list of files to delete so each user can start with same file structure
    files_to_del = list(['0.prj', '1.prj', '0.tif', '1.tif', '0.shx', '1.shx', '0.shp', '1.cpg', '1.shp', '1.dbf', '0.dbf', '1.jpg', '0.jpg'])
    for f in files_to_del:
        os.system('rm -rf ' + MAIN_DIRECTORY + f)
        print('rm -rf ' + MAIN_DIRECTORY + f)

    # delete the files in static images
    for f in os.listdir('static/images/'):
        os.system('rm -rf ' + MAIN_DIRECTORY + 'static/images/' + f)
        print('rm -rf ' + MAIN_DIRECTORY + 'static/images/' + f)


    html = render_template('index.html')
    response = make_response(html)
    return response

@server.route('/upload', methods=['POST'])
def upload():

    print('in upload')
    # download() delete all the images from the prev instance of running the website

    # check if the post request has the file part
    if 'file' not in request.files:
        print('No file part')
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        print('No Selected file')
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        print('filename:', filename)
        file.save(os.path.join(server.config['UPLOAD_FOLDER'], filename))

    html = render_template('index.html')
    response = make_response(html)
    return response

# create geojson file to store the polygon
def create_geojson(FILENAME, final_filename):
    # list of GeoJSON feature objects (later this becomes a FeatureCollection)
    features = []

    print('File that is being opened: ', FILENAME)
    dataset = rasterio.open(os.path.abspath(FILENAME))

    # Read the dataset's valid data mask as a ndarray.
    mask = dataset.dataset_mask()

    # Extract feature shapes and values from the array.
    for geom, val in rasterio.features.shapes(mask, transform=dataset.transform):
    # val is the value of the raster feature corresponding to the shape
    # val = 0: no shape and val = 255 means shape (drone footage, aka tiles we want)
        if (val == 255.0):  

            # Transform shapes from the dataset's own coordinate reference system to CRS84 (EPSG:4326) tbh idk what this means
            geom = rasterio.warp.transform_geom(dataset.crs, 'EPSG:4326', geom, precision=30)

            # store GeoJSON shapes to features list.
            # might have to put the probabilty value in properties ... tbd
            features.append(Feature(geometry=geom, properties={'name': FILENAME}))

    # all features become a feature collection
    feature_collection = FeatureCollection(features)

    '''# Feature collection goes into a geojson file
    with open(final_filename, 'w') as f:
        dump(feature_collection, f)'''
    
    return feature_collection


@server.route('/visualization', methods=['GET'])
def visualization():

    '''# open the tif image and create geojson file
    FILENAME = '0.tif'
    final_filename = 'mngrv.geojson'
    mngrv_geojson = create_geojson(FILENAME, final_filename)

    # open the tif image and create geojson file
    FILENAME = '1.tif'
    final_filename = 'n-mngrv.geojson'
    n_mngrv_geojson = create_geojson(FILENAME, final_filename)

    mngrv_tiles = len(mngrv_geojson['features'])
    n_mngrv_tiles = len(n_mngrv_geojson['features'])

    sources = visualize.make_sources(mngrv_geojson) # dictionary to create the borders
    lons, lats = visualize.get_centers(mngrv_geojson) # lat lon of all the centers of the scatter plot (to mimic hover text effect)
    latmin_m, latmax_m, lonmin_m, lonmax_m = visualize.get_latlonminmax(mngrv_geojson)
    latmin_nm, latmax_nm, lonmin_nm, lonmax_nm = visualize.get_latlonminmax(n_mngrv_geojson)

    # use a scatter map box to create the hover text
    data = dict(type='scattermapbox',
                lat=lats,
                lon=lons,
                mode='markers',
                text='mangrove',
                showlegend=False,
                hoverinfo='text'
                )
    
    green_hue = (180-78)/360.0
    red_hue = (180-180)/360.0

    ds_factor = 8

    FILENAME = '0.tif'
    image_m_green = visualize.get_im(FILENAME, ds_factor, green_hue)

    FILENAME = '1.tif'
    image_nm_red = visualize.get_im(FILENAME, ds_factor, red_hue)

    layers=([dict(sourcetype = 'geojson',
              source =sources[k],
              below="",
              type = 'line',    # the borders
              line = dict(width = 2),
              color = 'black',
              ) for k in range(n_tiles)
          ] +
        [dict(below ='',
                      opacity=0.9,
                  source = image_m_green,
                  sourcetype= "image",
                  coordinates =  [
                          [lonmin_m, latmax_m], [lonmax_m, latmax_m], [lonmax_m, latmin_m], [lonmin_m, latmin_m]
                                 ])] + 
           [dict(below ='',
                      opacity=0.9,
                  source = image_nm_red,
                  sourcetype= "image",
                  coordinates =  [
                          [lonmin_nm, latmax_nm], [lonmax_nm, latmax_nm], [lonmax_nm, latmin_nm], [lonmin_nm, latmin_nm]
                                 ])]
                     )

    avg_lon = np.average(lons)
    avg_lat = np.average(lats)

    layout = dict(title="Visualization of Mangrove CNN",
                autosize=False,
                width=700,
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
                            style = 'mapbox://styles/mapbox/outdoors-v11'
                            )
                )
    '''
    return redirect(url_for('/visualization/_dash-update-component'))

@server.route('/download_m', methods=['GET'])
def download_m():
    filename = '1.tif'
    return redirect(url_for('uploaded_file', filename=filename))

@server.route('/', methods=['GET'])
def download_nm():
    filename = '0.tif'
    return redirect(url_for('uploaded_file', filename=filename))

@server.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(MAIN_DIRECTORY,
                               filename)

@server.route('/unzip', methods=['GET'])
def unzip():

    uploaded_file_name = os.listdir('images/images')[0]
    os.system("unzip " + UPLOAD_FOLDER + uploaded_file_name)
    print("unzip " + UPLOAD_FOLDER + uploaded_file_name)

    os.system("rm -rf " + UPLOAD_FOLDER + uploaded_file_name)
    print("rm -rf " + UPLOAD_FOLDER + uploaded_file_name)

    # !mv *.tif /content/images/images/
    os.system("mv " + "*.tif " + UPLOAD_FOLDER)
    print("mv " + "*.tif " + UPLOAD_FOLDER)


    html = render_template('index.html')
    response = make_response(html)
    return response



@server.route('/classify', methods=['GET'])
def classify():
    print(request.args.get('author'))

    #Read images using keras and split into batches
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    data_gen = image_generator.flow_from_directory(directory=IMAGE_DIRECTORY,
                                                        batch_size=32,
                                                        shuffle=False,
                                                        target_size=(256, 256))

    #Set up dataframe that will hold classifications
    column_names = ["prediction","p_0","p_1","filename"]
    result_df = pd.DataFrame(columns=column_names)

    #predict probabilities from model for the batches
    predictions = model.predict(data_gen)

    print(predictions.shape)

    #associate filenames and classification for each prediction
    for i,prediction in tqdm(enumerate(predictions)):
        result_df.loc[i,"filename"] = data_gen.filenames[i]

        #calculating predictions 
        result_df.loc[i,"p_0"] = sigmoid(prediction[0])
        result_df.loc[i,"p_1"] = sigmoid(prediction[1])
        
        #getting final class prediction
        result_df.loc[i,"prediction"] = np.argmax(prediction)
        # print(i, prediction)


    # GENERATING PROBABILITY TILES
    for index, sample in tqdm(result_df.iterrows()):
        # loading original image
        original = os.path.abspath(os.path.join(IMAGE_DIRECTORY, sample["filename"]))
        # print('original: ', original)
        img = rasterio.open(original) 

        #creating new raster mask with pixel values of conditional probability
        mask = sample["p_0"] * np.ones(shape=(img.width, img.height))

        #saving file output to new file
        filename = "prob_" + os.path.basename(sample["filename"])
        output = os.path.abspath(os.path.join(IMAGE_DIRECTORY, os.path.dirname(sample["filename"]), filename))
        # print('output: ', output)
        #creates new file with projection of past image
        with rasterio.open(output,'w',driver='GTiff',height=img.height,width=img.width,count=1,dtype=mask.dtype,crs='+proj=latlong',transform=img.transform,) as dst:dst.write(mask, 1)
    print('probability tiles generated')
    # Moving Files to Folders

    dest_folders = []
    #Organize tiles into folders
    for index, row in tqdm(result_df.iterrows()):
        cur_file = IMAGE_DIRECTORY + "/" + row['filename']
        cur_file = cur_file.replace("jpg","tif",2)
        classification = row['prediction'] 

        #set destination folder, and creates the folder if it doesn't exist
        dest_folder = os.path.join(os.path.abspath(IMAGE_DIRECTORY),str(classification))
        dest_folders.append(dest_folder)
        if os.path.exists(dest_folder) == False:
            os.mkdir(dest_folder)
        dest = os.path.join(dest_folder,os.path.basename(cur_file))
    
        #moves file
        src = cur_file
        os.rename(src, dest)
    print('organized into folders')

    # recombine classified tiles for each class

    # run gdal_merge.py and prepare the argument array: !gdal_merge.py -o /content/1.tif /content/images/1/*
    # first 2 args are '-o' and '1.tif' because you want to create the file 1.tif    # list of non-mangrove tif
    nm_img_list = list(os.listdir(IMAGE_DIRECTORY + '/1/'))
    nm_img_path = IMAGE_DIRECTORY + '/1/'
    nm_img_list = prepend(nm_img_list, nm_img_path)

    files_string = " ".join(nm_img_list)
    # concat to create complete list of args
    command = "gdal_merge.py -o " + MAIN_DIRECTORY + "1.tif " + files_string
    os.system(command)
    print('ran !gdal_merge.py -o /content/1.tif /content/images/1/*')

    # TO DO: Put the next 3 blocks into functions

    # run gdal_merge.py and prepare the argument array: !gdal_merge.py -o /content/0.tif /content/images/0/*
    # first 2 args are '-o' and '0.tif' because you want to create the file 0.tif
    gdal_merge_args = []
    gdal_merge_args = list(['-o', str(MAIN_DIRECTORY + '0.tif')])
    # list of non-mangrove tif
    m_img_list = list(os.listdir(IMAGE_DIRECTORY + '/0/'))
    m_img_path = IMAGE_DIRECTORY + '/0/'
    m_img_list = prepend(m_img_list, m_img_path)
    files_string = " ".join(m_img_list)
    # concat to create complete list of args
    command = "gdal_merge.py -o " + MAIN_DIRECTORY + "0.tif " + files_string
    os.system(command)
    print('ran !gdal_merge.py -o /content/0.tif /content/images/0/*')

   
    #probability tiles remain unmoved, so just get all the leftover tiles
    # run gdal_merge.py and prepare the argument array:     !gdal_merge.py -o /content/p.tif /content/images/images/*'''
    # first 2 args are '-o' and '0.tif' because you want to create the file 0.tif
    gdal_merge_args = []
    gdal_merge_args = list(['-o', str(MAIN_DIRECTORY + 'p.tif')])
    # list of prob tif
    prob_img_list = list(os.listdir(IMAGE_DIRECTORY + '/images/'))
    prob_img_path = IMAGE_DIRECTORY + '/images/'
    prob_img_list = prepend(prob_img_list, prob_img_path)
    files_string = " ".join(prob_img_list)
    # concat to create complete list of args
    command = "gdal_merge.py -o " + MAIN_DIRECTORY + "p.tif " + files_string
    os.system(command)
    print('ran !gdal_merge.py -o /content/p.tif /content/images/images/*')

    # TO DO HERE: STORE 1.TIF 0.TIF P.TIF SOMEWHERE

    # create .shp files
    os.system('gdal_polygonize.py 1.tif -f "ESRI Shapefile" -b 4 1.shp')
    os.system('gdal_polygonize.py 0.tif -f "ESRI Shapefile" -b 4 0.shp')
    print('ran gdal_polygonize')

    # store tif as jpg for visualization
    tif_to_jpg(MAIN_DIRECTORY + "1.tif")
    tif_to_jpg(MAIN_DIRECTORY + "0.tif")

    '''# create visualizations -- CRASHES
    # save the plot into 1.png
    src = rasterio.open("1.tif")
    shp = fix_shp("1.shp")
    fig, ax = plt.subplots(figsize=(5, 10))
    rasterio.plot.show(src, ax=ax)
    shp.plot(ax=ax, facecolor='green', edgecolor='red', alpha=0.25)
    plt.savefig('1.png')
    print('created 1.png')

    # save the plot into 0.png
    src = rasterio.open("0.tif")
    shp = fix_shp("0.shp")
    fig, ax = plt.subplots(figsize=(5, 10))
    rasterio.plot.show(src, ax=ax)
    shp.plot(ax=ax, facecolor='green', edgecolor='red', alpha=0.25)
    plt.savefig('0.png')

    # save the plot into p.png
    src = rasterio.open("p.tif")
    rasterio.plot.show(src,cmap='viridis')
    plt.savefig('p.png')'''
        
    html = render_template('index.html')
    response = make_response(html)
    return response
    
'''@app.before_request
def require_login():

    allowed_routes = ['login']
    if request.endpoint not in allowed_routes:
        return redirect('/login')
'''

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, server=server, routes_pathname_prefix='/visualization/')

# open the tif image and create geojson file
FILENAME = '0.tif'
final_filename = 'mngrv.geojson'
mngrv_geojson = create_geojson(FILENAME, final_filename)

# open the tif image and create geojson file
FILENAME = '1.tif'
final_filename = 'n-mngrv.geojson'
n_mngrv_geojson = create_geojson(FILENAME, final_filename)

mngrv_tiles = len(mngrv_geojson['features'])
n_mngrv_tiles = len(n_mngrv_geojson['features'])

sources = visualize.make_sources(mngrv_geojson) # dictionary to create the borders
print('got sources')
lons, lats = visualize.get_centers(mngrv_geojson) # lat lon of all the centers of the scatter plot (to mimic hover text effect)
print('got lats lons')
latmin_m, latmax_m, lonmin_m, lonmax_m = visualize.get_latlonminmax(mngrv_geojson)
print('got min max lats lons')
latmin_nm, latmax_nm, lonmin_nm, lonmax_nm = visualize.get_latlonminmax(n_mngrv_geojson)

# use a scatter map box to create the hover text
data = dict(type='scattermapbox',
            lat=lats,
            lon=lons,
            mode='markers',
            opacity = 0,
            text='mangrove',
            showlegend=False,
            hoverinfo='text'
            )

layers=([dict(sourcetype = 'geojson',
            source =sources[k],
            below="",
            type = 'line',    # the borders
            line = dict(width = 2),
            color = 'black',
            ) for k in range(mngrv_tiles)
        ])

green_hue = (180-78)/360.0
red_hue = (180-180)/360.0

ds_factor = 8

FILENAME = '0.tif'
image_m_green = visualize.get_im(FILENAME, ds_factor, green_hue)
image_m_green.save("image_m_green.png","PNG")
print("green m image saved")

FILENAME = '1.tif'
image_nm_red = visualize.get_im(FILENAME, ds_factor, red_hue)
image_nm_red.save("image_nm_red.png","PNG")
print("red nm image saved")

image_filename = "image_m_green.png"
image_m_green = base64.b64encode(open(image_filename, 'rb').read())

image_filename = "image_nm_red.png"
image_nm_red = base64.b64encode(open(image_filename, 'rb').read())

avg_lon = np.average(lons)
avg_lat = np.average(lats)


'''layers = ([dict(below ='',
                    opacity=0.9,
                source = 'data:image/png;base64,{}'.format(image_m_green.decode()), # 'https://docs.mapbox.com/mapbox-gl-js/assets/radar.gif',
                sourcetype= "image",
                coordinates =  [
                        [lonmin_m, latmax_m], [lonmax_m, latmax_m], [lonmax_m, latmin_m], [lonmin_m, latmin_m]
                                ])])'''
'''layers=([dict(sourcetype = 'geojson',
            source =sources[k],
            below="",
            type = 'line',    # the borders
            line = dict(width = 2),
            color = 'black',
            ) for k in range(mngrv_tiles)
        ]  +
    [dict(below ='',
                    opacity=0.9,
                source = "image_m_green.png", # 'https://docs.mapbox.com/mapbox-gl-js/assets/radar.gif',
                sourcetype= "image",
                coordinates =  [
                        [lonmin_m, latmax_m], [lonmax_m, latmax_m], [lonmax_m, latmin_m], [lonmin_m, latmin_m]
                                ])])'''

layers=([dict(sourcetype = 'geojson',
            source =sources[k],
            below="",
            type = 'line',    # the borders
            line = dict(width = 2),
            color = 'black',
            ) for k in range(mngrv_tiles)
        ]  +
    [dict(below ='',
                    opacity=1,
                source = 'data:image/png;base64,{}'.format(image_m_green.decode()),
                sourcetype= "image",
                coordinates =  [
                        [lonmin_m, latmax_m], [lonmax_m, latmax_m], [lonmax_m, latmin_m], [lonmin_m, latmin_m]
                                ])] + 
        [dict(below ='',
                    opacity=1,
                source = 'data:image/png;base64,{}'.format(image_nm_red.decode()),
                sourcetype= "image",
                coordinates =  [
                        [lonmin_nm, latmax_nm], [lonmax_nm, latmax_nm], [lonmax_nm, latmin_nm], [lonmin_nm, latmin_nm]
                                ])]
                    )


layout = dict(title="Visualization of Mangrove CNN",
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
            )

dict_of_fig = dict(data=[data], layout=layout)
app.layout = html.Div(children=[
    dcc.Checklist(id='radiobtn', 
    options=[
        {'label': 'Mangrove', 'value': 'mangrove'},
        {'label': 'Non-Mangrove', 'value': 'non-mangrove'},
        # {'label': 'Everything', 'value': 'everything'},
        {'label': 'Probability', 'value': 'prob'}
    ],
    value=['mangrove', 'non-mangrove'],
    labelStyle={'display': 'inline-block', 'textAlign': 'center'}
)  , 
    dcc.Graph(id='viz', figure=dict_of_fig)
    ])

def get_fig(version):
    # use a scatter map box to create the hover text
    data = dict(type='scattermapbox',
                lat=lats,
                lon=lons,
                opacity = 0,
                mode='markers',
                text='mangrove',
                showlegend=False,
                hoverinfo='text'
                )

    border = [dict(sourcetype = 'geojson',
                source =sources[k],
                below="",
                type = 'line',    # the borders
                line = dict(width = 2),
                color = 'black',
                ) for k in range(mngrv_tiles)
            ]
    mangrove = [dict(below ='',
                        opacity=1,
                    source = 'data:image/png;base64,{}'.format(image_m_green.decode()),
                    sourcetype= "image",
                    coordinates =  [
                            [lonmin_m, latmax_m], [lonmax_m, latmax_m], [lonmax_m, latmin_m], [lonmin_m, latmin_m]
                                    ])]
    non_mangrove = [dict(below ='',
                        opacity=1,
                    source = 'data:image/png;base64,{}'.format(image_nm_red.decode()),
                    sourcetype= "image",
                    coordinates =  [
                            [lonmin_nm, latmax_nm], [lonmax_nm, latmax_nm], [lonmax_nm, latmin_nm], [lonmin_nm, latmin_nm]
                                    ])]
    print('version:', version)
    if len(version) == 0:
        layers=(border)
    if len(version) == 1:
        if (version[0] == 'mangrove'):
            layers=(border+mangrove)
        # gotta add probability
        # elif (version[0] == 'non-mangrove'):
        else:
            layers=(border+non_mangrove)
    # everything
    if len(version)==2:
        layers=(border+mangrove+non_mangrove)
    # version =='prob'

    layout = dict(title="Visualization of Mangrove CNN",
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
                )

    dict_of_fig = dict(data=[data], layout=layout)
    return dict_of_fig


@app.callback(
    dash.dependencies.Output('viz', 'figure'),
    [dash.dependencies.Input('radiobtn', 'value')])
def update_figure(version):
    print(version)
    dict_of_fig = get_fig(version)
    return dict_of_fig


if __name__ == '__main__':
    app.run_server(debug=True)
    server.run(debug=True)


