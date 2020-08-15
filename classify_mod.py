import os, shutil
import sys
import itertools
from tqdm.autonotebook import tqdm
import tensorflow as tf
import pandas as pd
import numpy as np
import json
import raster
from PIL import Image
from urllib.request import urlopen
import time
from flask import jsonify

import shutil
import requests

import PIL
import math
import azure_blob
import gc

from tensorflow.keras.models import load_model

PIL.Image.MAX_IMAGE_PIXELS = None

MAIN_DIRECTORY = os.path.dirname(os.path.realpath(__file__)) + "/"
# image directory. images/images contains the tif files
IMAGE_DIRECTORY = MAIN_DIRECTORY + "images"

UPLOAD_FOLDER = IMAGE_DIRECTORY + '/images/'



account = 'mangroveclassifier'   # Azure account name
key = 's0T0RoyfFVb/Efc+e/s1odYn2YuqmspSxwRW/c5IrQcH5gi/FpHgVYpAinDudDQuXdMFgrha38b0niW6pHzIFw=='      # Azure Storage account access key  
CONTAINER_NAME = 'mvnmv4-merced' # Container name
CONNECTION_STRING = 'DefaultEndpointsProtocol=https;AccountName=mangroveclassifier;AccountKey=s0T0RoyfFVb/Efc+e/s1odYn2YuqmspSxwRW/c5IrQcH5gi/FpHgVYpAinDudDQuXdMFgrha38b0niW6pHzIFw==;EndpointSuffix=core.windows.net'
MODEL_CONTAINER_NAME = 'mvnmv4-merced' # Container name

# add str to the begining of every element in list
def prepend(list, str): 
    # Using format() 
    str += '{0}'
    list = [str.format(i) for i in list] 
    return(list) 

#Since the original model outputs the values from the last dense layer (no final activation), we need to definte the sigmoid function for predicted class conditional probabilities
def sigmoid(x):
    return 1/(1 + np.exp(-x)) 
    

# pass in image path (file), read and save jpg in the static/images directory
def tif_to_jpg(file):
	with Image.open(file) as im:
		new_im = im.convert("RGB")
		new_file = file.rstrip(".tif")
		new_im.save(MAIN_DIRECTORY + "static/images/" + str(new_file)[-1] + ".jpg", "JPEG")

# input: list of files in the .zip files 
# output: list of batch files where each sub list is of size 32 except the last one (to account for remainders)
def get_batch_list(list_of_files, BATCH_SIZE):
    length = len(list_of_files)
    num_batches = math.floor(length/BATCH_SIZE) # number of batches with 32 files

    length_to_split = [BATCH_SIZE] * num_batches
    
    last_batch_length = length % BATCH_SIZE
    if last_batch_length != 0: 
        length_to_split.append(last_batch_length)

    from itertools import accumulate 
    batch_list = [list_of_files[x - y: x] for x, y in zip(accumulate(length_to_split), length_to_split)]
    return batch_list

def delete_files_in_dir(folder):
    if not folder.endswith('/'):
      folder += '/'
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)

            try:
                # print('deleting: ' + file_path)
                os.remove(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    return

# input the azure client
def download_model(client_model):
    client_model.download_file('mvnmv4-merced/saved_model.pb', MAIN_DIRECTORY + 'mvnmv4-merced/')
    client_model.download_file('mvnmv4-merced/variables/variables.data-00000-of-00002', MAIN_DIRECTORY + 'mvnmv4-merced/variables/')
    client_model.download_file('mvnmv4-merced/variables/variables.data-00001-of-00002', MAIN_DIRECTORY + 'mvnmv4-merced/variables/')
    client_model.download_file('mvnmv4-merced/variables/variables.index', MAIN_DIRECTORY + 'mvnmv4-merced/variables/')
    return 

def download_result_df():
    PRED_CONTAINER_NAME = 'prediction-results'
    client_pred = azure_blob.DirectoryClient(CONNECTION_STRING, PRED_CONTAINER_NAME)
    client_pred.download_file('content.csv', MAIN_DIRECTORY)

    try: 
        client_pred.rmdir('')
    except: 
        print("error when deleting from blob storage")

    return

def save_result_df(filename):
    PRED_CONTAINER_NAME = 'prediction-results'
    client_pred = azure_blob.DirectoryClient(CONNECTION_STRING, PRED_CONTAINER_NAME)
    client_pred.upload_file(filename, filename)
    return


def post_classify():

    download_result_df()

    output_container_name = 'output-files'
    client = azure_blob.DirectoryClient(CONNECTION_STRING, output_container_name)

    # KEEP THIS 
    gc.collect()

    blobs = client.ls_files(path='')
    for blob in blobs: 
        client.download_file(source=blob, dest=IMAGE_DIRECTORY+'/images/')
    
    result_df = pd.read_csv('content.csv')

    # KEEP THIS
    dest_folders = []
    # Organize tiles into folders
    for index, row in tqdm(result_df.iterrows()):
        cur_file = UPLOAD_FOLDER + row['filename']
        # cur_file = cur_file.replace("jpg","tif",2)
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

    nonmangrove_exists = False
    mangrove_exists = False
    if (os.path.isdir(IMAGE_DIRECTORY + '/1/')):
        nonmangrove_exists = True
    
    if (os.path.isdir(IMAGE_DIRECTORY + '/0/')):
        mangrove_exists = True

    # rename all the file to have .tif extension
    '''nm_img_list = list(os.listdir(IMAGE_DIRECTORY + '/1/'))
    nm_img_path = IMAGE_DIRECTORY + '/1/'
    nm_img_list = prepend(nm_img_list, nm_img_path)
    for nm_img in nm_img_list: 
        os.rename(nm_img, nm_img+'.tif')

    m_img_list = list(os.listdir(IMAGE_DIRECTORY + '/0/'))
    # list of non-mangrove tif
    m_img_path = IMAGE_DIRECTORY + '/0/'
    m_img_list = prepend(m_img_list, m_img_path)
    for m_img in m_img_list: 
        os.rename(m_img, m_img+'.tif')'''

    # recombine classified tiles for each class

    # run gdal_merge.py and prepare the argument array: !gdal_merge.py -o /content/1.tif /content/images/1/*
    # first 2 args are '-o' and '1.tif' because you want to create the file 1.tif    # list of non-mangrove tif

    # KEEP THIS
    if nonmangrove_exists: 
        nm_img_list = list(os.listdir(IMAGE_DIRECTORY + '/1/'))
        nm_img_path = IMAGE_DIRECTORY + '/1/'
        nm_img_list = prepend(nm_img_list, nm_img_path)

        raster.merge_raster(nm_img_list, output_file=MAIN_DIRECTORY+"1.tif")
        print('created 1.tif')
        os.system('gdal_polygonize.py 1.tif -f "ESRI Shapefile" -b 4 1.shp')
        tif_to_jpg(MAIN_DIRECTORY + "1.tif")
        delete_files_in_dir(IMAGE_DIRECTORY+'/1/')


    if mangrove_exists: 
        m_img_list = list(os.listdir(IMAGE_DIRECTORY + '/0/'))
        m_img_path = IMAGE_DIRECTORY + '/0/'
        m_img_list = prepend(m_img_list, m_img_path)

        raster.merge_raster(m_img_list, output_file=MAIN_DIRECTORY+"0.tif")
        print('created 0.tif')
        os.system('gdal_polygonize.py 0.tif -f "ESRI Shapefile" -b 4 0.shp')

        # store tif as jpg for visualization
        tif_to_jpg(MAIN_DIRECTORY + "0.tif")
        # Delete files in images/images
        delete_files_in_dir(IMAGE_DIRECTORY+'/0/')

    
    
    delete_files_in_dir(IMAGE_DIRECTORY+'/images/')

    # Delete the files in the blob containers 
    # remove files in output-files container
    try: 
        client.rmdir('')
    except: 
        print("error when deleting from blob storage")
    # remove files in input-files container
    input_container_name = 'input-files'
    client = azure_blob.DirectoryClient(CONNECTION_STRING, input_container_name)
    try: 
        client.rmdir('')
    except: 
        print("error when deleting from blob storage")
    
    # delete model

    print("classification finished")
    return

def classify():

    '''url = 'https://mangroveclassifier.blob.core.windows.net/output-files/'

    output_container_name = 'output-files'
    client = azure_blob.DirectoryClient(CONNECTION_STRING, output_container_name)
    list_of_files = list(client.ls_files('', recursive=False))
    print("number of tif files in output-files: ", len(list_of_files))

    # generate batches of 32 and download the files 32 at a time
    BATCH_SIZE = 32
    batch_list = get_batch_list(list_of_files, BATCH_SIZE)
    # print(batch_list)

    #Set up dataframe that will hold classifications
    column_names = ["prediction","p_0","p_1","filename"]
    result_df = pd.DataFrame(columns=column_names)
    

    # load model
    model = MAIN_DIRECTORY + "mvnmv4-merced"
    model = load_model(model)
    
    for n, batch in enumerate(batch_list[4:6]):
        # batch is a list of image names

        # create list of img urls 
        img_urls = []
        for img in batch:
            image_url = url + img
            img_urls.append(image_url)

        # Create array of 32 images
        images = []
        for img_url in img_urls:
            with urlopen(img_url) as testImage:
                image = Image.open(testImage)
                image = np.asarray(image)
                images.append(image[:, :, :3])
        
        # scale images
        images = np.array(images)/255

        # predict
        predictions = model.predict(images)



    #associate filenames and classification for each prediction
    for i,prediction in tqdm(enumerate(predictions)):
        # result_df.loc[i,"filename"] = data_gen.filenames[i]
        result_df.loc[i,"filename"] = list_of_files[i]

        #calculating predictions 
        result_df.loc[i,"p_0"] = sigmoid(prediction[0])
        result_df.loc[i,"p_1"] = sigmoid(prediction[1])
        
        #getting final class prediction
        result_df.loc[i,"prediction"] = np.argmax(prediction)'''

    '''print('result_df: ')
    print(result_df)
    print('sleeping 30 seconds')

    time.sleep(30)
    # msg = requests.get('https://predict-mangroves.azurewebsites.net/api/classify?img=https://raw.githubusercontent.com/Azure-Samples/functions-python-tensorflow-tutorial/master/resources/assets/samples/cat1.png')
    start = 0
    msg = requests.get('https://predict-mangroves.azurewebsites.net/api/classify?method=predict&start=' + start)
    print(msg)

    

    # print(msg_get)
    json_msg = msg.json()

    print('json 1st row: ', json_msg[0])
    json_msg = json.loads(json_msg)
    result_df = pd.DataFrame.from_records(json_msg)
    print(result_df.head())




    '''
    
    output_container_name = 'output-files'
    client = azure_blob.DirectoryClient(CONNECTION_STRING, output_container_name)
    list_of_files = list(client.ls_files('', recursive=False))
    print("number of tif files in output-files: ", len(list_of_files))

    # KEEP THIS
    BATCH_SIZE = 10
    BIG_BATCH_SIZE = 10

    batch_list = get_batch_list(list_of_files, BATCH_SIZE)
    n_batches = len(batch_list)
    print('number of batches:' + str(n_batches))

    big_batches = math.floor(n_batches/BIG_BATCH_SIZE)
    
    json_msg_final = []

    for bigbatch in range(big_batches):
        start = str(bigbatch * BATCH_SIZE * BIG_BATCH_SIZE)
        print('start:' + start)
        msg = requests.get('https://predict-mangroves.azurewebsites.net/api/classify?method=predict&start=' + start)
        print(msg)

        # try again
        while( msg.status_code != 200) : 
            # restart the function
            msg = requests.get('https://predict-mangroves.azurewebsites.net/api/classify?method=predict&start=' + start)
            print(msg)
                
        
        # print(msg_get)
        json_msg = msg.json()
        json_msg = json.loads(json_msg)
        # print('json 1st row: ', json_msg[0])
        print('entire: ', json_msg)
        json_msg_final.append(json_msg)
    
    json_msg_final = list(itertools.chain.from_iterable(json_msg_final))
    result_df = pd.DataFrame.from_records(json_msg_final)
    print(result_df.head())

    for n, batch in enumerate(batch_list):
        # Download all tifs in the batch
        # Memory: 0.16015625
        for i in range(len(batch)):
            client.download_file(batch[i], str(MAIN_DIRECTORY + "images/images/"))
            
        print('downsampling images')
        ds_factor = 1/10
        for rel_filename in batch:
            FILENAME = UPLOAD_FOLDER + rel_filename
            img, _ = raster.load_image(FILENAME)
            _, _ = raster.downsample_raster(img, ds_factor, FILENAME)


        # REUPLOAD DOWNSAMPLE TIFS TO DATABASE
        # memory: 0Mb
        print('reuploading batch')
        for rel_filename in batch:
            FILENAME = UPLOAD_FOLDER + rel_filename
            client.upload_file(FILENAME, rel_filename)


        # DELETE ALL TIFS IN images/images to prepare for the next batch 
        # Memory: 54.1953125 Mb 
        print('deleting images in folder')
        delete_files_in_dir(UPLOAD_FOLDER)
        
        # gc.get_stats()
        gc.collect()
    result_df.to_csv(r'content.csv')
    filename_result_df = 'content.csv'
    save_result_df(filename_result_df)
    return 
    '''
    
    # generate batches of 32 and download the files 32 at a time
    BATCH_SIZE = 32
    batch_list = get_batch_list(list_of_files, BATCH_SIZE)

    #Set up dataframe that will hold classifications
    column_names = ["prediction","p_0","p_1","filename"]
    result_df = pd.DataFrame(columns=column_names)
    
    # download model from azure
    client_model = azure_blob.DirectoryClient(CONNECTION_STRING, MODEL_CONTAINER_NAME)
    download_model(client_model)
    
    # load model
    model = MAIN_DIRECTORY + "mvnmv4-merced"
    model = load_model(model)
    
    for n, batch in enumerate(batch_list):
        
        # Download all tifs in the batch
        # Memory: 0.16015625
        for i in range(len(batch)):
            client.download_file(batch[i], str(MAIN_DIRECTORY + "images/images/"))

        #Read images using keras and split into batches
        # Memory: 0
        image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        data_gen = image_generator.flow_from_directory(directory=IMAGE_DIRECTORY,
                                                            batch_size=32,
                                                            shuffle=False,
                                                            target_size=(256, 256))


        #predict probabilities from model for the batches
        # Memory: 50.3984375 Mb
        print('predict for batch', n)
        predictions = model.predict(data_gen)
        tf.keras.backend.clear_session()

        gc.collect()



        #associate filenames and classification for each prediction
        # Memory: 0.19 Mb
        for j,prediction in tqdm(enumerate(predictions)):
            idx = (n*BATCH_SIZE) + j
            result_df.loc[idx,"filename"] = data_gen.filenames[i]

            #calculating predictions 
            result_df.loc[idx,"p_0"] = sigmoid(prediction[0])
            result_df.loc[idx,"p_1"] = sigmoid(prediction[1])
            
            #getting final class prediction
            result_df.loc[idx,"prediction"] = np.argmax(prediction)
            


        # DOWNSAMPLE ALL THE IMAGES
        # Memory: 3.43Mb
        print('downsampling images')
        ds_factor = 1/10
        for rel_filename in batch:
            FILENAME = UPLOAD_FOLDER + rel_filename
            img, _ = raster.load_image(FILENAME)
            _, _ = raster.downsample_raster(img, ds_factor, FILENAME)


        # REUPLOAD DOWNSAMPLE TIFS TO DATABASE
        # memory: 0Mb
        print('reuploading batch')
        for rel_filename in batch:
            FILENAME = UPLOAD_FOLDER + rel_filename
            client.upload_file(FILENAME, rel_filename)
        


        # DELETE ALL TIFS IN images/images to prepare for the next batch 
        # Memory: 54.1953125 Mb 
        print('deleting images in folder')
        delete_files_in_dir(UPLOAD_FOLDER)
        
        # gc.get_stats()
        gc.collect()
        
    '''
    
    '''
    print('WAITING 20 seconds')
    time.sleep(20)
    print('finished waiting 20 seconds')'''
    # DOWNLOAD ALL files in output blob in the hash folder 
    # to fix this issue, ask the user for the prefix of their files? idk...
    

    '''shutil.rmtree(MAIN_DIRECTORY+'mvnmv4-merced/')
    os.mkdir(MAIN_DIRECTORY+'mvnmv4-merced/')

    # delete_files_in_dir(MAIN_DIRECTORY+'mvnmv4-merced/')
    os.mkdir(MAIN_DIRECTORY+'mvnmv4-merced/variables/')'''
    

    
'''
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
   
    #probability tiles remain unmoved, so just get all the leftover tiles
    # run gdal_merge.py and prepare the argument array:     !gdal_merge.py -o /content/p.tif /content/images/images/*
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
'''
