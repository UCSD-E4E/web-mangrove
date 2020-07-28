import os, shutil
import sys
from tqdm.autonotebook import tqdm
import tensorflow as tf
import pandas as pd
import numpy as np
import raster
from PIL import Image

import memory_profiler

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
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)

        try:
            '''if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)'''
            print('deleting: ' + file_path)
            os.remove(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    return
    

def classify():

    '''# download model from azure
    client_model = azure_blob.DirectoryClient(CONNECTION_STRING, MODEL_CONTAINER_NAME)
    client_model.download('mvnmv4-merced/', MAIN_DIRECTORY)'''

    # load model
    model = MAIN_DIRECTORY + "mvnmv4-merced"
    model = load_model(model)

    output_container_name = 'output-files'
    client = azure_blob.DirectoryClient(CONNECTION_STRING, output_container_name)
    list_of_files = list(client.ls_files('', recursive=False))
    print("number of tif files in output-files: ", len(list_of_files))
    
    # generate batches of 32 and download the files 32 at a time
    BATCH_SIZE = 32
    batch_list = get_batch_list(list_of_files, BATCH_SIZE)

    #Set up dataframe that will hold classifications
    column_names = ["prediction","p_0","p_1","filename"]
    result_df = pd.DataFrame(columns=column_names)

    for n, batch in enumerate(batch_list):
        m1 = memory_profiler.memory_usage()
        # Download all tifs in the batch
        # Memory: 0.16015625
        for i in range(len(batch)):
            client.download_file(batch[i], str(MAIN_DIRECTORY + "images/images/"))
            
        m2 = memory_profiler.memory_usage()
        mem_diff = m2[0] - m1[0]
        print(f"It took {mem_diff} Mb to execute this method")

        #Read images using keras and split into batches
        # Memory: 0
        image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        data_gen = image_generator.flow_from_directory(directory=IMAGE_DIRECTORY,
                                                            batch_size=32,
                                                            shuffle=False,
                                                            target_size=(256, 256))
        m3 = memory_profiler.memory_usage()
        mem_diff = m3[0] - m2[0]
        print(f"It took {mem_diff} Mb to execute this method")

        #predict probabilities from model for the batches
        # Memory: 50.3984375 Mb
        print('predict for batch', n)
        predictions = model.predict(data_gen)

        m4 = memory_profiler.memory_usage()
        mem_diff = m4[0] - m3[0]
        print(f"It took {mem_diff} Mb to execute this method")


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

        m5 = memory_profiler.memory_usage()
        mem_diff = m5[0] - m4[0]
        print(f"It took {mem_diff} Mb to execute this method")

        # DOWNSAMPLE ALL THE IMAGES
        # Memory: 3.43Mb
        print('downsampling images')
        ds_factor = 1/10
        for rel_filename in batch:
            FILENAME = UPLOAD_FOLDER + rel_filename
            img, _ = raster.load_image(FILENAME)
            _, _ = raster.downsample_raster(img, ds_factor, FILENAME)

        m6 = memory_profiler.memory_usage()
        mem_diff = m6[0] - m5[0]
        print(f"It took {mem_diff} Mb to execute this method")


        # REUPLOAD DOWNSAMPLE TIFS TO DATABASE
        # memory: 0Mb
        print('reuploading batch')
        for rel_filename in batch:
            FILENAME = UPLOAD_FOLDER + rel_filename
            client.upload_file(FILENAME, rel_filename)
        

        m7 = memory_profiler.memory_usage()
        mem_diff = m7[0] - m6[0]
        print(f"It took {mem_diff} Mb to execute this method")


        # DELETE ALL TIFS IN images/images to prepare for the next batch 
        # Memory: 54.1953125 Mb 
        print('deleting images in folder')
        delete_files_in_dir(UPLOAD_FOLDER)
        m8 = memory_profiler.memory_usage()
        mem_diff = m8[0] - m1[0]
        print(f"It took {mem_diff} Mb to execute this method")
        
        gc.get_stats()
        # gc.collect()
    
    # DOWNLOAD ALL files in output blob in the hash folder 
    # to fix this issue, ask the user for the prefix of their files? idk...
    client.download(source='', dest=IMAGE_DIRECTORY+'/images')
    

    # result_df = pd.read_csv('content.csv') # TEMP!

    dest_folders = []
    # Organize tiles into folders
    for index, row in tqdm(result_df.iterrows()):
        cur_file = IMAGE_DIRECTORY + "/" + row['filename']
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

    # create .shp files
    os.system('gdal_polygonize.py 1.tif -f "ESRI Shapefile" -b 4 1.shp')
    os.system('gdal_polygonize.py 0.tif -f "ESRI Shapefile" -b 4 0.shp')
    print('ran gdal_polygonize')

    # store tif as jpg for visualization
    tif_to_jpg(MAIN_DIRECTORY + "1.tif")
    tif_to_jpg(MAIN_DIRECTORY + "0.tif")

    # Delete files in images/images
    delete_files_in_dir(IMAGE_DIRECTORY+'/0/')
    delete_files_in_dir(IMAGE_DIRECTORY+'/1/')
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
    delete_files_in_dir(MAIN_DIRECTORY+'mvnmv4-merced/')
    os.mkdir(MAIN_DIRECTORY+'mvnmv4-merced/variables')
    print("classification finished")
    return
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

