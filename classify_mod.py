import os
import sys
from tqdm.autonotebook import tqdm
import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image
import PIL
PIL.Image.MAX_IMAGE_PIXELS = None



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

def classify(model, IMAGE_DIRECTORY, MAIN_DIRECTORY):

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
