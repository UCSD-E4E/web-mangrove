import rasterio
from rasterio.enums import Resampling

'''
load_image

Inputs-

file(str): full path of image/orthmosaic

Outputs-

img: generator of original image/orthomosaic
meta(dict): contains meta information from image, including location, size, etc. 

'''

#for loading orthomosaic into memory 
def load_image(file):
    img = rasterio.open(file)
    meta = img.meta.copy()
    return img, meta


def downsample_raster(dataset, downscale_factor, out_file=None):
    
    # resample data to target shape
    resampled = dataset.read(
        out_shape=(
            dataset.count,
            int(dataset.height * downscale_factor),
            int(dataset.width * downscale_factor)
        ),
        resampling= rasterio.enums.Resampling.nearest
    )

    transform = dataset.transform * dataset.transform.scale(
        (dataset.width / resampled.shape[-1]),
        (dataset.height / resampled.shape[-2])
    )

    #if there is more than one band, output numpy array size will be (1,i,j,k), so we need to flatten the array
    if dataset.count > 1:
        resampled = resampled.squeeze()

    #writing file
    if out_file != None:
        with rasterio.open(out_file,'w',driver='GTiff',height=int(dataset.height * downscale_factor),width=int(dataset.width * downscale_factor),count=dataset.count,dtype=resampled.dtype,crs='+proj=latlong',transform=transform,) as dst:
            dst.write(resampled)

    return resampled, transform