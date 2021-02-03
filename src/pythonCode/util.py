import skimage.io
import skimage.transform
import numpy as np
import matplotlib as plt
from PIL import Image

def chunker(seq, size):
    # http://stackoverflow.com/a/25701576/1189865
    return (seq[pos:pos + size] for pos in xrange(0, len(seq), size))

def load_image(path):
    try:
        img = skimage.io.imread( path ).astype(np.float)
        print(img)
        print(img.shape)
        print(type(img))
        img /= 255.0
        print(img)
        X = img.shape[0]        # number of rows (= height of the image)
        Y = img.shape[1]        # number of columns (= width of the image)
        print(X, Y)
        # make the image square 
        if (X<Y):       # l'immagine è più alta che larga
            new_img = np.zeros((Y, Y, 3))
            print(new_img.shape)
            # we have to increase the number of rows
            missing_rows = Y-X
            print("missing_rows= ", missing_rows)
            missing_pixels = np.array(np.zeros((missing_rows, Y, 3)))
            print(missing_pixels.shape)
            print(img.shape)
            np.concatenate((missing_pixels, img), axis = 0, out=new_img)
            print(new_img.shape)
            img = new_img
            print("arriva qua")
        elif (X>Y):     # l'immagine è più larga che alta
            new_img = np.zeros((X, X, 3))
            print(new_img.shape)
            # we have to increase the number of rows
            missing_columns = X-Y
            print("missing_columns= ", missing_columns)
            missing_pixels = np.array(np.zeros((X, missing_columns, 3)))
            print(missing_pixels.shape)
            print(img.shape)
            np.concatenate((missing_pixels, img), axis = 1, out=new_img)
            print(new_img.shape)
            img = new_img
            print("arriva qua")
        print(img.shape)
        print(img)
        # X = img.shape[0]        # number of rows (= height of the image)
        # Y = img.shape[1]        # number of columns (= width of the image)
        # S = min(X,Y)
        # XX = int((X - S) / 2)
        # YY = int((Y - S) / 2)
    except:
        return Exception("You need skimage to load the image")

    # if black and white image, repeat the channels
    if len(img.shape) == 2: img=np.tile(img[:,:,None], 3)
    #return skimage.transform.resize( img[XX:XX+S, YY:YY+S], [224,224] )
    return skimage.transform.resize( img, [224,224] )

def load_single_image(image):
    return np.expand_dims(load_image(image),0)



def array2PIL(arr):
    mode = 'RGBA'
    shape = arr.shape
    arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])
    if len(arr[0]) == 3:
        arr = np.c_[arr, 255*np.ones((len(arr),1), np.uint8)]

    return Image.frombuffer(mode, (shape[1], shape[0]), arr.tostring(), 'raw', mode, 0, 1)


def normalize(x):
    min = np.min(x)
    max = np.max(x)
    return (x-min)/(max-min)
