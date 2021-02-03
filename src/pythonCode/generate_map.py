from __future__ import division


import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from util import load_single_image, normalize

if os.path.exists(sys.argv[1]): 
    image = load_single_image(sys.argv[1])
    folder = sys.argv[1]
    folder = folder[:-18]
else: 
    raise Exception("Image file " + sys.argv[1] + " does not exist")

# this is to ensure fast failure, why load other modules if no input file
# matplotlib has to be loaded before inorder to change backend

import pandas as pd
import numpy as np
from model import CNN
from params import HyperParams
import skimage.io


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
hyper = HyperParams(verbose=False)
images_tf = tf.placeholder(tf.float32, [None, hyper.image_h, hyper.image_w, hyper.image_c], name="images")
class_tf  = tf.placeholder(tf.int64, [None], name='class')

cnn = CNN()
if hyper.fine_tuning: 
    cnn.load_vgg_weights()

conv_last, gap, class_prob = cnn.build(images_tf)
classmap = cnn.get_classmap(class_tf, conv_last)

with tf.Session() as sess:
    tf.train.Saver().restore( sess, hyper.model_path )
    conv_last_val, class_prob_val = sess.run([conv_last, class_prob], feed_dict={images_tf: image})

    # use argsort instead of argmax to get all the classes
    class_predictions_all = class_prob_val.argsort(axis=1)

    roi_map = None
    for i in range(-1 * hyper.top_k,0):
        current_class = class_predictions_all[:,i]
        classmap_vals = sess.run(classmap, feed_dict={class_tf: current_class, conv_last: conv_last_val})
        normalized_classmap = normalize(classmap_vals[0])
        
        if roi_map is None:
            roi_map = 1.2 * normalized_classmap 
        else:
            # simple exponential ranking
            roi_map = (roi_map + normalized_classmap)/2
    roi_map = normalize(roi_map)    

print(roi_map)
# Plot the heatmap on top of image
fig, ax = plt.subplots(1, 1, figsize=(12, 9))
ax.margins(0)
plt.axis('off')
plt.imshow( roi_map, cmap=plt.cm.jet, interpolation='nearest' )
plt.imshow( image[0], alpha=0.4)
plt.savefig('immagine_1/overlayed_heatmap.png')
print(folder)



df = pd.DataFrame(data=roi_map.astype(float))
#df = pd.read_csv('/home/andrea/Desktop/tensorDecompositionProject/immagine_1/roi_map.csv')
plt.imshow( df, cmap=plt.cm.jet, interpolation='nearest' )
plt.imshow( image[0], alpha=0.4)
plt.savefig('immagine_1/overlayed_heatmap.png')

df.to_csv('immagine_1/roi_map.csv', sep = ',', header = False, index = False)

print(df.shape)

k = int(224/8)
print(k)
for i in range(8):
    for j in range(8):
        # media della zona (i, j)
        print("i, j = ",i ," ", j, "\t")
        media = pd.DataFrame.mean( df.iloc[(i*k ):((i+1)*k-1),  (j*k + 1):((j+1)*k-1) ]  ).mean()
        df.iloc[(i*k ):((i+1)*k-1),  (j*k ):((j+1)*k-1) ] = media
        print("media(",i,",",j,") = " , media )
        print("limits = [",i*k,':',((i+1)*k-1) ,',', (j*k ),':',((j+1)*k-1), ']')
        # sostituzione di tutti i valori

# for i in range(8):
#     for j in range(8):
#         for k1 in range(8):
#             for k2 in range(8):
#                 df.iloc[(i*k + 1)+k1+1,  ((j)*k + 1)+k2+1 ] = media(i, j)


plt.axis('off')
plt.imshow( df, cmap=plt.cm.jet, interpolation='nearest')
plt.imshow( image[0], alpha=0.4)
plt.savefig('immagine_1/overlayed_heatmap_zones.png')
df.to_csv( 'immagine_1/roi_map_zones.csv', header=False, index = False, sep = ',')

# save the plot and the map
# if not os.path.exists('output'):
#     os.makedirs('output')
# 
