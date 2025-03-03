# for loading/processing the images  
from tensorflow.keras.preprocessing.image import load_img 
from tensorflow.keras.preprocessing.image import img_to_array 
from tensorflow.keras.applications.vgg16 import preprocess_input 

# models 
from tensorflow.keras.applications.vgg16 import VGG16 
from tensorflow.keras.models import Model

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# for everything else
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle

class image_cluster:
    def __init__(self):
        self.path = r"/home2/hyjung/works/analyses_code/alpha4/base_sn_malfunction_suspiscious/FAM_baseline_suspicious_1"
        self.curves = []
    def file_name_list():
        with os.scandir(self.path) as files:
        # loops through each file in the directory
            for file in files:
                if file.name.endswith('.png'):
                # adds only the image files to the flowers list
                    self.curves.append(file.name)
    
    def model_build():
        # removing output layer manually: the new final layer is a fully-connected layer with 4,096 output nodes
        # This vector of 4,096 numbers is the feature vector that we will use to cluster the images
        model = Model(inputs = model.inputs, outputs = model.layers[-2].output)              
    def extract_feature(self, file, model):
        # load the image as a 224x224 array
        img = load_img(file, target_size=(224,224))
        # convert from 'PIL.Image.Image' to numpy array
        img = np.array(img) 
        # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
        reshaped_img = img.reshape(1,224,224,3) 
        # prepare image for model
        imgx = preprocess_input(reshaped_img)
        # get the feature vector
        features = model.predict(imgx, use_multiprocessing=True)
        return features
    def util_extract_features():
        data = {}

        # loop through each image in the dataset
        for curve in self.curves:
            # try to extract the features and update the dictionary
            try:
                feat = extract_features(curve,model)
                data[curve] = feat
            # if something fails, save the extracted features as a pickle file (optional)
            except:
                with open(self.path,'wb') as file:
                    pickle.dump(data,file)
    def standardize_features():
        # get a list of the filenames
        filenames = np.array(list(data.keys()))

        # get a list of just the features
        feat = np.array(list(data.values()))
        
        # reshape so that there are __ samples of 4096 vectors
        feat = feat.reshape(-1,4096)
        scaler = StandardScaler()
        feat_sc = scaler.fit_transform(feat)
        
    def autoencoder():
        # autoencoder 로 차원축소 후 K-means로 cluster

        #https://cypision.github.io/deep-learning/clustering_with_autoENC/


        ## input_layer
        input_layer = tf.keras.Input( shape = (4096, ))

        ## Dense_layer first Encoder 

        encoder_layer_1 = Dense(2000, activation = 'relu')(input_layer)

        ## Dense_layer first Encoder 

        encoder_layer_2 = Dense(200, activation = 'relu')(encoder_layer_1)

        ## Dense_layer Second Encoder <-- Bottleneck

        encoder_layer_3 = Dense(100, activation = 'relu')(encoder_layer_2)

        ## Dense_layer First Decoder
        decoder_layer_1 = Dense(200, activation = 'relu', kernel_initializer='glorot_uniform')(encoder_layer_3)

        ## Dense_layer First Decoder
        decoder_layer_2 = Dense(2000, activation = 'relu', kernel_initializer='glorot_uniform')(decoder_layer_1)

        ## Dense_layer Second Decoder
        decoder_layer_3 = Dense(4096, activation = 'relu', kernel_initializer='glorot_uniform')(decoder_layer_2)


        autoencoder = tf.keras.Model(input_layer, decoder_layer_3)

        autoencoder.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
        autoencoder.fit(feat_sc, feat_sc, batch_size= 120, epochs = 100, verbose = 1)
    
        # create encoder model

        encoder = tf.keras.Model(input_layer, encoder_layer_3)  # encoder_layer_3 는 bottleneck
        
        # get latent vector for visualization
        latent_vector = encoder.predict(feat_sc)

    def cluster():
        # choose number of clusters K:
        Sum_of_squared_distances = []
        K = range(1,20)
        for k in K:
            km = KMeans(init='k-means++', n_clusters=k, n_init=10)
            km.fit(feat_sc)
            Sum_of_squared_distances.append(km.inertia_)

        plt.plot(K, Sum_of_squared_distances, 'bx-')
        #plt.vlines(ymin=0, ymax=15, x=8, colors='red')
        #plt.text(x=8.2, y=130000, s="optimal K=8")
        plt.xlabel('Number of Clusters K')
        plt.ylabel('Sum of squared distances')
        plt.title('Elbow Method For Optimal K')
        plt.show()
        # cluster feature vectors
        kmeans = KMeans(n_clusters=6, random_state=22)
        kmeans.fit(feat_sc)

        # holds the cluster id and the images { id: [images] }
        groups = {}
        for file, cluster in zip(filenames,kmeans.labels_):
            if cluster not in groups.keys():
                groups[cluster] = []
                groups[cluster].append(file)
            else:
                groups[cluster].append(file)  
    def view_cluster(cluster):

        cols = 5
        rows = int(len(groups[cluster])/cols) + int(len(groups[cluster])%cols != 0) # row 수는 col 수로 나눈것 + 나머지가 있으면 1

        fig, ax = plt.subplots(rows,cols,figsize=(50,rows*5))

        for row in range(rows):
            for col in range(cols):
                try:
                    ax[row,col].imshow(np.array(load_img(groups[cluster][row*cols+col])))
                    ax[row,col].axis('off')
                except: # files 안의 index 를 넘어가게 되면 에러가 나기 때문에 중단.
                    row*cols+col>len(groups[cluster])            
          