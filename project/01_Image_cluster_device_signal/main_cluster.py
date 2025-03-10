from image_cluster import image_cluster
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import os

def main():
    # Initialize your image clustering instance
    img_cluster = image_cluster()

    # Step 1: Get list of image filenames
    img_cluster.file_name_list()
    
    # Step 2: Build the VGG16 model (without the top/output layer)
    model = img_cluster.model_build()
    
    # Step 3: Extract features from images and save them
    img_cluster.extract_feature(file, model)

    # Step 4: Standardize features
  
    # Step 5: autoencoder demention reduction
    
    # step 6: cluster with KMeans
    
    # step 7: visualize clusters



if __name__ == "__main__":
    main()
