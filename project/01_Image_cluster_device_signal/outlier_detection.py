# load raw file
# standard scale
# DBscan
## calculate min_pts
## calculate ops
## visulaize with PCA

def outlier_detection():
    path = 
    raw_df = 
    
    
    scaler = StandardScaler()
    scaler.fit(raw_df)
    df_scaled = scaler.transform(raw_df)
    
    # DBscan
    min_pts = raw_df.shape[1]*2
    
    neighbors = NearestNeighbors(n_neighbors = min_pts, metric = 'euclidean')
    neighbors_fit = neighbors.fit(df_scaled)
    distances, indices = neighbors_fit.kneighbors(df_scaled)
    
    # eps = 1.5
    distances = np.sort(distances, axis = 0)
    distances - distances[:,1]
    
    plt.plot(distances)

    # change x axis range to 3000 - 3500
    plt.xlim([75000,80000])
    plt.ylim([0,5])
     
    m = DBSCAN(eps=1.5, min_samples=min_pts)
    m.fit(df_scaled)

    clusters = m.fit_predict(df_scaled)
    
    # PCA reduction
    pca = PCA()
    pipeline = make_pipeline(scaler,pca)
    pipeline.fit(df_scaled)
    
    #pca.n_components_(차원축소 주성분 개수)

    features = range(pca.n_components_)
    feature_df=pd.DataFrame(data = features, columns=['pc_feature'])

    # pca.explained_variance_ratio (설명력)
    variance_df=pd.DataFrame(data = pca.explained_variance_ratio_,columns=['variance'])
    pc_feature_df = pd.concat([feature_df,variance_df],axis=1)
    
    pca = PCA(n_components = 3)
    principal_comp = pca.fit_transform(df_scaled)
    principal_comp
    
    pca_df = pd.DataFrame(data = principal_comp, columns = ['pca1', 'pca2','pca3'])
    pca_df = pd.concat([pca_df, pd.DataFrame({'cluster': clusters})], axis = 1)
    
    # Number of outliers
    pca_df.cluster.value_counts()