import os
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

def prepare_gsv_data(src_path):
    gsv = pd.read_csv(src_path, sep=',', names=['status','id','lat','lng'], engine="python")
    gsv = gsv.dropna()
    gsv.drop('status', axis=1, inplace=True)
    gsv.reset_index(drop=True, inplace=True)
    return gsv

def prepare_segres(src_path):

    cols=['id', 'concrete', 'road', 'bckg', 'asphalt',
          'granite_block', 'hexagonal','bluestone_granite',
          'bricks', 'cobblestone', 'mixed']

    f = pd.read_csv(src_path, header=None, names=cols)

    cols_ord=['id', 'concrete','bricks', 'bluestone_granite',
              'asphalt', 'mixed', 'granite_block',
              'hexagonal', 'cobblestone', 'road', 'bckg']
    f=f.reindex(columns=cols_ord)

    direction = [f['id'][i].split('_')[-1] for i in range(len(f))]
    f.insert(1, 'direction', direction)
    f['id'] = [f['id'][i][:-6] if f.direction[i]=='right' else f['id'][i][:-5] for i in range(len(f))]

    f = f.fillna(0)
    return f

def prepare_matdata(df,err_thr=0.05):
    #remove images that are all background
    df.drop(df[df['bckg']==1].index, axis=0, inplace=True)
    #remove images that have no sidewalks or less than 3 percent sidewalks
    df.drop(df[(df.bckg + df.road) >= 0.97].index, axis=0, inplace=True)

    #what percentage of the detected sidewalk is which material
    #create a column with the sum of all freq belonging to sidewalks
    df['sum_sw']= df.iloc[:,2:-2].sum(axis=1)

    #the material percentage for each detected sidewalk
    df.iloc[:,2:-3] = df.iloc[:,2:-3].div(df.sum_sw, axis=0)
    #sum all materials that are below a threshold of the sidewalk since they are potentially error
    df['poten_error'] = df.iloc[:,2:-3][df.iloc[:,2:-3]<err_thr].sum(axis=1)
    #put all those less than threshold to zero, keep the sum in the poten_error
    df.iloc[:,2:-4] = df.iloc[:,2:-4].apply(lambda x: np.where(x < 0.05, 0, x))

    df.reset_index(drop=True, inplace=True)
    return df

def merge_materials(matdf,gsvdf):
    """
    Args:
        matdf: dataframe with predicted materials data per image id
        gsvdf: dataframe with the gsv lat lon (locations) and image id

    Returns:
        geodataframe with materials and location
        """
    merged = pd.merge(matdf, gsvdf, on=['id'], how='inner')
    merged.reset_index(inplace=True, drop=True)
    geoloc = gpd.GeoDataFrame(merged, geometry = [Point(x,y) for x,y in zip(merged.lng, merged.lat)], crs=4326)
    return geoloc



def spatial_join_stctl(merged_mat, city, max_distance=5):
    """
    it uses osmnx street network data, which should be downloaded and stored in "./data/city_name/"
    using ...

    Args:
        merged_mat (GeoDataFrame):
        city (str):
        max_distance (int):

    Returns:
        dataframe of material percentage aggregated on street centerline
    """

    shp_data = f'./data/{city}/{city}_network_shp/edges.shp'
    city_net = gpd.read_file(shp_data)
    #both datafarmes to utm
    merged_mat.geometry = merged_mat.geometry.to_crs(3857)
    city_net.geometry = city_net.geometry.to_crs(3857)
    city_net = city_net[~((city_net.highway =='motorway') | (city_net.highway =='motorway_link') | (city_net.highway =='trunk'))]
    city_net.reset_index(inplace=True)
    #match the lines to the nearest image
    spj = gpd.sjoin_nearest(merged_mat, city_net, max_distance=max_distance)
    mats = spj.groupby(by='index_right').agg({'concrete':'sum',
                                              'bricks':'sum',
                                              'bluestone_granite':'sum',
                                              'asphalt':'sum',
                                              'mixed':'sum',
                                              'granite_block':'sum',
                                              'hexagonal':'sum',
                                              'cobblestone':'sum',
                                              'bckg':'count'})
    mats.rename(columns={"bckg": "num_images"}, inplace=True)
    #sum of the sidewalk materials for each segment
    mats['sum_sw']= mats.iloc[:,0:-1].sum(axis=1)
    #the material percentage for each detected sidewalk
    mats.iloc[:,0:-2] = mats.iloc[:,0:-2].div(mats.sum_sw, axis=0)
    mats.reset_index(inplace=True)
    city_net.reset_index(inplace=True)
    final_network = city_net.merge(mats,right_on=['index_right'], left_on=['index'], how='inner')
    dest = f'./results/{city}/{city}_swmat_stcntl'
    if not os.path.exists(dest):
        os.makedirs(dest)

    final_network.to_file(os.path.join(dest,f'{city}_swmat_stcntl.shp'))
    # return final_network


def create_final_matnet(gsv,segres,city, matdata=None):
    """Create the sidewalk material data aggregated on street centerline.
    uses the osmnx road network for cities. The road data should be downloaded and
    prepared using the get_stnet module and the data should be in "./data/city_name/"

    Args:
        gsv (str): path to the file with gsv image id and lat lon location
        segres (str): path to the segmentation result file generated by the model
        with image id and material data
        city (str): name of the city
        matdata (str): Optional - the prepared segmentation results file using prepare_segres

    Returns:
        None - saves the shapefile of network with material data
    """
    gsvdf = prepare_gsv_data(gsv)
    if not matdata:
        _mat = prepare_segres(segres)
        matdf = prepare_matdata(_mat)
    else:
        matdf = prepare_matdata(matdata)
    merged = merge_materials(matdf,gsvdf)
    spatial_join_stctl(merged, city)


