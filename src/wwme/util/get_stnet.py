import os.path

import pandas as pd
import geopandas as gpd
import argparse
import osmnx as ox

from shapely.geometry import Point, MultiPoint

# https://nominatim.openstreetmap.org/

cities = [['Manhattan',1],['Chicago',1],['Boston',1],['Brooklyn',1], ['Miami',1], ['Washington DC',1], ['Philadelphia',1]]

def download_shapefile():
    for c in cities:
        name = c[0]
        which_result = c[1]
        print("Downloading graph for %s..."%name)
        G = ox.graph_from_place(name, network_type='drive',retain_all=True,which_result=which_result)
        print("Done.\nGraph to gdf...")
        G = ox.add_edge_bearings(G)
        filename = name.lower().replace(' ','_')
        ox.config(use_cache=True, log_console=True)
        ox.save_graph_shapefile(G, f'./data/{filename}/{filename}_network_shp')
        print("Done.")


def download_osm():
    for c in cities:
        name = c[0]
        which_result = c[1]
        print("Downloading graph for %s..."%name)
        G = ox.graph_from_place(name, network_type='drive',retain_all=True,which_result=which_result)
        print("Done.\nGraph to gdf...")
        G = ox.add_edge_bearings(G)
        filename = name.lower().replace(' ','_')
        gdf = ox.graph_to_gdfs(G,nodes=False,edges=True,fill_edge_geometry=True)
        gdf.to_pickle(f'./data/{filename}/{filename}.pkl')
        print("Done.")

def get_points():
    for c in cities:
        print("Loading...")
        name = c[0]
        filename = name.lower().replace(' ','_')
        gdf = pd.read_pickle(f'./data/{filename}/{filename}.pkl')

        #     gdf.plot()
        print("Projecting to UTC...")
        gdf_utc = ox.projection.project_gdf(gdf, to_latlong=False)
        print("Saving csv for %s..."%name)
        # aux = gdf_utc.apply(lambda x: MultiPoint(ox.geo_utils.redistribute_vertices(x.geometry, 50))[1:-1], axis=1)

        aux = []
        for index,row in gdf_utc.iterrows():
            length = row['geometry'].length
            if length > 250:
                mp = MultiPoint([i for i in ox.utils_geo.interpolate_points(row['geometry'], length/5.0)])[1:-1]
                aux.append(mp)
            elif length > 200:
                mp = MultiPoint([i for i in ox.utils_geo.interpolate_points(row['geometry'], length/5.0)])[1:-1]
                aux.append(mp)
            elif length > 150:
                mp = MultiPoint([i for i in ox.utils_geo.interpolate_points(row['geometry'], length/5.0)])[1:-1]
                aux.append(mp)
            elif length > 100:
                mp = MultiPoint([i for i in ox.utils_geo.interpolate_points(row['geometry'], length/5.0)])[1:-1]
                aux.append(mp)
            elif length > 75:
                mp = MultiPoint([i for i in ox.utils_geo.interpolate_points(row['geometry'], length/5.0)])[1:-1]
                aux.append(mp)
            elif length > 50:
                mp = MultiPoint([i for i in ox.utils_geo.interpolate_points(row['geometry'], length/5.0)])[1:-1]
                aux.append(mp)
            else:
                mp = MultiPoint([i for i in ox.utils_geo.interpolate_points(row['geometry'], length/5.0)])[1:-1]
                aux.append(mp)



        gdf_utc = gdf_utc.set_geometry(aux)
        gdf_latlng = ox.projection.project_gdf(gdf_utc,to_latlong=True)

        out = open(f'./data/{filename}/{filename}.csv','w')

        for index, row in gdf_latlng.iterrows():
            points = [(pt.y, pt.x) for pt in row.geometry]
            bearing = row.bearing
            for i in range(0,len(points)):
                out.write('%f,%f,%f\n'%(points[i][0],points[i][1],bearing))
        out.close()
        print(filename)
        print("Done.\n")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get points.')
    parser.add_argument('--download', dest='download', action='store_true', help='Download OSM files.')
    parser.add_argument('--shp', dest='shp', action='store_true', help='Download OSM files as shapefile')

    args = parser.parse_args()
    if args.shp:
        download_shapefile()
    if args.download:
        download_osm()
    get_points()


