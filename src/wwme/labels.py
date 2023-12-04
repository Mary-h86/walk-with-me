from __future__ import annotations
import pandas as pd
from io import StringIO

import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from pandas import IndexSlice as idx, Series, DataFrame, Index, MultiIndex, Categorical, CategoricalDtype
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy
import geopandas as gpd

import magicpandas as magic

if False:
    from .citysurfaces import CitySurfaces


class Labels(magic.Frame):
    ilabel = magic.Index()
    name: Series[str]
    trainid: Series[int]
    r: Series[int]
    g: Series[int]
    b: Series[int]

    def from_nested(self, owner: CitySurfaces) -> Labels:
        # name id ilabel color r g b mapping,
        csv = """
        concrete,1,0,255,127,14
        bricks,2,1,43,160,43
        granite,3,2,31,119,179
        asphalt,4,3,153,153,153
        mixed,5,4,214,39,40
        road,6,5,54,54,54
        background,7,6,0,0,0
        granite block-stone,8,7,138,0,138
        hexagonal,9,8,240,110,170
        cobblestone,10,9,139,109,48
        """
        columns = 'name id ilabel r g b'.split()
        # todo: map to sidewalk, road, background
        csv = StringIO(csv)
        result = (
            pd.read_csv(csv, header=None, names=columns)
            .set_index('ilabel')
            .pipe(self.brand)
        )
        return result

    @magic.column
    def color(self):
        raise NotImplementedError


