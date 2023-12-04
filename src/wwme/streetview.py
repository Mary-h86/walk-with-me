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


class StreetView:
    x = magic.Index()
    y = magic.Index()
    heading: Series[float]
