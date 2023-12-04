from __future__ import annotations

import collections
import weakref

import pandas as pd
from io import StringIO

import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from pandas import IndexSlice as idx, Series, DataFrame, Index, MultiIndex, Categorical, CategoricalDtype
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy
import geopandas as gpd

from citysurfaces.matrix import Matrix
import magicpandas as magic

if False:
    from .citysurfaces import CitySurfaces


class Truth(Matrix):
    filename = magic.Index()
    x: Series[int]
    y: Series[int]
    gx: Series[float]
    gy: Series[float]
    px: Series[float]
    py: Series[float]
    heading: Series[float]
    pitch: Series[float]
    root: CitySurfaces
    owner: CitySurfaces

    @magic.column
    def path(self) -> Series[str]:
        ...
