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
    from .wwme import WalkWithMe


class Matrix(magic.Frame):
    owner: WalkWithMe
    x = magic.Index()
    y = magic.Index()

    # todo: automated caching/loader to improve IO speed (like the 1200% speedup in deep-umbra)

    @magic.column
    def path(self) -> Series[str]:
        """Path to the file containing the matrix"""

    def overlay(self, *args, **kwargs):
        """Overlay onto the ground truth images"""
