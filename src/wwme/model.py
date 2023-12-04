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
    from .wwme import WalkWithMe


class Model(magic.Frame):
    ...

