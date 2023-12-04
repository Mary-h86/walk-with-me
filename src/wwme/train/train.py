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
    from ..citysurfaces import CitySurfaces

from citysurfaces.train.args import Args


class Train(Args):
    def __call__(
            self,
            *args,
            **kwargs
    ) -> CitySurfaces:
        """

        :param args:
        :param kwargs:
        :return: A CitySurfaces object with a new model that may be saved or run again
        """
        result: CitySurfaces
        result.stage += 1
        return result
