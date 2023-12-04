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


class Prediction(Matrix):
    root: CitySurfaces
    owner: Epoch

    @magic.column
    def path(self) -> Series[str]:
        for name in self.root.truth.filename:
            ...


class Uncertainty(Matrix):
    root: CitySurfaces
    owner: Epoch

    @magic.column
    def path(self) -> Series[str]:
        for name in self.root.truth.filename:
            ...


class Epoch(magic.Frame):
    x = magic.Index()
    y = magic.Index()
    prediction = Prediction.from_kwargs(align=True)
    uncertainty = Uncertainty.from_kwargs(align=True)
    owner: CitySurfaces
    root: CitySurfaces  # todo assign

    @magic.column
    def miou(self) -> Series[float]:
        ...


# todo: allow epochs to be used as a dict, and also align with root
class Epochs(collections.UserDict, magic.Frame):
    def __init__(self):
        super().__init__()
        # noinspection PyTypeChecker
        self.data: dict[CitySurfaces, list[Epoch]] = weakref.WeakKeyDictionary()

    def __get__(self, instance, owner) -> Epochs:
        self.owner = instance
        self.Owner = owner
        # return self.data
        return self

    def __getitem__(self, item) -> Epoch:
        return super().__getitem__(item)

"""
prediction.stitch[::3]
"""