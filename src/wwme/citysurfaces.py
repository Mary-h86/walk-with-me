from __future__ import annotations
from geopandas import GeoDataFrame, GeoSeries
from pandas import IndexSlice as idx, Series, DataFrame, Index, MultiIndex, Categorical, CategoricalDtype
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy
import geopandas as gpd

import magicpandas as magic
import magicpandas.geo

from citysurfaces.labels import Labels
from citysurfaces.epochs import Epochs, Epoch
from citysurfaces.model import Model
from citysurfaces.truth import Truth
from citysurfaces.train.train import Train


class CitySurfaces(magic.geo.Root):
    x = magic.Index()
    y = magic.Index()
    labels = Labels()
    truth = Truth.from_kwargs(align='x y'.split())
    epochs = Epochs.from_kwargs(align='x y'.split())
    model = Model()
    train = Train()

    @classmethod
    def from_assets(
            cls,
            *args,
    ) -> CitySurfaces:
        ...

    @classmethod
    def to_assets(
            cls,
            *args,
            **kwargs,
    ) -> None:
        ...

    @magic.attr
    def stage(self) -> int:
        return 1


if __name__ == '__main__':
    cs = CitySurfaces.from_assets()
    epoch = cs.epochs[9]
    _ = epoch.prediction, epoch.uncertainty, epoch.miou, cs.labels
    cs.train.lr = .003
    stage2 = cs.train(...)
    stage2.to_assets(...)

    for n in range(10):
        cs = cs.train(...)
