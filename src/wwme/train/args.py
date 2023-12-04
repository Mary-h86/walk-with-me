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


class Args(magic.Commandline):
    @magic.arg
    @property
    def lr(self) -> float:
        return .002

    @magic.arg
    @property
    def arch(self) -> str:
        """Network architecture."""
        return 'ocrnet.HRNet_Mscale'

    @magic.arg
    @property
    def old_data(self) -> bool:
        """sets the dataset to the first one in hrnet"""
        return False

    @magic.arg
    @property
    def dataset(self) -> str:
        """Dataset name."""
        return 'cityscapes'

    @magic.arg
    @property
    def num_workers(self) -> int:
        """CPU worker threads per dataloader instance."""
        return 4

    @magic.arg
    @property
    def cv(self) -> int:
        """Cross-validation split id to use."""
        return 0

    @magic.arg
    @property
    def class_uniform_pct(self) -> float:
        """Fraction of images uniformly sampled."""
        return 0.5

    @magic.arg
    @property
    def class_uniform_tile(self) -> int:
        """Tile size for class uniform sampling."""
        return 512

    @magic.arg
    @property
    def coarse_boost_classes(self) -> str:
        """Use coarse annotations for specific classes."""
        return None

    @magic.arg
    @property
    def custom_coarse_dropout_classes(self) -> str:
        """Drop some classes from auto-labelling."""
        return None

    @magic.arg
    @property
    def img_wt_loss(self) -> bool:
        """Per-image class-weighted loss."""
        return True

    @magic.arg
    @property
    def rmi_loss(self) -> bool:
        """Use RMI loss."""
        return False

    @magic.arg
    @property
    def batch_weighting(self) -> bool:
        """Batch weighting for class."""
        return True

    @magic.arg
    @property
    def jointwtborder(self) -> bool:
        """Enable boundary label relaxation."""
        return False

    @magic.arg
    @property
    def strict_bdr_cls(self) -> str:
        """Enable boundary label relaxation for specific classes."""
        return ''

    @magic.arg
    @property
    def rlx_off_epoch(self) -> int:
        """Turn off border relaxation after a specific epoch count."""
        return 80

    @magic.arg
    @property
    def rescale(self) -> float:
        """Warm Restarts new lr ratio compared to original lr."""
        return 1.0

    @magic.arg
    @property
    def repoly(self) -> float:
        """Warm Restart new poly exponent."""
        return 1.5

    @magic.arg
    @property
    def apex(self) -> bool:
        """Use Nvidia Apex Distributed Data Parallel."""
        return True

    @magic.arg
    @property
    def fp16(self) -> bool:
        """Use Nvidia Apex AMP."""
        return True

    @magic.arg
    @property
    def local_rank(self) -> int:
        """Parameter used by apex library."""
        return 0

    @magic.arg
    @property
    def global_rank(self) -> int:
        """Parameter used by apex library."""
        return 0

    @magic.arg
    @property
    def optimizer(self) -> str:
        """Optimizer."""
        return 'sgd'

    @magic.arg
    @property
    def amsgrad(self) -> bool:
        """Amsgrad for adam."""
        return False

    @magic.arg
    @property
    def freeze_trunk(self) -> bool:
        """Freeze trunk model."""
        return False

    @magic.arg
    @property
    def hardnm(self) -> int:
        """Hard negative mining iterations."""
        return 0

    @magic.arg
    @property
    def trunk(self) -> str:
        """Trunk model."""
        return 'hrnetv2'

    @magic.arg
    @property
    def max_epoch(self) -> int:
        """Maximum number of training epochs."""
        return 180

    @magic.arg
    @property
    def max_cu_epoch(self) -> int:
        """Class Uniform Max Epochs."""
        return 150

    @magic.arg
    @property
    def start_epoch(self) -> int:
        """Starting epoch for training."""
        return 0

    @magic.arg
    @property
    def color_aug(self) -> float:
        """Level of color augmentation."""
        return 0.25

    @magic.arg
    @property
    def gblur(self) -> bool:
        """Use Gaussian Blur Augmentation."""
        return False

    @magic.arg
    @property
    def bblur(self) -> bool:
        """Use Bilateral Blur Augmentation."""
        return True

    @magic.arg
    @property
    def brt_aug(self) -> bool:
        """Use brightness augmentation."""
        return False

    @magic.arg
    @property
    def lr_schedule(self) -> str:
        """Name of lr schedule."""
        return 'poly'

    @magic.arg
    @property
    def poly_exp(self) -> float:
        """Polynomial LR exponent."""
        return 2.0

    @magic.arg
    @property
    def poly_step(self) -> int:
        """Polynomial epoch step."""
        return 110

    @magic.arg
    @property
    def bs_trn(self) -> int:
        """Batch size for training per GPU."""
        return 2

    @magic.arg
    @property
    def bs_val(self) -> int:
        """Batch size for validation per GPU."""
        return 1

    @magic.arg
    @property
    def crop_size(self) -> str:
        """Training crop size."""
        return '640,640'

    @magic.arg
    @property
    def scale_min(self) -> float:
        """Minimum dynamic scale for training images."""
        return 0.5

    @magic.arg
    @property
    def scale_max(self) -> float:
        """Maximum dynamic scale for training images."""
        return 2.0

    @magic.arg
    @property
    def weight_decay(self) -> float:
        """Weight decay."""
        return 1e-4

    @magic.arg
    @property
    def momentum(self) -> float:
        """Momentum."""
        return 0.9

    @magic.arg
    @property
    def snapshot(self) -> str:
        """Snapshot."""
        return None

    @magic.arg
    @property
    def resume(self) -> str:
        """Continue training from a checkpoint."""
        return None

    @magic.arg
    @property
    def restore_optimizer(self) -> bool:
        """Restore optimizer from a checkpoint."""
        return False

    @magic.arg
    @property
    def restore_net(self) -> bool:
        """Restore network from a checkpoint."""
        return False

    @magic.arg
    @property
    def exp(self) -> str:
        """Experiment directory name."""
        return 'default'

    @magic.arg
    @property
    def result_dir(self) -> str:
        """Directory for log output."""
        return None

    @magic.arg
    @property
    def syncbn(self) -> bool:
        """Use Synchronized BN."""
        return False

    @magic.arg
    @property
    def dump_augmentation_images(self) -> bool:
        """Dump Augmented Images for sanity check."""
        return False

    @magic.arg
    @property
    def test_mode(self) -> bool:
        """Minimum testing to verify functionality."""
        return False

    @magic.arg
    @property
    def wt_bound(self) -> float:
        """Weight Scaling for the losses."""
        return 1.0

    @magic.arg
    @property
    def maxSkip(self) -> int:
        """Skip frames of video augmented dataset."""
        return 0

    @magic.arg
    @property
    def scf(self) -> bool:
        """Scale correction factor."""
        return False

    @magic.arg
    @property
    def full_crop_training(self) -> bool:
        """Full Crop Training."""
        return False

    @magic.arg
    @property
    def multi_scale_inference(self) -> bool:
        """Run multi scale inference."""
        return False

    @magic.arg
    @property
    def default_scale(self) -> float:
        """Default scale to run validation."""
        return 1.0

    @magic.arg
    @property
    def log_msinf_to_tb(self) -> bool:
        """Log multi-scale Inference to Tensorboard."""
        return False

    @magic.arg
    @property
    def eval(self) -> str:
        """Just run evaluation."""
        return None

    @magic.arg
    @property
    def eval_folder(self) -> str:
        """Path to frames to evaluate."""
        return None

    @magic.arg
    @property
    def three_scale(self) -> bool:
        """Use three scales for inference."""
        return False

    @magic.arg
    @property
    def alt_two_scale(self) -> bool:
        """Use alternative two scales for inference."""
        return False

    @magic.arg
    @property
    def do_flip(self) -> bool:
        """Use flip augmentation."""
        return False

    @magic.arg
    @property
    def extra_scales(self) -> str:
        """Extra scales for inference."""
        return '0.5,1.5,2.0'

    @magic.arg
    @property
    def n_scales(self) -> str:
        """Number of scales for inference."""
        return None

    @magic.arg
    @property
    def align_corners(self) -> bool:
        """Align corners in spatial operations."""
        return False

    @magic.arg
    @property
    def translate_aug_fix(self) -> bool:
        """Fix for translation augmentation."""
        return False

    @magic.arg
    @property
    def mscale_lo_scale(self) -> float:
        """Low resolution training scale."""
        return 0.5

    @magic.arg
    @property
    def pre_size(self) -> int:
        """Resize long edge of images before augmentation."""
        return None

    @magic.arg
    @property
    def amp_opt_level(self) -> str:
        """AMP optimization level."""
        return 'O1'

    @magic.arg
    @property
    def rand_augment(self) -> str:
        """RandAugment setting."""
        return None

    @magic.arg
    @property
    def init_decoder(self) -> bool:
        """Initialize decoder with kaiming normal."""
        return False

    @magic.arg
    @property
    def dump_topn(self) -> int:
        """Dump worst validation images."""
        return 50

    @magic.arg
    @property
    def dump_assets(self) -> bool:
        """Dump interesting assets."""
        return False

    @magic.arg
    @property
    def dump_all_images(self) -> bool:
        """Dump all images, not just a subset."""
        return False

    @magic.arg
    @property
    def dump_for_submission(self) -> bool:
        """Dump assets for submission."""
        return False

    @magic.arg
    @property
    def dump_for_auto_labelling(self) -> bool:
        """Dump assets for auto-labelling."""
        return False

    @magic.arg
    @property
    def dump_topn_all(self) -> bool:
        """Dump top N worst failures."""
        return True

    @magic.arg
    @property
    def custom_coarse_prob(self) -> float:
        """Custom Coarse Probability."""
        return None

    @magic.arg
    @property
    def only_coarse(self) -> bool:
        """Use only coarse annotations."""
        return False

    @magic.arg
    @property
    def mask_out_cityscapes(self) -> bool:
        """Mask out cityscapes data."""
        return False

    @magic.arg
    @property
    def ocr_aspp(self) -> bool:
        """Use OCR ASPP."""
