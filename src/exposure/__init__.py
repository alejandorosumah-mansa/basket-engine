"""Side detection and exposure normalization for prediction market baskets."""

from .side_detection import detect_phrasing_polarity, detect_side_batch
from .normalization import normalize_exposures, ExposureInfo
from .basket_rules import check_exposure_conflicts, filter_opposing_exposures
from .report import generate_exposure_report
