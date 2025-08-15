#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

from collections import namedtuple

DetectionPredTuple = namedtuple(
    typename="DetectionPredTuple",
    field_names=("labels", "scores", "boxes", "masks", "rois"),
    defaults=(None, None, None, None, None),
)
