import os
import numpy as np
from typing import List
import copy
from collections import defaultdict
from tqdm.auto import tqdm

int_to_label: dict = {0: 'IPH', 1: 'IVH', 2: 'SDH', 3: 'EDH', 4: 'SAH'}
h_types: List[str] = list(int_to_label.values())

# Begin by reading the text files you stored during your model's inference on your calibration dataset.
calibration_scores: dict = defaultdict(dict)
"""
conformal_scores = {

}
"""