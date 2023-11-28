from detectron2.config import LazyCall as L
from modeling.criterion.matting_criterion import DiffusionMattingCriterion

use_mse = True
use_mat = False

loss = L(DiffusionMattingCriterion)(
                use_mse = use_mse,
                use_mat = use_mat
            )