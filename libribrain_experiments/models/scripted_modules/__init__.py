from .baseline import BaselineClassificationModule
from .baseline_bn import BaselineClassificationModule as BaselineBNClassificationModule
from .baseline_ln import BaselineClassificationModule as BaselineLNClassificationModule
from .megt import MegFormerClassificationModule

N_CLASSES = 39  # Number of output classes for classification (e.g., phonemes)

scripted_modules = {
    "baseline": BaselineClassificationModule,
    "baseline_bn": BaselineBNClassificationModule,
    "baseline_ln": BaselineLNClassificationModule,
    "megt": MegFormerClassificationModule,
}