from .baseline import BaselineClassificationModule
from .megt import MegFormerClassificationModule

N_CLASSES = 39  # Number of output classes for classification (e.g., phonemes)

scripted_modules = {
    "baseline": BaselineClassificationModule,
    "megt": MegFormerClassificationModule,
}