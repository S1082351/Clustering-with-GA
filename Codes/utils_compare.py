import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils_parser import *


def calculate_jaccard(real_labels, computed_labels):
    a = 0
    b = 0
    c = 0
    for i, label in enumerate(real_labels):
        for j, label2 in enumerate(real_labels):
            if j > i:
                if label == label2:
                    if computed_labels[i] == computed_labels[j]:
                        a = a + 1
                    else:
                        c = c + 1
                else:
                    if computed_labels[i] == computed_labels[j]:
                        b = b + 1              

    jaccard = a / (a + b + c)
    return jaccard