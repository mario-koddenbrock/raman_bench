"""
Preprocessing pipeline implementation using RamanSPy.
"""

import ramanspy as rp


def get_preprocessing_pipeline():

    # TODO
    steps = [
        rp.preprocessing.baseline.ASLS(),
        rp.preprocessing.normalise.MinMax(),
        rp.preprocessing.denoise.SavGol(window_length=9, polyorder=3),
    ]

    return steps





