import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod


class DataVisualizer:
    def __init__(self, df):
        self.df = df

    def setup_plot(self, num_plots, cols_per_row, figsize):
        num_rows = (num_plots + cols_per_row - 1) // cols_per_row
        fig, axes = plt.subplots(num_rows, cols_per_row, figsize=figsize, squeeze=False)
        return fig, axes.flatten()

    @abstractmethod
    def plot():
        pass
