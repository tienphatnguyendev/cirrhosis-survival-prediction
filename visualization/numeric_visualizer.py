from .data_visualizer import DataVisualizer
import seaborn as sns
import matplotlib.pyplot as plt

colors = [
    "#191D88",
    "#1450A3",
    "#FFC436",
    "#337CCF",
]
sns.set_palette(sns.color_palette(colors))


class NumericVisualizer(DataVisualizer):
    def plot(
        self,
        kind,
        cols,
        hue=None,
        vert=False,
        bins=10,
        figsize=(20, 5),
        cols_per_row=3,
        box_width=0.5,
        vin_color=colors[2],
        box_color=colors[1],
    ):
        """
        Plots distribution plots for specified numeric columns of a DataFrame.

        Parameters:
        - cols (list): List of numeric column names to plot.
        """
        fig, axes = self.setup_plot(len(cols), cols_per_row, figsize)

        for i, col in enumerate(cols):
            if col in self.df.columns:
                if kind == "dist":
                    sns.histplot(self.df[col], kde=True, ax=axes[i], bins=bins, hue=hue)
                    axes[i].set_title(f"Distribution of {col}")
                elif kind == "box":
                    if not vert:
                        sns.violinplot(
                            data=self.df,
                            x=col,
                            ax=axes[i],
                            hue=hue,
                            color=vin_color,
                            edgecolor="black",
                            inner=None,
                            alpha=0.9,
                            linewidth=1.2,
                        )
                        sns.boxplot(
                            data=self.df,
                            x=col,
                            ax=axes[i],
                            hue=hue,
                            color=box_color,
                            width=box_width,
                            fill=False,
                            linewidth=2,
                            medianprops={"color": "red"},
                        )

                    else:
                        sns.boxplot(
                            data=self.df[col],
                            ax=axes[i],
                            hue=hue,
                            vert=vert,
                            color=box_color,
                        )
                    axes[i].set_title(f" of {col}")
            else:
                print(f"Column {col} not found in DataFrame.")

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.show()
