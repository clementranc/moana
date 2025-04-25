import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator, AutoMinorLocator
import moana
import numpy as np
import pandas as pd
import scipy
from scipy.interpolate import interp1d
from scipy.optimize import minimize


class SampledPosterior:
    """Class to derive statistical properties from a sample.

    Args:
        sample: sample for 1+ parameters.
        labels: labels of columns to include in statistics.
        weights: this str indicates which column is used to weight
            the samples.

    Attributes:
        sample: :obj:`pandas.DataFrame` of the parameters.
        cdf: dict and for each label, cdf[label] is a :obj:`numpy.array` with
            sorted values in cdf[label][0] and the cumulative function in
            cdf[label][1].
    """

    def __init__(
        self, sample: pd.DataFrame, labels: list, limit=None, weights: str = None
    ):
        self.sample = sample
        self.request = labels
        self.weights = weights

        if limit == None:
            self.limit = [0.1585, 0.5, 0.8415]  # Default limits
        else:
            self.limit = limit

        self.cdf = dict()
        self.ci = dict()

        for l in self.request:
            try:
                self.cdf.update({l: self.build_cdf(l, weights=weights)})
                self.ci.update({l: self.find_limits(l, self.limit)})
            except:
                print(f"Cannot determine credible intervals for {l}")
                print("Values cannot be NaN, they must be real.")

    def build_cdf(self, label: str, weights: str = None) -> np.array:
        """Build the cumulative distribution function.

        Args:
            label: this str indicates which column is used.
            weights: this str indicates which column is used to weight
                the samples.

        Return:
            Array of the sorted values (column 0) and the cumulative function
                (column 1).
        """

        if not weights == None:
            table = self.sample.sort_values(label)
            w = table[weights].values
            return np.array([table[label].values, np.cumsum(w) / np.sum(w)])
        else:
            return np.array(
                [
                    self.sample.sort_values(label)[label].values,
                    (np.arange(len(self.sample)) + 1) / len(self.sample),
                ]
            )

    def find_limits(
        self,
        label: str,
        limit: list,
    ) -> np.array:
        """Determine the parameter values based on a given probability.

        Args:
            label: this str indicates which column is used.
            limit: list of probability values.
            weights: this str indicates which column is used to weight
                the samples.

        Return:
            Array of the corresponding values of the parameter.
        """
        ci = np.array([])
        cdf = self.cdf[label]

        for l in limit:
            fintp = interp1d(cdf[0], cdf[1] - l, kind="linear")
            ci = np.append(
                ci, scipy.optimize.brentq(fintp, np.min(cdf[0]), np.max(cdf[0]))
            )
        return ci

    def get2dcontours(self, cdf_levels=None, weights=None, **bins_opts):
        labels = self.request
        samples = self.sample
        N = len(labels)

        bins = dict()
        [bins.update({a + b: [50, 50]}) for a in labels for b in labels]
        [bins.update({b + a: [50, 50]}) for a in labels for b in labels]
        bins.update(bins_opts)

        xhist = [[None] * N for a in [None] * N]
        yhist = [[None] * N for a in [None] * N]
        hist = [[None] * N for a in [None] * N]
        hist_levels = [[None] * N for a in [None] * N]

        for i in range(N):
            for j in range(N):
                if j < i:
                    x = samples[labels[j]].values
                    y = samples[labels[i]].values
                    a, b, c, d = moana.corner.compute_2dcontours(
                        x,
                        y,
                        bins=[
                            bins[labels[j] + labels[i]][0],
                            bins[labels[j] + labels[i]][1],
                        ],
                        cdf_levels=cdf_levels,
                        weights=weights,
                    )
                    xhist[i][j] = a
                    yhist[i][j] = b
                    hist[i][j] = c
                    hist_levels[i][j] = d

        return xhist, yhist, hist, hist_levels

    def _get_plot_config_scatter_plots(self, rcfile=None, rotation=0, rcparams=dict()):
        """Default plot configuration for scatter plots."""
        try:
            if not rcfile == None:
                plt.style.use(rcfile)
            else:
                path = "/".join(moana.__file__.split("/")[:-1])
                plt.style.use(f"{path}/stylelib/corner_plots_times.mplstyle")
        except:
            pass

        if rotation > 10:
            mpl.rcParams["xtick.major.pad"] = 2  # Useful when ticks are rotated

        try:
            mpl.rcParams.update(rcparams)
        except:
            pass

        plt.close()
        plt.clf()

    def scatter_plot_scaling(
        self, width: float = 2 * 84 / 25.4, optimize: bool = True
    ) -> list:
        """Compute size of scatter plots and margins

        Args:
            width: Size of the plot.
            optimize: If True, the function will reduce each subplot so that the
                total plot size is the value of width. If False, then the margins
                will enlarge a little bit the plot, and the total size will be
                slightly larger than the given width parameter.

        Return:
            List of float with with:
                - the total size of the plot;
                - the bottom and left margins;
                - the top and right margins;
                - the interplot spacing.
        """
        N = len(self.request)

        if optimize:
            width = self._get_scatter_plots_size(target=width)

        wh = 0.05  # w/hspace size
        plot_size = width + width * (N - 1) * wh / N
        width_new = 0.8 * width / N + plot_size + 0.8 * width / N  # inches
        bl = 0.8 * width / (N * width_new)
        tr = (0.8 * width / N + plot_size) / width_new
        return (width_new, bl, tr, wh)

    def _plotsize(self, width: float, target: float) -> float:
        """Define a metric for actual plot size, and the target.

        Args:
            width: initial condition in inches.
            target: target width in inches.

        Return:
            Quadratic distance between target and actual plot size.
        """
        width_new, _, _, _ = self.scatter_plot_scaling(width, optimize=False)
        return np.power(width_new - target, 2)

    def _get_scatter_plots_size(self, target: float) -> float:
        """Find the optimal plot size so that plot+margins are matching the target.

        Args:
            target: target width in inches.

        Return:
            Optimal plot size in inches.
        """
        bnds = [[1e-2, target]]
        res = minimize(self._plotsize, [target * 0.9], args=(target), bounds=bnds)
        try:
            final = moana.dbc.custom_floor(res.x[0], precision=3)
        except IndexError:
            final = target
        print(
            f"Each sub-plot must be {final:.3f}in x {final:.3f}in to have a figure of {target:.3f}in. Okay! I'm using it!"
        )
        return final

    def corner_plot(
        self,
        axes_options=dict(),
        contourf_options=dict(),
        figure=None,
        axes=None,
        bins_options=None,
        credible_intervals=True,
        fill_ci=True,
        show_samples=False,
        diagonal="cumul",
        display_plot_coords=False,
        labels=dict(),
        bins=dict(),
        filename=None,
        saving_options=dict(),
        align_xlabels=True,
        align_ylabels=True,
        cdf_levels=None,
        weights=True,
        rotation=0,
        display_1sigma=False,
        rcfile=None,
        rcparams=dict(),
        bins_density=dict(),
    ):
        N = len(self.request)

        # Definition of the figure and axes
        if isinstance(figure, mpl.figure.Figure) | isinstance(axes, np.ndarray):
            fig = figure
            ax = axes
        else:
            self._get_plot_config_scatter_plots(
                rcfile=rcfile, rotation=rotation, rcparams=rcparams
            )
            width, lb, tr, wh_margin = self.scatter_plot_scaling()
            fig, ax = plt.subplots(N, N, figsize=[width, width])
            fig.subplots_adjust(
                left=lb, bottom=lb, right=tr, top=tr, wspace=wh_margin, hspace=wh_margin
            )

        # Compute credible intervals
        if weights & isinstance(self.weights, str):
            xhist, yhist, hist, hist_levels = self.get2dcontours(
                cdf_levels=cdf_levels, weights=self.sample[self.weights].values, **bins
            )
        else:
            xhist, yhist, hist, hist_levels = self.get2dcontours(
                cdf_levels=cdf_levels, weights=None, **bins
            )

        # How label are displayed
        label_names = dict()
        [label_names.update({a: a}) for a in self.request]
        label_names.update(labels)

        labels = self.request

        # Range, major and minor ticks locators
        ax_options = dict()
        [ax_options.update({a: 4 * [None]}) for a in self.request]
        ax_options.update(axes_options)

        spectral_map = plt.get_cmap("Spectral")
        alpha = [0.5, 0.9, 0.9, 0.9, 0.9]
        s_list = [1, 5, 5, 5, 5]
        dchi2_list = np.array([16, 9, 4, 1, 0])
        colors = [
            spectral_map(255.0 / 255.0),
            spectral_map(170.0 / 255.0),
            spectral_map(85.0 / 255.0),
            spectral_map(0.0),
        ]

        samples = self.sample.copy(deep=True)
        samples["reject"] = 0

        if cdf_levels == None:
            contourf_opts = {
                "colors": [
                    (0.72, 0.72, 0.9003921568627451),
                    (0.36, 0.36, 0.7011764705882353),
                    (0.0, 0.0, 0.5019607843137255),
                ],
                "antialiased": False,
            }  # with blue shades
            contourf_opts.update(contourf_options)
        else:
            if len(cdf_levels) <= 3:
                contourf_opts = {
                    "colors": [
                        (0.72, 0.72, 0.9003921568627451),
                        (0.36, 0.36, 0.7011764705882353),
                        (0.0, 0.0, 0.5019607843137255),
                    ],
                    "antialiased": False,
                }  # with blue shades
                contourf_opts.update(contourf_options)
            else:
                contourf_opts = dict(contourf_options)

        # Create subplots
        for i in range(N):
            for j in range(N):
                if j > i:
                    ax[i][j].set_xlabel("")
                    ax[i][j].set_ylabel("")
                    ax[i][j].axes.get_xaxis().set_visible(False)
                    ax[i][j].axes.get_yaxis().set_visible(False)
                    ax[i][j].set_frame_on(False)
                else:
                    ax[i][j].xaxis.set_minor_locator(AutoMinorLocator(2))
                    ax[i][j].yaxis.set_minor_locator(AutoMinorLocator(2))
                    ax[i][j].set_facecolor("none")
                    if i == N - 1:
                        ax[i][j].set_xlabel(rf"${label_names[labels[j]]}$", labelpad=0)
                        ax[i][j].tick_params(axis="x", which="major", pad=2)
                        for tick in ax[i][j].get_xticklabels():
                            tick.set_rotation(rotation)
                            if rotation > 10:
                                tick.set_ha("right")  # Pour la rotation de 30 deg
                    else:
                        ax[i][j].set_xlabel("")
                        ax[i][j].set_xticklabels([])
                        ax[i][j].tick_params(labelbottom=False)
                        if j != N - 2:
                            ax[i][j].sharex(ax[i + 1][j])

                    if j == 0:
                        if i == 0:
                            ax[i][j].set_ylabel(rf"${label_names[labels[i]]}$")
                        else:
                            ax[i][j].set_ylabel(
                                rf"${label_names[labels[i]]}$", labelpad=0
                            )
                    else:
                        ax[i][j].set_ylabel("")
                        if not i == j:
                            ax[i][j].set_yticklabels([])
                    if (j <= i) and display_plot_coords:
                        ax[i][j].annotate(
                            f"({i}, {j})",
                            (0.05, 0.05),
                            xycoords="axes fraction",
                            c="k",
                            size=11,
                            weight=500,
                            ha="left",
                            va="bottom",
                            bbox=dict(boxstyle="square, pad=0", fc="None", ec="None"),
                        )
                    if not i == j:
                        if credible_intervals:
                            levels = np.concatenate(
                                [hist_levels[i][j], [hist[i][j].max() * (1 + 1e-6)]]
                            )
                            if fill_ci:
                                ax[i][j].contourf(
                                    xhist[i][j],
                                    yhist[i][j],
                                    hist[i][j].T,
                                    levels,
                                    **contourf_opts,
                                )
                            else:
                                ax[i][j].contour(
                                    xhist[i][j],
                                    yhist[i][j],
                                    hist[i][j].T,
                                    levels,
                                    **contourf_opts,
                                )
                        if show_samples:
                            for id_dchi2 in range(len(dchi2_list) - 1):
                                cond = samples.reject == 1
                                samples.loc[cond, "reject"] = 0
                                samples.loc[
                                    (
                                        (samples.dchi2 < dchi2_list[id_dchi2 + 1])
                                        | (samples.dchi2 >= dchi2_list[id_dchi2])
                                    ),
                                    "reject",
                                ] = 1
                                cond = samples.reject == 0
                                ax[i][j].scatter(
                                    samples[cond][labels[j]].values,
                                    samples[cond][labels[i]].values,
                                    s=s_list[id_dchi2],
                                    facecolors=colors[id_dchi2],
                                    marker="o",
                                    alpha=alpha[id_dchi2],
                                    linewidths=0,
                                    zorder=-100 + id_dchi2,
                                )
                        # Limits
                        if not ax_options[labels[j]][0] == None:
                            ax[i][j].set_xlim(
                                ax_options[labels[j]][0][0], ax_options[labels[j]][0][1]
                            )
                        if not ax_options[labels[i]][1] == None:
                            ax[i][j].set_ylim(
                                ax_options[labels[i]][1][0], ax_options[labels[i]][1][1]
                            )

                        # Ticks position and occurence
                        if not ax_options[labels[j]][2] == None:
                            ax[i][j].xaxis.set_major_locator(
                                MultipleLocator(ax_options[labels[j]][2][0])
                            )
                            ax[i][j].xaxis.set_minor_locator(
                                AutoMinorLocator(ax_options[labels[j]][2][1])
                            )
                        if not ax_options[labels[i]][3] == None:
                            ax[i][j].yaxis.set_major_locator(
                                MultipleLocator(ax_options[labels[i]][3][0])
                            )
                            ax[i][j].yaxis.set_minor_locator(
                                AutoMinorLocator(ax_options[labels[i]][3][1])
                            )
                    else:
                        if diagonal == "chi2":
                            x = np.linspace(
                                samples[labels[j]].min(), samples[labels[j]].max(), 100
                            )
                            xx = list()
                            yy = list()
                            for k in range(100 - 1):
                                mask = (samples[labels[j]] > x[k]) & (
                                    samples[labels[j]] <= x[k + 1]
                                )
                                if mask.sum() > 0:
                                    y = samples.loc[mask, "dchi2"]
                                    xx.append(np.mean([x[k], x[k + 1]]))
                                    yy.append(np.min(y))
                            ax[i][j].plot(xx, yy, ls="-", lw=1, c="k")

                            ax[i][i].yaxis.set_label_position("right")
                            ax[i][i].spines["left"].set_visible(False)
                            ax[i][i].spines["top"].set_visible(False)
                            ax[i][i].tick_params(
                                which="both",
                                bottom=True,
                                top=False,
                                left=False,
                                right=True,
                                labelbottom=True,
                                labeltop=False,
                                labelleft=False,
                                labelright=True,
                                pad=2,
                            )
                            ax[i][i].set_ylabel(r"$\Delta\chi^2$")
                            ax[i][i].set_ylim(-0.4, 9)
                            ax[i][i].yaxis.set_major_locator(MultipleLocator(2))
                            ax[i][i].yaxis.set_minor_locator(AutoMinorLocator(4))

                            # Choose same ticks for a column
                            if i < N - 1:
                                ax[i][i].sharex(ax[i + 1][i])
                        elif diagonal == "cumul":
                            cdf = self.cdf[labels[j]]
                            ax[i][i].plot(cdf[0], cdf[1], ls="-", lw=1, c="k")

                            if display_1sigma:
                                x = self.ci[labels[j]][1]
                                y = 0.1
                                xerr = np.array(
                                    [
                                        [
                                            x - self.ci[labels[j]][0],
                                            self.ci[labels[j]][2] - x,
                                        ]
                                    ]
                                ).T
                                ax[i][i].errorbar(
                                    x,
                                    y,
                                    xerr=xerr,
                                    marker="o",
                                    ms=1,
                                    lw=0.5,
                                    capsize=0,
                                    color="k",
                                )

                            ax[i][i].yaxis.set_label_position("right")
                            ax[i][i].spines["left"].set_visible(False)
                            ax[i][i].spines["top"].set_visible(False)
                            ax[i][i].tick_params(
                                which="both",
                                bottom=True,
                                top=False,
                                left=False,
                                right=True,
                                labelbottom=True,
                                labeltop=False,
                                labelleft=False,
                                labelright=True,
                                pad=2,
                            )
                            ax[i][i].set_ylabel(r"CDF")
                            ax[i][i].set_ylim(0, 1.05)
                            ax[i][i].yaxis.set_major_locator(MultipleLocator(0.2))
                            ax[i][i].yaxis.set_minor_locator(AutoMinorLocator(4))

                            # Choose same ticks for a column
                            if i < N - 1:
                                ax[i][i].sharex(ax[i + 1][i])

                            # Limits
                            if not ax_options[labels[j]][0] == None:
                                ax[i][j].set_xlim(
                                    ax_options[labels[j]][0][0],
                                    ax_options[labels[j]][0][1],
                                )

                            # Ticks position and occurence
                            if not ax_options[labels[j]][2] == None:
                                ax[i][j].xaxis.set_major_locator(
                                    MultipleLocator(ax_options[labels[j]][2][0])
                                )
                                ax[i][j].xaxis.set_minor_locator(
                                    AutoMinorLocator(ax_options[labels[j]][2][1])
                                )

                        elif diagonal == "density":
                            bins_density_hist = dict()
                            [bins_density_hist.update({label: 100}) for label in labels]
                            bins_density_hist.update(bins_density)
                            if weights & isinstance(self.weights, str):
                                weight_bins, edge_bins, patches = ax[i][i].hist(
                                    self.sample[labels[i]],
                                    bins_density_hist[labels[i]],
                                    density=True,
                                    histtype="step",
                                    color="k",
                                    weights=self.sample[self.weights].values,
                                )
                            else:
                                weight_bins, edge_bins, patches = ax[i][i].hist(
                                    self.sample[labels[i]],
                                    bins_density_hist[labels[i]],
                                    density=True,
                                    histtype="step",
                                    color="k",
                                )

                            if display_1sigma:
                                x = self.ci[labels[j]][1]
                                y = 0.1 * np.amax(weight_bins)
                                xerr = np.array(
                                    [
                                        [
                                            x - self.ci[labels[j]][0],
                                            self.ci[labels[j]][2] - x,
                                        ]
                                    ]
                                ).T
                                ax[i][i].errorbar(
                                    x,
                                    y,
                                    xerr=xerr,
                                    marker="o",
                                    ms=2,
                                    lw=0.5,
                                    capsize=0,
                                    color="k",
                                )

                            ax[i][i].yaxis.set_label_position("right")
                            ax[i][i].spines["left"].set_visible(False)
                            ax[i][i].spines["top"].set_visible(False)
                            ax[i][i].tick_params(
                                which="both",
                                bottom=True,
                                top=False,
                                left=False,
                                right=True,
                                labelbottom=True,
                                labeltop=False,
                                labelleft=False,
                                labelright=True,
                                pad=2,
                            )
                            ax[i][i].set_ylabel(r"Density")

                            # Choose same ticks for a column
                            if i < N - 1:
                                ax[i][i].sharex(ax[i + 1][i])

                            # Limits
                            if not ax_options[labels[j]][0] == None:
                                ax[i][j].set_xlim(
                                    ax_options[labels[j]][0][0],
                                    ax_options[labels[j]][0][1],
                                )

                            # Ticks position and occurence
                            if not ax_options[labels[j]][2] == None:
                                ax[i][j].xaxis.set_major_locator(
                                    MultipleLocator(ax_options[labels[j]][2][0])
                                )
                                ax[i][j].xaxis.set_minor_locator(
                                    AutoMinorLocator(ax_options[labels[j]][2][1])
                                )


        ax[N-2][N-2].tick_params(labelbottom=False)

        if align_xlabels:
            fig.align_xlabels(ax[N - 1, :])
        if align_ylabels:
            fig.align_ylabels(ax[:, 0])

        if not filename == None:
            opts = dict(
                {
                    "transparent": False,
                    "bbox_inches": "tight",
                    "dpi": 300,
                    "pad_inches": 0.01,
                }
            )
            opts.update(saving_options)
            fig.savefig(filename, **opts)
        else:
            return fig, ax
