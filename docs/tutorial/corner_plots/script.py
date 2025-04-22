import matplotlib as mpl
import moana
import numpy as np
import pandas as pd

if __name__ == '__main__':

    # Path, file names and MCMC parameters
    path_mcmc = "./"
    run_name = 'mcmc_samples'
    N_burn = 100

    # --- Load MCMC file ---
    # Define column names
    colnames = np.array(['chi2', 'invtE', 't0', 'u0', 'sep', 'theta', 'eps1', '1oTbin', 'vsep', 'tS', 't_fix', 'piex', 'piey'])
    
    # Add flux for each dataset
    datasets = ['k']
    flux = np.array([f"fs_{a} fb_{a}".split(' ') for a in datasets])
    colnames = np.append(colnames, flux)

    # Number of columns
    cols = range(len(colnames))

    # Load MCMC output
    fname = f'{path_mcmc}/{run_name}.dat'
    print(f'Loading {fname}...')
    data = pd.read_table(fname, sep=r'\s+', names=colnames, usecols=cols, 
        dtype=np.float64, skiprows=N_burn)

    # --- Preparation of the Dataframe ---
    # Remove column if parameter is constant
    data = data.loc[:, (data != data.iloc[0]).any()]

    # Add some missing quantities
    data['dchi2'] = data['chi2'] - np.min(data['chi2'])
    data['tE'] = 1 / data['invtE']
    data['q'] = moana.dbc.mass_fration_to_mass_ratio(data['eps1'])
    data['qrescaled'] = 1e2 * data['q']

    # Reset Dataframe index
    data.reset_index(drop=True, inplace=True)

    # Display some basic information
    print('Preview of first few lines...')
    print(data.head(3))
    print(f"Minimum chi-square: {data['chi2'].min():.2f}")
    print(f"Sample size: {len(data)}")

    # --- Create object to get statistical info, including scatter plot  ---
    # Choose here which parameters you want to include in the corner plot
    labels = ['sep', 'qrescaled', 'tE', 'u0']

    # Create the object
    posterior = moana.SampledPosterior(data, labels=labels)

    # --- Plot options ---

    # Define label names using LaTeX
    # If not defined, label names will be parameter name.
    column_labels = dict()
    column_labels.update({
        'sep': 's',
        'tE': r't_E\ (days)', 
        'u0': r'u_0',
        'q': 'q',
        'qrescaled': r'q/10^2',
    })

    # Range, major and minor ticks locators
    # -- Example 1:
    # {'tE': [[11.7, 17.1], [10.1, 19], [2, 4], [1, 2]]}
    # means for the subplots:
    # * tE_min = 11.7, tE_max = 17.1 when tE is represented on x-axis
    # * tE_min = 10.1, tE_max = 19 when tE is represented on y-axis
    # * Major ticks will be mutiples of 2 when tE is represented on x-axis
    # * Major ticks will be mutiples of 1 when tE is represented on x-axis
    # * There will be 4 minor ticks between each major ticks when tE is represented on x-axis
    # * There will be 2 minor ticks between each major ticks when tE is represented on y-axis
    # -- Example 2:
    # {'tE': [None, [10.1, 19], [2, 4], None]}
    # means that the choice of x range and y ticks will be automated.
    # TIP --> It is highly recommanded to start without specifying any ax_opts, because
    # if you ask for too many ticks, then the code may crash. If you do not
    # specify any ax_opts, then everything will be automated.
    ax_opts = dict()
    ax_opts.update({
        'u0': [[0.7, 1.25], [0.7, 1.25], [0.2, 4], [0.2, 4]],
        'tE': [[11.7, 17.1], [11.7, 17.1], [2, 4], [2, 4]],
        'sep': [[0.4, 0.53], [0.4, 0.53], [0.05, 5], [0.05, 5]],
        'qrescale': [[0.42, 0.75], [0.42, 0.75], [0.1, 4], [0.1, 4]],
    })

    # Bins for 2-dimensional contours
    # -- Example 1:
    # {'u0tE': [100, 80]}
    # means for the subplot where u0 is on the x axis, and tE on the y axis:
    # * 100 bins will be used in the x axis direction
    # * 80 bins will be used in the y axis direction
    # TIP --> It is highly recommanded to start without specifying any bins.
    # If you do not specify any bins, then 80 bins in both x and y axes will
    # be used.
    bins = dict()  # We create a dictionnary
    # Let's impose 20 bins by default for all the plots, instead of 80.
    [bins.update({labels[i]+labels[j]: [20, 20]}) 
        for i in range(len(labels)) for j in range(len(labels)) if j>i]
    # Now, let's increase the bins number for 2 plots manualy
    bins.update({
        'septE': [100, 80],
        'sepu0': [100, 80],
    })

    # --- Plot the beautiful corner plot: EXAMPLE 1 ---
    print("Example 1...")
    fname = 'Example_1.pdf'
    posterior.corner_plot(filename=fname, labels=column_labels, axes_options=ax_opts,
        bins=bins, rotation=30)

    # --- Plot the beautiful corner plot: EXAMPLE 2 ---
    # We want more to change saving options (in function Figure.save())
    # We also want to show the median and 1 sigma on the diagonal plots.
    print("Example 2...")
    fname = 'Example_2.pdf'
    output_opts = dict({'pad_inches': 0.1, 'dpi': 200})
    posterior.corner_plot(filename=fname, labels=column_labels, axes_options=ax_opts,
        bins=bins, saving_options=output_opts, rotation=30, display_1sigma=True)

    # --- Plot the beautiful corner plot: EXAMPLE 3 ---
    # We want chi-square in diagonals
    print("Example 3...")
    fname = 'Example_3.pdf'
    posterior.corner_plot(filename=fname, labels=column_labels, axes_options=ax_opts,
        bins=bins, diagonal='chi2', rotation=30)

    # --- Plot the beautiful corner plot: EXAMPLE 4 ---
    # We want to see the samples and hide the 1-3 sigmas filled contours
    print("Example 4...")
    fname = 'Example_4.png'  # Can be also a PDF, but PDF may become very large files.
    posterior.corner_plot(filename=fname, labels=column_labels, axes_options=ax_opts,
        bins=bins, diagonal='chi2', show_samples=True, credible_intervals=False,
        rotation=30)

    # --- Plot the beautiful corner plot: EXAMPLE 5 ---
    # Same as example 1, but we choose other levels. By default, the correct
    # 1-, 2-, 3-sigma contours (in 2D) are displayed.
    print("Example 5...")
    fname = 'Example_5.pdf'
    levels =[0.16, 0.68, 0.84, 0.95]
    contourf_opts = {'colors': None}  # or any matplotlib color map for fonction Axes.contourf. None will call default value
    posterior.corner_plot(filename=fname, labels=column_labels, axes_options=ax_opts,
        bins=bins, cdf_levels=levels, contourf_options=contourf_opts, rotation=30)

    # --- Plot the beautiful corner plot: EXAMPLE 6 ---
    # Same as example 5 but:
    # * we change the bin numbers for the plot tE(x-axis) vs. u0(y-axis)
    # * we change the range and ticks of qrescaled, tE and u0.
    # * we remove the rotation of the ticks
    print("Example 6...")
    fname = 'Example_6.pdf'
    levels =[0.16, 0.68, 0.84, 0.95]
    contourf_opts = {'colors': None}  # or any matplotlib color map for fonction Axes.contourf. None will call default value
    bins.update({'tEu0': [300, 300]})
    ax_opts.update({
        'qrescaled': [[0.45, 0.7], [0.45, 0.7], [0.1, 2], [0.1, 2]],
        'tE': [[12.7, 16.5], [12.7, 16.5], [1, 4], [1, 4]],
        'u0': [[0.8, 1.18], [0.8, 1.18], [0.1, 4], [0.1, 4]],
        })
    posterior.corner_plot(filename=fname, labels=column_labels, axes_options=ax_opts,
        bins=bins, cdf_levels=levels, contourf_options=contourf_opts)

    # --- Plot the beautiful corner plot: EXAMPLE 7 ---
    # Everything automated, no option given, exept the filename to save the
    # plot, and we ask to display the Axis number to help to understand Example 8.
    print("Example 7...")
    fname = 'Example_7.pdf'
    posterior.corner_plot(filename=fname, display_plot_coords=True)

    # --- Plot the beautiful corner plot: EXAMPLE 8 ---
    # We show how you can plot anything you want in any subplot, and control the
    # content of any subplot, as well as saving options etc.
    print("Example 8...")
    fig, axes = posterior.corner_plot(labels=column_labels, axes_options=ax_opts, bins=bins,
        rotation=40, display_plot_coords=True)

    # Edit any subplot
    axes[1][1].clear()  # Not required if you just want to add something in the subplot
    axes[1][1].set_ylabel("Density")
    axes[1][1].hist(data[labels[1]], 50, density=True, histtype='step', color='k')
    axes[1][1].set_xticks(axes[3][1].get_xticks())
    axes[1][1].set_xticks(axes[3][1].get_xticks(minor=True), minor=True)
    axes[1][1].set_xticklabels([])
    axes[1][1].set_xlim(axes[3][1].get_xlim()[0], axes[3][1].get_xlim()[1]) # This should be the last line

    # We can also change any spacing, if we want!
    fig.subplots_adjust(hspace=0)

    # Save the figure
    fig.savefig('Example_8.pdf', transparent=False, bbox_inches='tight', dpi=300,
                    pad_inches=0.1)

    # --- Plot the beautiful corner plot: EXAMPLE 9 ---
    # Same as example 1. We show how to parse Matplotlib rcParameters to
    # control everything, inclusing fonts etc.
    # For more details, see:
    # https://matplotlib.org/3.2.1/tutorials/introductory/customizing.html
    print("Example 9...")
    fname = 'Example_9.pdf'
    rcparams=dict({'xtick.major.size': 8, 'xtick.labelsize': 12})
    posterior.corner_plot(filename=fname, labels=column_labels, axes_options=ax_opts,
        bins=bins, rcparams=rcparams)
    mpl.rcParams.update(mpl.rcParamsDefault)  # Restaure default Matplotlib rcparams for next examples

    # --- Plot the beautiful corner plot: EXAMPLE 10 ---
    # We want probability density function in all the diagonals.
    # The bins for each diagonal plot can be set individually.
    print("Example 10...")
    fname = 'Example_10.png'
    bins_for_pdf = dict()
    bins_for_pdf.update({
        'u0': 10,
        'tE': 1000,
    })
    posterior.corner_plot(filename=fname, labels=column_labels, axes_options=ax_opts,
                          rotation=30, display_1sigma=True,
                          diagonal='density', bins_density=bins_for_pdf,
                          bins=bins)

    # Note 1: If an option in rcparams is not respected, then it means that
    # this option has been changed in the plotting routine. If it happens,
    # follow EXAMPLE 8 to get the matplotlib Figure and Axes. Then, you can
    # change everything you want, before saving the plot.
    #
    # Note 2: You may write your own rcParams file for matplotlib. To parse it,
    # add the option rcfile='the-path/your-style-name.mplstyle'. If your custom 
    # file is installed in the default Matplotlib location, then you can add simply,
    # rcfile='your-file-name' option in posterior.corner_plot().
















