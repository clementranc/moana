import numpy as np

# Two-dimensional contour computation
def compute_2dcontours(x, y, bins=50, cdf_levels=None, weights=None):
    
    # Choose the default "sigma" contour levels.
    if cdf_levels is None:
        cdf_levels = 1 - np.exp(-0.5 * np.array([1.0, 2.0, 3.0])**2)

    # We'll make the 2D histogram to directly estimate the density.
    try:
        hist, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=weights)
    except ValueError:
        raise ValueError(
            "It looks like at least one of your sample columns "
            "have no dynamic range. You could try using the "
            "'range' argument."
        )
    
    # Compute the bin centers
    xcenter = 0.5 * (np.roll(xedges, -1) + xedges)[:-1]
    ycenter = 0.5 * (np.roll(yedges, -1) + yedges)[:-1]
    
    # Compute the density levels
    hist1d = hist.flatten()
    hist1d = hist1d[np.argsort(hist1d)[::-1]]
    cumul = np.cumsum(hist1d)
    if weights is not None:
        print("Test for weighted histogram: okay is {:.9e} ~ 0.".format(
        cumul[-1] - np.cumsum(weights)[-1]))
    cumul = cumul / cumul[-1]
    levels = np.zeros(len(cdf_levels))
    for i in range(len(cdf_levels)):
        try:
            levels[i] = hist1d[cumul <= cdf_levels[i]][-1]
        except:
            levels[i] = hist1d[0]
    levels = np.sort(levels)
    
    # [Optional] Increase array size for plots
    hist2 = hist.min() + np.zeros((hist.shape[0] + 4, hist.shape[1] + 4))
    hist2[2:-2, 2:-2] = hist
    hist2[2:-2, 1] = hist[:, 0]
    hist2[2:-2, -2] = hist[:, -1]
    hist2[1, 2:-2] = hist[0]
    hist2[-2, 2:-2] = hist[-1]
    hist2[1, 1] = hist[0, 0]
    hist2[1, -2] = hist[0, -1]
    hist2[-2, 1] = hist[-1, 0]
    hist2[-2, -2] = hist[-1, -1]
    xcenter2 = np.concatenate([
        xcenter[0] + np.array([-2, -1]) * np.diff(xcenter[:2]),
        xcenter,
        xcenter[-1] + np.array([1, 2]) * np.diff(xcenter[-2:])])
    ycenter2 = np.concatenate([
        ycenter[0] + np.array([-2, -1]) * np.diff(ycenter[:2]),
        ycenter,
        ycenter[-1] + np.array([1, 2]) * np.diff(ycenter[-2:])])
    
    return xcenter2, ycenter2, hist2, levels























