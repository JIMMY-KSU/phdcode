import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt


def centered_colormap(data, start=0.0, midpoint=None, stop=1.0, cmap=None, name='tmp'):
    # Modified from stackoverflow:
    # https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib

    if cmap is None:
        cmap = plt.get_cmap('viridis')

    if midpoint is None:
        max = np.max(data)
        min = np.min(data)
        midpoint = 1.0 - max/(max + abs(min))

    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


def viridis_baseline(basecolor=None):
    if basecolor is None:
        basecolor = (1.,1.,1.,1.)

    cmap = plt.get_cmap('viridis')
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    index = np.linspace(0.0, 1.0, 257)

    cdict['red'].append((index[0], basecolor[0], basecolor[0]))
    cdict['green'].append((index[0], basecolor[1], basecolor[1]))
    cdict['blue'].append((index[0], basecolor[2], basecolor[2]))
    cdict['alpha'].append((index[0], basecolor[3], basecolor[3]))
    for i in index[1:]:
        r, g, b, a = cmap(i)

        cdict['red'].append((i, r, r))
        cdict['green'].append((i, g, g))
        cdict['blue'].append((i, b, b))
        cdict['alpha'].append((i, a, a))

    newcmap = LinearSegmentedColormap('viridis_baseline', cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap
