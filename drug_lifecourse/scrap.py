__author__ = 'thor'

import ut as ms

from numpy import *
import numpy as np
from matplotlib.pyplot import *
import ut.pplot.get


def plot_relrisk_matrix(relrisk):
    t = relrisk.copy()
    matrix_shape = (t['exposure'].nunique(), t['event'].nunique())
    m = ms.daf.to.map_vals_to_ints_inplace(t, cols_to_map=['exposure'])
    m = m['exposure']
    ms.daf.to.map_vals_to_ints_inplace(t, cols_to_map={'event': dict(zip(m, range(len(m))))})
    RR = zeros(matrix_shape)
    RR[t['exposure'], t['event']] = t['relative_risk']
    RR[range(len(m)), range(len(m))] = nan

    RRL = np.log2(RR)
    def normalizor(X):
        min_x = nanmin(X)
        range_x = nanmax(X) - min_x
        return lambda x: (x - min_x) / range_x
    normalize_this = normalizor(RRL)
    center = normalize_this(0)

    from ut.pplot.color import shifted_color_map

    color_map = shifted_color_map(cmap=cm.get_cmap('coolwarm'),
                                  start=0, midpoint=center, stop=1)
    imshow(RRL, cmap=color_map, interpolation='none');

    xticks(range(shape(RRL)[0]), m, rotation=90)
    yticks(range(shape(RRL)[1]), m)
    cbar = colorbar()
    cbar.ax.set_yticklabels(["%.02f" % x for x in np.exp2(array(ms.pplot.get.get_colorbar_tick_labels_as_floats(cbar)))])