import os
import math
from pathlib import Path
import pymoto as pym
import matplotlib.pyplot as plt
import matplotlib.patches as plt_patch
import numpy as np


# UTILITIES
def unwrap(angle, period=2*math.pi):
    """Unwrap a phase angle to give a continuous curve

    Parameters
    ----------
    angle : array_like
        Array of angles to be unwrapped
    period : float, optional
        Period (defaults to `2*pi`)

    Returns
    -------
    angle_out : array_like
        Output array, with jumps of period/2 eliminated

    Examples
    --------
    import numpy as np
    theta = [5.74, 5.97, 6.19, 0.13, 0.35, 0.57]
    unwrap(theta, period=2 * np.pi)
    [5.74, 5.97, 6.19, 6.413185307179586, 6.633185307179586, 6.8531853071795865]

    """
    dangle = np.diff(angle)
    dangle_desired = (dangle + period/2.) % period - period/2.
    correction = np.cumsum(dangle_desired - dangle, axis=-1)
    angle[..., 1:] += correction
    return angle


class GeneralPlot:
    """ Plot helper for plots that are interactively updated """
    def __init__(self, ax, *args, **kwargs):
        self.ax = ax
        self.args = args
        self.kwargs = kwargs
        self.lines = None
        self.shape = None

    def update(self):
        raise NotImplementedError("Update function should be implemented")

    @staticmethod
    def get_data(obj):
        return np.asarray(obj.state if hasattr(obj, 'state') else obj)

    def get_args(self, idx):
        # Parse arguments
        arguments = []
        for a in self.args:
            try:
                aa = np.asarray(a)
                assert aa.shape[len(self.shape)] == self.shape
                arguments.append(aa[idx[:len(self.shape)]])
            except:
                arguments.append(a)
        return arguments

    def get_kwargs(self, idx):
        # Parse keyword arguments
        kwarguments = {}
        for k, a in self.kwargs.items():
            try:
                aa = np.asarray(a)
                assert aa.shape[0:len(self.shape)] == self.shape
                kwarguments[k] = aa[idx[:len(self.shape)]]
            except:
                if k == "label":
                    a = a + " " + str(idx)
                kwarguments[k] = a
        return kwarguments

    def init_lines(self, shape):
        if self.lines is not None:
            return
        self.shape = shape
        self.lines = np.ndarray(shape=self.shape, dtype=np.object)
        for (idx, line) in np.ndenumerate(self.lines):
            arguments = self.get_args(idx)
            kwarguments = self.get_kwargs(idx)
            self.lines[idx] = self.ax.plot([], [], *arguments, **kwarguments)[0]


class UpdatePlotXY(GeneralPlot):
    """ Helper to update XY-data plots """
    def __init__(self, ax, *args, complex_data=None, x_data=None, y_data=None, which_x=Ellipsis, which_y=Ellipsis, **kwargs):
        super().__init__(ax, *args, **kwargs)
        self.which_x = which_x
        self.which_y = which_y
        if complex_data is None and x_data is None and y_data is None:
            return
        assert (complex_data is None) != (x_data is None), "Either (xdata, ydata) or complexdata must be set"
        assert (complex_data is None) != (y_data is None), "Either (xdata, ydata) or complexdata must be set"
        self.complexmode = complex_data is not None
        self.complex_data = complex_data
        self.x_data, self.y_data = x_data, y_data

    def update(self, xdata=None, ydata=None):
        if xdata is None or ydata is None:
            if self.complexmode:
                compldat = self.get_data(self.complex_data)
                xdata, ydata = np.real(compldat), np.imag(compldat)
            else:
                xdata, ydata = self.get_data(self.x_data), self.get_data(self.y_data)
        self.set_xydata(xdata[self.which_x], ydata[self.which_y])

    def set_xydata(self, xdata, ydata):
        # Parse into numpy array
        xdata, ydata = np.asarray(xdata), np.asarray(ydata)

        # Check if x and y sizes are conforming
        if xdata.shape != ydata.shape:
            if xdata.size == 1:
                xdata = np.ones_like(ydata)*xdata.flatten()
            elif ydata.size == 1:
                ydata = np.ones_like(xdata)*ydata.flatten()
            elif xdata.shape[-1] == ydata.shape[-1]:
                if np.prod(xdata.shape[:-1], dtype=int) == 1:
                    xdata = np.ones_like(ydata) * xdata.flatten()
                elif np.prod(ydata.shape[:-1], dtype=int) == 1:
                    ydata = np.ones_like(xdata) * ydata.flatten()
            elif xdata.shape[0] == ydata.shape[0]:
                if np.prod(xdata.shape[1:], dtype=int) == 1:
                    xdata = np.broadcast_to(xdata, reversed(ydata.shape))
                    ydata = ydata.T
                elif np.prod(ydata.shape[1:], dtype=int) == 1:
                    ydata = np.broadcast_to(ydata, reversed(xdata.shape))
                    xdata = xdata.T

        self.init_lines(xdata.shape[:-1])
        for (idx, line) in np.ndenumerate(self.lines):
            line.set_xdata(xdata[idx])
            line.set_ydata(ydata[idx])

    def set_xdata(self, xdata):
        xdata = np.asarray(xdata)
        self.init_lines(xdata.shape[:-1])
        for (idx, line) in np.ndenumerate(self.lines):
            line.set_xdata(xdata[idx])

    def set_ydata(self, ydata):
        ydata = np.asarray(ydata)
        self.init_lines(ydata.shape[:-1])
        for (idx, line) in np.ndenumerate(self.lines):
            line.set_xdata(ydata[idx])


class UpdatePlotCircle(GeneralPlot):
    """ Helper for plots with circles """
    def __init__(self, ax, midp, radius, *args, which=Ellipsis, **kwargs):
        super().__init__(ax, *args, **kwargs)
        self.midp = midp
        self.radius = radius
        self.circles = None
        self.which = which

    def init_circles(self, shape):
        if self.circles is not None:
            return
        if len(shape)>1:
            self.shape = shape[:-1]
        else:
            self.shape = shape
        self.circles = np.ndarray(shape=shape, dtype=np.object)
        for (idx, circ) in np.ndenumerate(self.circles):
            # Parse arguments
            arguments = self.get_args(idx)
            kwarguments = self.get_kwargs(idx)
            self.circles[idx] = plt_patch.Circle((0, 0), 0, *arguments, **kwarguments)
            self.ax.add_artist(self.circles[idx])

    def update(self):
        midpt = self.get_data(self.midp)
        x, y = np.real(midpt), np.imag(midpt)
        r = self.get_data(self.radius)
        x = np.asarray(x)[self.which]
        y = np.asarray(y)[self.which]
        r = np.asarray(r)[self.which]
        assert x.shape == y.shape == r.shape, "Midpoints and radius must be of equal size"
        self.init_circles(x.shape)

        for (idx, circ) in np.ndenumerate(self.circles):
            circ.center = x[idx], y[idx]
            circ.set_radius(r[idx])


class UpdatePlotInterpXY(UpdatePlotXY):
    """ Interpolates parametric xy-data (x(w), y(w)) at points w_i """
    def __init__(self, ax, w_inter_pts,  *args, complex_data=None, x_data=None, y_data=None, w_data=None, which_x=Ellipsis, which_y=Ellipsis, **kwargs):
        super().__init__(ax, *args, complex_data=complex_data, x_data=x_data, y_data=y_data, which_x=which_x, which_y=which_y, **kwargs)
        self.interpolation_points = w_inter_pts
        self.w_data = w_data

    def update(self):
        if self.complexmode:
            compldat = self.get_data(self.complex_data)
            xdata, ydata = np.real(compldat), np.imag(compldat)
        else:
            xdata, ydata = self.get_data(self.x_data), self.get_data(self.y_data)
        wdata = self.get_data(self.w_data)
        wdata = np.imag(wdata) if np.any(np.iscomplex(wdata)) else wdata
        w_inter = self.get_data(self.interpolation_points)
        w_inter = np.imag(w_inter) if np.any(np.iscomplex(w_inter)) else w_inter
        # Remove frequencies out of bound
        w_inter = w_inter[w_inter >= np.min(wdata)]
        w_inter = w_inter[w_inter <= np.max(wdata)]

        x_inter = np.interp(w_inter, wdata, xdata[self.which_x])
        y_inter = np.interp(w_inter, wdata, ydata[self.which_y])
        self.set_xydata(x_inter, y_inter)


class UpdatePlotInterpY(UpdatePlotXY):
    """ Interpolates ydata for y(x) at x, with given y(x_i)"""
    def __init__(self, ax, x_inter_pts,  *args, complex_data=None, x_data=None, y_data=None, which_x=Ellipsis, which_y=Ellipsis, **kwargs):
        super().__init__(ax, *args, complex_data=complex_data, x_data=x_data, y_data=y_data, which_x=which_x, which_y=which_y, **kwargs)
        self.interpolation_points = x_inter_pts

    def update(self, xdata=None, ydata=None):
        if xdata is None or ydata is None:
            if self.complexmode:
                compldat = self.get_data(self.complex_data)
                xdata, ydata = np.real(compldat), np.imag(compldat)
            else:
                xdata, ydata = self.get_data(self.x_data), self.get_data(self.y_data)

        x_inter = self.get_data(self.interpolation_points)
        x_inter = np.imag(x_inter) if np.any(np.iscomplex(x_inter)) else x_inter
        # Remove frequencies out of bound
        x_inter = x_inter[x_inter >= np.min(xdata)]
        x_inter = x_inter[x_inter <= np.max(xdata)]

        xdata, ydata, x_inter = xdata[self.which_x], ydata[self.which_y], x_inter[self.which_x]
        y_inter = np.zeros((*ydata.shape[:-1], x_inter.shape[0]))
        for idx in np.ndindex(ydata.shape[:-1]):
            y_inter[idx] = np.interp(x_inter, xdata, ydata[idx])
        self.set_xydata(x_inter, y_inter)


# PLOTTING MODULES
class PlotNyquist(pym.Module):
    def _prepare(self, stab_lim=6.0, saveto="out/nyquist.png", iter=0, xlim=[-3, 1], ylim=[-2, 2], title="Nyquist"):
        self.stab_lim = stab_lim
        if saveto is not None:
            Path(saveto).parent.mkdir(parents=True, exist_ok=True)
            self.saveloc, self.saveext = os.path.splitext(saveto)
        else:
            self.saveloc, self.saveext = None, None
        self.iter = iter
        self.plotlist = []
        self.title = title

        plt.ion()
        self.fig, _ = plt.subplots(1, 1)
        self.ax = self.fig.axes[0]
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_aspect('equal')

        self.ax.set_xlabel('Real')
        self.ax.set_ylabel('Imag')
        if stab_lim is not None:
            UpdatePlotCircle(self.ax, -1.0, 1/(10**(stab_lim/20)), label=f"{stab_lim} dB", fill=False, linestyle='--', color='k').update()
            UpdatePlotXY(self.ax, 'k+', complex_data=-1).update()
        self.draw()
        self.leg = False

        plt.show()

    def __del__(self):
        if hasattr(self, 'fig'):
            plt.close(self.fig)

    def add_xyplot(self, *args, complex_data=None, x_data=None, y_data=None, **kwargs):
        self.plotlist.append(UpdatePlotXY(self.ax, *args, complex_data=complex_data, x_data=x_data, y_data=y_data, **kwargs))

    def add_circles(self, midp, radius, *args, **kwargs):
        self.plotlist.append(UpdatePlotCircle(self.ax, midp, radius, *args, **kwargs))

    def add_interp(self, complex_data, freq_data, freq_points, *args, **kwargs):
        self.plotlist.append(UpdatePlotInterpXY(self.ax, complex_data, freq_data, freq_points, *args, **kwargs))

    def draw(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _response(self, *g):
        self.ax.set_title(f"{self.title}, It. {self.iter}")
        [p.update() for p in self.plotlist]
        if not self.leg:
            self.ax.legend()
            self.leg = True
        self.draw()

        if self.saveloc is not None:
            self.fig.savefig("{0:s}.{1:04d}{2:s}".format(self.saveloc, self.iter, self.saveext))


class PlotSensitivity(pym.Module):
    def _prepare(self, stab_lim=6.0, saveto="out/sensitivity.png", iter=0, ylim=None):
        self.stab_lim = stab_lim
        if saveto is not None:
            Path(saveto).parent.mkdir(parents=True, exist_ok=True)
            self.saveloc, self.saveext = os.path.splitext(saveto)
        else:
            self.saveloc, self.saveext = None, None
        self.iter = iter
        self.plotlist = []
        self.ylist = []
        self.rangelist = []
        self.leg = False
        self.ylim = ylim

        # Signal for the limits
        self.s_ylimits = pym.Signal('vertical limits')
        self.s_xlimits = pym.Signal('horizontal limits')

        # Create figure
        plt.ion()
        self.fig, _ = plt.subplots(1, 1)
        self.ax = self.fig.axes[0]

        self.ax.set_xscale('log')
        self.ax.grid(True, which='both')
        self.ax.set_xlabel("Frequency (rad/s)")
        self.ax.set_ylabel("Amplitude (dB)")
        plt.show()

        # Add limit line
        self.add_horizontal_line(self.stab_lim, 'k--', label="Stability limit")

    def __del__(self):
        if hasattr(self, 'fig'):
            plt.close(self.fig)

    def add_vertical_line(self, s_x, *args, **kwargs):
        self.plotlist.append(UpdatePlotXY(self.ax, *args, x_data=s_x, y_data=self.s_ylimits, **kwargs))

    def add_horizontal_line(self, s_y, *args, **kwargs):
        self.plotlist.append(UpdatePlotXY(self.ax, *args, x_data=self.s_xlimits, y_data=s_y, **kwargs))

    def add_limit_line(self, s_y, range_inds, *args, s_x=None, **kwargs):
        n = len(self.rangelist)
        s_xrange = pym.Signal(f"range{n}")
        self.rangelist.append((range_inds, s_xrange, s_x))
        self.add_xyplot(s_xrange, s_y, *args, **kwargs)

    def add_xyplot(self, y_data, *args, s_x=None, **kwargs):
        self.ylist.append(y_data)
        self.plotlist.append(UpdatePlotXY(self.ax, *args, x_data=s_x if s_x is not None else self.sig_in[0], y_data=y_data, **kwargs))

    def add_interp(self, y_data, x_points, *args, s_x=None, **kwargs):
        self.plotlist.append(UpdatePlotInterpY(self.ax, s_x if s_x is not None else self.sig_in[0], y_data, x_points, *args, **kwargs))

    def draw(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _response(self, w_range):
        self.ax.set_title('Sensitivity, It. {}'.format(self.iter))

        # Update ranges
        for (r, s, sx) in self.rangelist:
            if sx is None:
                s.state = np.array([w_range[r[0]], w_range[r[1]]])
            else:
                xdata = sx.state if hasattr(sx, 'state') else sx
                s.state = np.array([xdata[r[0]], xdata[r[1]]])

        # Find upper and lower limits
        wmin = np.min(w_range)
        wmax = np.max(w_range)
        self.s_xlimits.state = np.array([wmin, wmax])
        ymin = 0.0
        ymax = 10.0
        for s_y in self.ylist:
            ydata = s_y.state if hasattr(s_y, 'state') else s_y
            ymin = np.floor((min(np.min(ydata), ymin) - 5.0) / 10.0) * 10.0
            ymax = np.ceil((max(np.max(ydata), ymax) + 5.0) / 10.0) * 10.0
        if self.ylim:
            ymin, ymax = self.ylim
        self.s_ylimits.state = np.array([ymin, ymax])

        [p.update() for p in self.plotlist]
        self.ax.set_xlim([wmin, wmax])
        self.ax.set_ylim([ymin, ymax])
        if not self.leg:
            self.ax.legend()
            self.leg = True
        self.draw()

        if self.saveloc is not None:
            self.fig.savefig("{0:s}.{1:04d}{2:s}".format(self.saveloc, self.iter, self.saveext))


class PlotBode(pym.Module):
    def _prepare(self, saveto="out/bode.png", iter=0, norm_scale='dB', phase_wrap=False):
        if saveto is not None:
            Path(saveto).parent.mkdir(parents=True, exist_ok=True)
            self.saveloc, self.saveext = os.path.splitext(saveto)
        else:
            self.saveloc, self.saveext = None, None
        self.iter = iter
        self.plotlist = []
        self.transferlist = []
        self.interplist = []
        self.leg = False
        self.norm_scale = norm_scale
        self.phase_wrap = phase_wrap

        # Signal for the limits
        self.s_ylimits_nrm = pym.Signal('vertical limits norm')
        self.s_ylimits_phs = pym.Signal('vertical limits phase')
        self.s_xlimits = pym.Signal('horizontal limits')

        # Create figure
        plt.ion()
        self.fig, _ = plt.subplots(2, 1, sharex=True)
        self.ax_nrm, self.ax_phs = self.fig.axes

        if self.norm_scale == 'log':
            self.ax_nrm.set_yscale('log')
        self.ax_nrm.set_xscale('log')
        self.ax_nrm.grid(True, which='both')
        self.ax_nrm.set_ylabel('Amplitude')

        self.ax_phs.set_xscale('log')
        self.ax_phs.grid(True, which='both')
        self.ax_phs.set_ylabel('Phase')
        self.ax_phs.set_xlabel('Frequency')
        plt.show()

    def __del__(self):
        if hasattr(self, 'fig'):
            plt.close(self.fig)

    def draw(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def is_tagged(self, tag):
        both_tags = ['both', 'all']
        norm_tags = ['norm', 'nrm', 'amplitude', 'a']
        phase_tags = ['phase', 'phs', 'pha', 'p', 'angle']
        is_both = tag.lower() in both_tags
        is_norm = is_both or tag.lower() in norm_tags
        is_phase = is_both or tag.lower() in phase_tags

        if not (is_phase or is_norm):
            raise ValueError(f"Invalid argument 'which' = {tag}")
        return is_norm, is_phase

    def add_vertical_line(self, s_x, *args, which='norm', **kwargs):
        is_norm, is_phase = self.is_tagged(which)
        if is_norm:
            self.plotlist.append(UpdatePlotXY(self.ax_nrm, *args, x_data=s_x, y_data=self.s_ylimits_nrm, **kwargs))
        if is_phase:
            self.plotlist.append(UpdatePlotXY(self.ax_phs, *args, x_data=s_x, y_data=self.s_ylimits_phs, **kwargs))

    def add_horizontal_line(self, s_y, *args, which='norm', **kwargs):
        is_norm, is_phase = self.is_tagged(which)
        if is_norm:
            self.plotlist.append(UpdatePlotXY(self.ax_nrm, *args, x_data=self.s_xlimits, y_data=s_y, **kwargs))
        if is_phase:
            self.plotlist.append(UpdatePlotXY(self.ax_phs, *args, x_data=self.s_xlimits, y_data=s_y, **kwargs))

    def add_transfer(self, s_G, *args, **kwargs):
        self.transferlist.append((s_G, UpdatePlotXY(self.ax_nrm, *args, **kwargs), UpdatePlotXY(self.ax_phs, *args, **kwargs)))

    def add_interp(self, s_G, x_points, *args, s_x=None, **kwargs):
        if s_x is None:
            s_x = self.sig_in[0]
        self.interplist.append((s_G, UpdatePlotInterpY(self.ax_nrm, x_points, *args, x_data=s_x, y_data=0, **kwargs),
                                     UpdatePlotInterpY(self.ax_phs, x_points, *args, x_data=s_x, y_data=0, **kwargs)))

    def _response(self, w_range):
        self.ax_nrm.set_title('Open loop, It. {}'.format(self.iter))

        # Update TFs
        nrm_min = 1e+100
        nrm_max = -1e+100
        pha_min = 1e+100
        pha_max = -1e+100
        w_min = min(w_range)
        w_max = max(w_range)

        for (Gs, line_nrm, line_phs) in self.transferlist:
            G = Gs.state if hasattr(Gs, 'state') else Gs
            G_nrm = np.real(np.sqrt(G*np.conj(G)))
            G_dB = 20*np.log10(G_nrm) if self.norm_scale == 'dB' else G_nrm
            finites = np.isfinite(G_dB)
            G_pha = np.angle(G, deg=True)
            if self.phase_wrap:
                G_pha = unwrap(G_pha, period=360)

            if self.norm_scale == 'dB':
                nrm_min = np.floor((min(np.min(G_dB[finites]), nrm_min) - 5.0) / 10.0) * 10.0
                nrm_max = np.ceil((max(np.max(G_dB[finites]), nrm_max) + 5.0) / 10.0) * 10.0
            else:
                nrm_min = min(np.min(G_dB[finites])*0.9, nrm_min)
                nrm_max = max(np.max(G_dB[finites])*1.1, nrm_max)

            line_nrm.update(w_range, G_dB)
            line_phs.update(w_range, G_pha)
            pha_min = np.floor((min(np.min(G_pha), pha_min) - 5.0) / 45.0) * 45.0
            pha_max = np.ceil((max(np.max(G_pha), pha_max) + 5.0) / 45.0) * 45.0

        for (Gs, line_nrm, line_phs) in self.interplist:
            G = Gs.state if hasattr(Gs, 'state') else Gs
            G_nrm = np.real(np.sqrt(G * np.conj(G)))
            G_dB = 20 * np.log10(G_nrm) if self.norm_scale == 'dB' else G_nrm
            G_pha = np.angle(G, deg=True)
            if self.phase_wrap:
                G_pha = unwrap(G_pha, period=360)
            line_nrm.update(w_range, G_dB)
            line_phs.update(w_range, G_pha)

        self.s_xlimits.state = np.array([w_min, w_max])
        self.s_ylimits_nrm.state = np.array([nrm_min, nrm_max])
        self.s_ylimits_phs.state = np.array([pha_min, pha_max])

        [p.update() for p in self.plotlist]
        self.ax_nrm.set_xlim([w_min, w_max])
        self.ax_nrm.set_ylim([nrm_min, nrm_max])

        self.ax_phs.set_xlim([w_min, w_max])
        self.ax_phs.set_ylim([pha_min, pha_max])
        self.ax_phs.set_yticks(np.arange(np.ceil(pha_min / 90) * 90, np.floor(pha_max / 90) * 90, 90))
        self.ax_phs.set_yticks(np.arange(pha_min, pha_max, 45), minor=True)

        if not self.leg:
            self.ax_phs.legend(loc='right')
            self.leg = True
        self.draw()

        if self.saveloc is not None:
            self.fig.savefig("{0:s}.{1:04d}{2:s}".format(self.saveloc, self.iter, self.saveext))


class Print(pym.Module):
    """ Print signals to console """
    def _prepare(self, showmax=False):
        self.showmax = showmax

    def _response(self, *args):
        for i, xx in enumerate(args):
            if self.showmax and hasattr(xx, 'shape'):
                print(f"Signal \"{self.sig_in[i].tag}\" : [max = {np.max(xx)}] {xx}")
            else:
                print(f"Signal \"{self.sig_in[i].tag}\" : {xx}")

        return []


