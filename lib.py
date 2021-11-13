#!/usr/bin/env python3

# Copyright (c) 2021 Dominik Madea

import numpy as np 
import math

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FixedLocator, ScalarFormatter
from matplotlib import cm
import matplotlib.colors as c
import matplotlib.gridspec as gridspec

from scipy.integrate import odeint
from numba import vectorize

from typing import Union, List


plt.rcParams.update({'font.size': 12})

plt.rcParams.update({'xtick.major.size': 5, 'ytick.major.size': 5})
plt.rcParams.update({'xtick.minor.size': 2.5, 'ytick.minor.size': 2.5})
plt.rcParams.update({'xtick.major.width': 1, 'ytick.major.width': 1})
plt.rcParams.update({'xtick.minor.width': 0.8, 'ytick.minor.width': 0.8})

WL_LABEL = 'Wavelength / nm'
WN_LABEL = "Wavenumber / $10^{4}$ cm$^{-1}$"

LEGEND_FONT_SIZE = 10
MAJOR_TICK_DIRECTION = 'in'  # in, out or inout
MINOR_TICK_DIRECTION = 'in'  # in, out or inout

X_SIZE, Y_SIZE = 5, 4.5


def eps_label(factor):
    num = np.log10(1 / factor).astype(int)
    return f'$\\varepsilon$ / $(10^{{{num}}}$ L mol$^{{-1}}$ cm$^{{-1}})$'

def setup_wavenumber_axis(ax, x_label=WN_LABEL, x_major_locator=None, x_minor_locator=AutoMinorLocator(5), factor=1e3):
    secondary_ax = ax.secondary_xaxis('top', functions=(lambda x: factor / x, lambda x: 1 / (factor * x)))

    secondary_ax.tick_params(which='major', direction=MAJOR_TICK_DIRECTION)
    secondary_ax.tick_params(which='minor', direction=MINOR_TICK_DIRECTION)

    if x_major_locator:
        secondary_ax.xaxis.set_major_locator(x_major_locator)

    if x_minor_locator:
        secondary_ax.xaxis.set_minor_locator(x_minor_locator)

    secondary_ax.set_xlabel(x_label)

    return secondary_ax


def set_main_axis(ax, x_label=WL_LABEL, y_label="Absorbance", xlim=(None, None), ylim=(None, None),
                  x_major_locator=None, x_minor_locator=None, y_major_locator=None, y_minor_locator=None):
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    if xlim[0] is not None:
        ax.set_xlim(xlim)
    if ylim[0] is not None:
        ax.set_ylim(ylim)

    if x_major_locator:
        ax.xaxis.set_major_locator(x_major_locator)

    if x_minor_locator:
        ax.xaxis.set_minor_locator(x_minor_locator)

    if y_major_locator:
        ax.yaxis.set_major_locator(y_major_locator)

    if y_minor_locator:
        ax.yaxis.set_minor_locator(y_minor_locator)

    ax.tick_params(axis='both', which='major', direction=MAJOR_TICK_DIRECTION)
    ax.tick_params(axis='both', which='minor', direction=MINOR_TICK_DIRECTION)


def setup_twin_x_axis(ax, y_label="$I_{0,\\mathrm{m}}$ / $10^{-10}$ einstein s$^{-1}$ nm$^{-1}$", 
                      x_label=None, ylim=(None, None), y_major_locator=None, y_minor_locator=None,
                      keep_zero_aligned=True):
    ax2 = ax.twinx()
    
    ax2.tick_params(which='major', direction=MAJOR_TICK_DIRECTION)
    ax2.tick_params(which='minor', direction=MINOR_TICK_DIRECTION)
    
    if y_major_locator:
        ax2.yaxis.set_major_locator(y_major_locator)
        
    if y_minor_locator:
        ax2.yaxis.set_minor_locator(y_minor_locator)

    ax2.set_ylabel(y_label)
    
    if keep_zero_aligned and ylim[0] is None and ylim[1] is not None:
        # a = bx/(x-1)
        ax1_ylim = ax.get_ylim()
        x = -ax1_ylim[0] / (ax1_ylim[1] - ax1_ylim[0])  # position of zero in ax1, from 0, to 1
        a = ylim[1] * x / (x - 1)  # calculates the ylim[0] so that zero position is the same for both axes
        ax2.set_ylim(a, ylim[1])
    
    elif ylim[0] is not None:
        ax2.set_ylim(ylim)
        
    return ax2


def plot_results_fit_pair(wls: np.ndarray, ST_fit: np.ndarray, D_aug: np.ndarray, D_fit: np.ndarray, 
                        DZ: np.ndarray, DE: np.ndarray, timesZ: np.ndarray, timesE: np.ndarray, traces_at=(280, 415, 450)):

    # plot spectra and residuals
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2*X_SIZE * 1.1, Y_SIZE))
    
    f = 1e-4
    set_main_axis(ax1, y_label=eps_label(f), xlim=(230, 600),
                    x_minor_locator=AutoMinorLocator(5), y_minor_locator=AutoMinorLocator(2))
    
    set_main_axis(ax2, y_label='$\\leftarrow$ Time axis ', xlim=(230, 600),
                    x_minor_locator=AutoMinorLocator(5), y_minor_locator=AutoMinorLocator(2))
    
    ax1.plot(wls, ST_fit.T * f)
    ax1.legend(['Z', 'E', 'Photoox.\nproducts'], frameon=False)
    ax1.set_title('Estimated Spectra')
    
    D_res = D_aug - D_fit
    dummy_time = np.arange(0, D_aug.shape[0])
    _x, _y = np.meshgrid(wls, dummy_time)
    mappable = ax2.pcolormesh(_x, _y, D_res, cmap='seismic', shading='auto', vmin=np.nanmin(D_res), vmax=np.nanmax(D_res))
    fig.colorbar(mappable, ax=ax2, label="$\Delta A$", orientation='vertical')
    
    ax2.invert_yaxis()
    ax2.set_title('Residuals ($\\bf{D_{aug}} - \\bf{D_{fit}}$)')
    
    plt.tight_layout()
    plt.show()
    
    # plot conc. profiles
    
    fig = plt.figure(figsize=(2 * X_SIZE * 1.1, len(traces_at) * Y_SIZE))
    
    outer_grid = gridspec.GridSpec(len(traces_at), 2, wspace=0.25, hspace=0.4)
    
    for i, og in enumerate(outer_grid):
        tracewl = traces_at[i // 2]
        idx = fi(wls, tracewl)
        
        inner_grid = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=og, wspace=0.1, hspace=0.1,
                                                    height_ratios=(4, 1))
        ax_data = fig.add_subplot(inner_grid[0])
        ax_res = fig.add_subplot(inner_grid[1])
        
        set_main_axis(ax_data, x_label="", y_label='A')
        set_main_axis(ax_res, x_label="Time / s", y_label='res.')
        
            # plot zero lines
        ax_res.axline((0, 0), slope=0, ls='--', color='black', lw=0.5)

        ax_data.tick_params(labelbottom=False)
        
        t = timesZ if i % 2 == 0 else timesE
        trace_data = DZ[:, idx] if i % 2 == 0 else DE[:, idx]
        trace_fit = D_fit[:timesZ.shape[0], idx] if i % 2 == 0 else D_fit[timesZ.shape[0]:, idx]
        trace_res = D_res[:timesZ.shape[0], idx] if i % 2 == 0 else D_res[timesZ.shape[0]:, idx]
        title = 'Z experiment' if i % 2 == 0 else 'E experiment'
        ax_data.set_title(title)
        
        ax_data.plot(t, trace_data, lw=1, color='red', label=f'data {tracewl} nm')
        ax_data.plot(t, trace_fit, lw=1, color='black', label=f'fit {tracewl} nm')
        ax_res.plot(t, trace_res, lw=1, color='red')
        
        ax_data.yaxis.set_ticks_position('both')
        ax_data.xaxis.set_ticks_position('both')
        ax_data.legend(frameon=False)

        ax_res.yaxis.set_ticks_position('both')
        ax_res.xaxis.set_ticks_position('both')
    
    plt.show()


def plot_dependence(wavelengths:np.ndarray, Phi_ox_space:np.ndarray, spectra:np.ndarray, 
                    ssq:np.ndarray, Phis:np.ndarray):

    # plot the results
    Phi_ox_space_cut = Phi_ox_space[:15]
    ssq_cut  = ssq[:15]
    plt.scatter(Phi_ox_space, ssq, s=15, c='r')
    plt.scatter(Phi_ox_space_cut, ssq_cut, s=15, c='b')

    plt.xlabel('$\Phi_{{ox}}$')
    plt.ylabel('SSq $\\vert\\vert \\bf{R(\\Theta)} \\vert\\vert_2^2 $')
    plt.show()

    cmap = cm.get_cmap('jet', Phi_ox_space.shape[0])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(X_SIZE*1.1 * 3,  Y_SIZE))

    f = 1e-4

    for ax in [ax1, ax2, ax3]:
        set_main_axis(ax,  xlim=(230, 600), y_label=eps_label(f),
                        x_label='Wavelength / nm', 
                        y_minor_locator=AutoMinorLocator(2), x_minor_locator=None)
        
    for i in range(Phi_ox_space.shape[0]):
        for j, ax in enumerate([ax1, ax2, ax3]):
            ax.plot(wavelengths, spectra[i, j].T * f, color=cmap(i), label=f'$\Phi_{{ox}}$={Phi_ox_space[i]:.3f}' if i % 5==0 else '')

    ax3.legend(frameon=False)
    ax1.set_title('Spectra of $Z$')
    ax2.set_title('Spectra of $E$')
    ax3.set_title('Spectra of photoproducts')

    plt.tight_layout()
    plt.show()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(X_SIZE*1.1 * 3,  Y_SIZE))

    set_main_axis(ax1, x_label='$\\Phi_{ZE}$', y_label='$\\Phi_{EZ}$')
    set_main_axis(ax2, x_label='$\\Phi_{ZE}$', y_label='$\\Phi_{ox}$')
    set_main_axis(ax3, x_label='$\\Phi_{EZ}$', y_label='$\\Phi_{ox}$')

    ax1.scatter(Phis[:, 0], Phis[:, 1], s=15, c='r')
    ax2.scatter(Phis[:, 0], Phis[:, 2], s=15, c='r')
    ax3.scatter(Phis[:, 1], Phis[:, 2], s=15, c='r')

    ax1.scatter(Phis[:15, 0], Phis[:15, 1], s=15, c='b')
    ax2.scatter(Phis[:15, 0], Phis[:15, 2], s=15, c='b')
    ax3.scatter(Phis[:15, 1], Phis[:15, 2], s=15, c='b')

    plt.tight_layout()
    plt.show()

    print('Values of Phi_ox in the blue color')
    print(Phi_ox_space_cut)

def plot_matrix(times: np.ndarray, wavelengths: np.ndarray, D: np.ndarray, zlim=(None, None),
               title='', cmap='hot_r'):
    
    zmin, zmax = zlim[0] if zlim[0] is not None else np.nanmin(D), zlim[1] if zlim[1] is not None else np.nanmax(D)
    
    x, y = np.meshgrid(wavelengths, times)  # needed for pcolormesh to correctly scale the image
    
    plt.pcolormesh(x, y, D, cmap=cmap, shading='auto', vmin=zmin, vmax=zmax)
    plt.colorbar(label='$A$')
    plt.xlabel('Wavelength $\\rightarrow$')
    plt.ylabel('$\\leftarrow$ Time ')
    plt.gca().invert_yaxis()
#     plt.yscale('symlog', linthresh=1, linscale=1)
    plt.title(title)
#     plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    plt.show()

def plot_kinetics_ax(ax, matrix, times, wavelengths, n_spectra=50, linscale=1, linthresh=100, cmap='jet_r', add_wn_axis=True,
                  major_ticks_labels=(100, 1000), emph_t=(0, 200, 1000), colorbar_xy_loc=(0.75, 0.1), colorbar_height=0.7, colorbar_width=0.03,
                  colorbar_label='Time / s', lw=0.5, alpha=0.5, y_label='Absorbance', x_label=WL_LABEL, LED_source_xy=(None, None),
                  x_lim=(230, 600), sec_axis_ylabel='', sec_axis_y_major_locator=FixedLocator([]),  filepath=None, dpi=500, transparent=True):
    """
    Plots kinetics. Spectra are chosen based on symlog scale.
    """

    t = times
    w = wavelengths

    set_main_axis(ax, x_label=x_label, y_label=y_label, xlim=x_lim, x_minor_locator=None, y_minor_locator=None)
    if add_wn_axis:
        _ = setup_wavenumber_axis(ax)

    cmap = cm.get_cmap(cmap)
    norm = c.SymLogNorm(vmin=t[0], vmax=t[-1], linscale=linscale, linthresh=linthresh, base=10, clip=True)

    tsb_idxs = fi(t, emph_t)
    ts_real = np.round(t[tsb_idxs])

    x_space = np.linspace(0, 1, n_spectra, endpoint=True, dtype=np.float64)

    t_idx_space = fi(t, norm.inverse(x_space))
    t_idx_space = np.sort(np.asarray(list(set(t_idx_space).union(set(tsb_idxs)))))

    for i in t_idx_space:
        x_real = norm(t[i])
        x_real = 0 if np.ma.is_masked(x_real) else x_real
        ax.plot(w, matrix[i, :], color=cmap(x_real),
                 lw=1.5 if i in tsb_idxs else lw,
                 alpha=1 if i in tsb_idxs else alpha,
                 zorder=1 if i in tsb_idxs else 0)

    inset_loc = (colorbar_xy_loc[0], colorbar_xy_loc[1], colorbar_width, colorbar_height)
    cbaxes = ax.inset_axes(inset_loc)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbaxes, orientation='vertical',
                        format=ScalarFormatter(),
                        label=colorbar_label)

    cbaxes.invert_yaxis()

    minor_ticks = [10, 20, 30, 40, 50, 60, 70, 80, 90, 200, 300, 400, 500, 600, 700, 800, 900] + list(
        np.arange(2e3, t[-1], 1e3))
    cbaxes.yaxis.set_ticks(cbar._locate(minor_ticks), minor=True)

    major_ticks = np.sort(np.hstack((np.asarray([100, 1000]), ts_real)))
    major_ticks_labels = np.sort(np.hstack((np.asarray(major_ticks_labels), ts_real)))

    cbaxes.yaxis.set_ticks(cbar._locate(major_ticks), minor=False)
    cbaxes.set_yticklabels([(f'{num:0.0f}' if num in major_ticks_labels else "") for num in major_ticks])

    for ytick, ytick_label, _t in zip(cbaxes.yaxis.get_major_ticks(), cbaxes.get_yticklabels(), major_ticks):
        if _t in ts_real:
            color = cmap(norm(_t))
            ytick_label.set_color(color)
            ytick_label.set_fontweight('bold')
            ytick.tick2line.set_color(color)
            ytick.tick2line.set_markersize(5)
            # ytick.tick2line.set_markeredgewidth(2)

    if LED_source_xy[0] is not None and LED_source_xy[1] is not None:
        x_LED, y_LED = LED_source_xy
        ax_sec = setup_twin_x_axis(ax, ylim=(None, y_LED.max() * 3), y_label=sec_axis_ylabel, y_major_locator=sec_axis_y_major_locator)
        ax_sec.fill(x_LED, y_LED, facecolor='gray', alpha=0.5)
        ax_sec.plot(x_LED, y_LED, color='black', ls='dotted', lw=1)


def fi(array: np.ndarray, value : Union[List, int, float]) -> Union[List[int], int]:
    """Finds nearest index(es) in the array that corresponds to nearest value(s) in array."""

    if isinstance(value, (int, float)):
        value = np.asarray([value])
    else:
        value = np.asarray(value)

    result = np.empty_like(value, dtype=int)
    for i in range(value.shape[0]):
        idx = np.searchsorted(array, value[i], side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value[i] - array[idx - 1]) < math.fabs(value[i] - array[idx])):
            result[i] = idx - 1
        else:
            result[i] = idx
    return result if result.shape[0] > 1 else result[0]

@vectorize(nopython=True, fastmath=False)
def photokin_factor(A: np.ndarray) -> np.ndarray:
    """Calculates the photokinetic factor from absorbance."""

    ln10 = np.log(10)
    ll2 = ln10 ** 2 / 2  # 0.5 * (ln 10)^2

    if A < 1e-4:  # photokinetics factor cannot be calculated at A == 0 due to zero division
        return ln10 - A * ll2  # approximation with first two taylor series terms
    else:
        return (1 - np.exp(-A * ln10)) / A  # exact photokinetic factor

def simul_photokin_model(I0_m: np.ndarray, c0: np.ndarray, K: np.ndarray, eps: np.ndarray, times: np.ndarray, V: float = 0.003, l: float = 1, 
                         R: float = 0.036, use_backref_correction: bool = True, scaling_coef: float = 1e6, dx: float = 1) -> np.ndarray:
    """
    Simulates the photokinetic model with no wavelength-dependence of the quantum yields.

    Parameters
    ----------
    I0_m : np.ndarray
        Incident spectral photon flux. In this particular context, this was measured after the cuvette with pure solvent, so
        it represents the I_solvent, and it means the flux is diminished by two reflections. Dimension (n_w,)
    c0 : np.ndarray
        Vector of initial concentrations. Dimension (k,).
    K : np.ndarray
        Trasfer matrix that describes the model. Dimension are (k, k).
    eps : np.ndarray
        Matrix of molar absorption coefficients. The dimensions are (k, n_w).
    times : np.ndarray
        Time points the model will be simulated on. Dimension (n_t,).
    V : float
        Volume of sample in the cuvette.
    l : float
        Length of light traveled in the cuvette.
    R : float
        Reflectivity coefficient of quartz-air interface, default value 0.036.
    use_backref_correction : bool
        If True (default) the backreflection correction will be used.
    scaling_coef : float
        Coefficient that will scale incident spectral flux and intial concetration and inversly scale the eps matrix.
        Scaling is there to keep the numerical integrator stable, if the simulated profiles has low values, it can cause problems
        numerical error in the integrator and in the final solution. The chose of the scaling coefficient does not change the overall solution
        but tries to minimize the integration errors.
    dx : float
        Spacing in the spectral data. Default 1 nm. If the non-equally spaced data are used, the function needs to be modified.

    Returns
    -------
    C : np.ndarray
        Matrix of simulated model in dimensions of (n_t, k).
    """

    c0 = np.asarray(c0)

    assert c0.shape[0] == K.shape[0]
    ln10 = np.log(10)

    eps_s = eps / scaling_coef
    I0_s = I0_m * scaling_coef
    c0_s = c0 * scaling_coef

    def dc_dt(c, t):
        """
        Calculates the matrix differential equation for the general photokinetic model with
        no wavelenth-dependence of the quantum yields.

        Parameters
        ----------
        c : np.ndarray
            Current concentration vector.
        t : float
            Current time.
        """
        
        A = l * c.dot(eps_s) # calculate the total absorbance
        
        F = photokin_factor(A) # calculate the photokinetic factor, F = (1-10^-A) / A

        # calculate effective spectral flux that hits the sample
        # I0 was measured after cuvette with pure solvent
        Ieff = I0_s * (1 + R * np.exp(-A * ln10)) / (1 - R) if use_backref_correction else I0_s / (1 - R)
        
        # calculate the integrals and multiply them with transfer matrix
        # integrate with the trapezoidal rule, we know that there is a uniform spacing
        product = K * np.trapz(Ieff * F * eps_s, dx=dx, axis=-1)  # K @ diag(int(I0 * F * eps))

        return l * product.dot(c) / V  # l/V * final dot product with c vector

    result = odeint(dc_dt, c0_s, times)  # numerically integrate

    return result / scaling_coef


def load_matrix(filepath: str, delimiter: str = '\t') -> tuple:
    """
    Loads the dataset.

    Parameters
    ----------
    filepath : str
        Path to file.
    delimiter : str
        Delimiter that separates the columns in the text files.

    Returns
    -------
    wavelengths : np.ndarray
        Array of wavelengths.
    times : np.ndarray
        Array of time points.
    D : np.ndarray
        Data matrix.
    """

    _data = np.genfromtxt(filepath, delimiter=delimiter, filling_values=0)
    D = _data[1:, 1:]
    wavelengths = _data[1:, 0]
    times = _data[0, 1:]
    return wavelengths, times, D

def load_LED_current(filepath: str) -> float:
    """Loads the electric current from file and returns it as float."""
    with open(filepath) as f:
        return float(f.readline())
    

def get_valerr(value: float, uncertainty: float, un_sig_values: int = 1) -> str:
    """
    Formates the value with uncertainty in the Latex notation. The value will be rounded to the same number of decimal places
    as the uncertainty has. Number of significant figures of uncertainty is denoted in parameter un_sig_values.

    Parameters
    ----------
    value : float
    uncertainty : float
    un_sig_values : int
        Number of significant figures of the uncertainty.
    """
    un = float(f'{uncertainty:.{un_sig_values}g}')  # round uncertainty to defined number of significant figures
    
    exp_un = int(np.floor(np.log10(un))) - un_sig_values  # exponent of last digit for uncertainty
    
    exp_val = int(np.floor(np.log10(value)))  # exponent for uncertainty
    n_sig_fig_val = exp_val - exp_un

    # https://docs.python.org/3/library/string.html#format-string-syntax
    # # option does not remove the trailing zeros from the output
    return f'{value:#.{n_sig_fig_val}g} $\pm$ {uncertainty:#.{un_sig_values}g}' 
