"""Sensorless adaptive optics.

TO DO:
- Load interaction matrix from disk
- Set and get Zernike mode amplitudes from mirror
- Might need HASO functions to do this, since Zernike need to be composed given the pupil

2024/12 DPS initial work
"""
from opm_v2.hardware.AOMirror import AOMirror
from opm_v2.hardware.OPMNIDAQ import OPMNIDAQ
from pymmcore_plus import CMMCorePlus
import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Tuple, Sequence, List
from scipy.fftpack import dct
from scipy.ndimage import center_of_mass
from scipy.optimize import curve_fit
from pathlib import Path
from tifffile import imwrite
    
def run_ao_optimization(
    image_mirror_step_size_um: float,
    image_mirror_sweep_um: float,
    exposure_ms: float,
    channel_states: List[bool],
    metric_to_use: Optional[str] = "shannon_dct",
    shannon_psf_radius_px: Optional[float] = 2,
    num_iterations: Optional[int] = 3,
    num_mode_steps: Optional[int] = 3,
    init_delta_range: Optional[float] = 0.300,
    delta_range_alpha_per_iter: Optional[float] = 0.5,
    modes_to_optimize: Optional[List[int]] = [7,14,23,3,4,5,6,8,9,10,11,12,13,15,16,17,18,19,20,21,22,24,25,26,27,28,29,30,31],
    roi_crop_size: Optional[int] = 101,
    save_results: Optional[bool] = False,
    verbose: Optional[bool] = True,
    ):
    
    #---------------------------------------------#
    # Create hardware controller instances
    #---------------------------------------------#
    opmNIDAQ_local = OPMNIDAQ.instance()
    aoMirror_local = AOMirror.instance()
    mmc = CMMCorePlus.instance()
    
    #---------------------------------------------#
    # setup the daq waveforms to run in projection mode
    #---------------------------------------------#
    opmNIDAQ_local.stop_waveform_playback()
    opmNIDAQ_local.clear_tasks()
    opmNIDAQ_local.set_acquisition_params(
        scan_type="projection",
        channel_states=channel_states,
        image_mirror_step_size_um=image_mirror_step_size_um,
        image_mirror_sweep_um=image_mirror_sweep_um,
        laser_blanking=True,
        exposure_ms=exposure_ms
    )
    opmNIDAQ_local.generate_waveforms()
    opmNIDAQ_local.prepare_waveform_playback()
    opmNIDAQ_local.start_waveform_playback()
    
    mmc.setProperty("OrcaFusionBT", "Exposure",float(exposure_ms))
    mmc.waitForDevice("OrcaFusionBT")
    
    #---------------------------------------------#
    # Setup Zernike modal coeff arrays
    #---------------------------------------------#
    initial_zern_modes = aoMirror_local.current_coeffs.copy() # coeff before optmization
    init_iter_zern_modes = initial_zern_modes.copy() # Mirror coeffs at the start of optimizing a mode 
    active_zern_modes = initial_zern_modes.copy() # modified coeffs to be or are applied to mirror
    optimized_zern_modes = initial_zern_modes.copy() # Mode coeffs after running all iterations

    #----------------------------------------------
    # TODO
    save_path = None
    save_results = True
    display_images = True
    if save_results:
        optimal_metrics = []
        optimal_coefficients = []
        mode_images = []
        iteration_images = []
    #----------------------------------------------
    
    
    #---------------------------------------------#
    # Start AO optimization
    #---------------------------------------------#   
    if verbose:
        print(f"Starting A.O. optimization using {metric_to_use} metric")
    
    # Snap an image and calculate the starting metric.
    mmc.snapImage()
    starting_image = mmc.getImage()
    imwrite(Path(r"g:/ao/ao_start.tiff"),starting_image)

    starting_metric = metric_shannon_dct(
        image=starting_image,
        shannon_psf_radius_px=shannon_psf_radius_px,
        crop_size=None
        )

    # initialize delta range
    delta_range=init_delta_range
    
    # Start AO iterations 
    for k in range(num_iterations): 
        if k==0:       
            # initialize the optimal metric, only gets updated when a better metric is obtained.
            optimal_metric = starting_metric 
            if save_results:
                iteration_images.append(starting_image)
        
        # Iterate over modes to optimize
        for mode in modes_to_optimize:
            if verbose:
                print(f"AO iteration: {k+1} / {num_iterations}")
                print(f"  Perturbing mirror mode: {mode+1} / {modes_to_optimize[-1]+1}")
                
            # Grab the current starting mode coeff for this iteration
            init_iter_zern_modes = aoMirror_local.current_coeffs.copy()
            deltas = np.linspace(-delta_range, delta_range, num_mode_steps)
            
            metrics = []
            for delta in deltas:
                # Create an array to modify the mode coeff. and write to the mirror
                active_zern_modes = init_iter_zern_modes.copy()
                active_zern_modes[mode] += delta
                
                # Write zernike modes to the mirror
                success = aoMirror_local.set_modal_coefficients(active_zern_modes)
                
                if not(success):
                    print("    Setting mirror coefficients failed!")
                    # Force metric and image to zero
                    metric = 0
                    image = np.zeros_like(starting_image)
                    
                    if display_images:
                        pass
                    if save_results:
                        mode_images.append(image)
                        
                else:
                    """acquire projection image"""
                    mmc.snapImage()
                    image = mmc.getImage()
                    imwrite(Path(f"g:/ao/ao_{mode}_{delta}.tiff"),image)
                        
                    """Calculate metric."""
                    metric = metric_shannon_dct(
                        image=image,
                        shannon_psf_radius_px=shannon_psf_radius_px,
                        crop_size=None
                        )
                    
                    if metric==np.nan:
                        print("Metric is NAN, setting to 0")
                        metric = float(np.nan_to_num(metric))
                    if verbose:
                        print(f"      Metric = {metric:.4f}")
                    
                metrics.append(metric)
            
            """After looping through all mirror pertubations for this mode, decide if mirror is updated"""

            #---------------------------------------------#
            # Fit metrics to determine optimal metric
            #---------------------------------------------#   
            if 0 in metrics:
                optimal_delta = 0
            else:
                try:
                    popt = quadratic_fit(deltas, metrics)
                    a, b, c = popt
                    
                    # Test if metric samples have a peak to fit, reject if not.
                    is_increasing = all(x < y for x, y in zip(np.asarray(metrics), np.asarray(metrics)[1:]))
                    is_decreasing = all(x > y for x, y in zip(np.asarray(metrics), np.asarray(metrics)[1:]))
                    if is_increasing or is_decreasing:
                        print("      Test metrics are monotonic and linear, fit rejected. ")
                        raise Exception
                    elif a >=0:
                        print("      Test metrics have a positive curvature, fit rejected.")
                        raise Exception
                    
                    # Optimal metric is at the peak of quadratic 
                    optimal_delta = -b / (2 * a)
                    if verbose:
                        print(f"    Quadratic fit result for optimal delta: {optimal_delta:.4f}")
                        
                    # Reject metric if it is outside the test range.
                    if (optimal_delta>delta_range) or (optimal_delta<-delta_range):
                        print(f"      Optimal delta is outside of delta_range: {-b / (2 * a):.3f}")
                        raise Exception
                            
                except Exception:
                    optimal_delta = 0
                    if verbose:
                        print(f"        Exception in fit occurred, optimal delta = {optimal_delta:.4f}")

            
            #---------------------------------------------#
            # Test the new optimal mode coeff. to verify the metric improves
            #---------------------------------------------#   
            coeff_opt = init_iter_zern_modes[mode] + optimal_delta
            active_zern_modes[mode] = coeff_opt

            # verify mirror successfully loads requested state
            success = aoMirror_local.set_modal_coefficients(active_zern_modes)
            if not(success):
                if verbose:
                    print("    Setting mirror positions failed, using current mode coefficient.")
                coeff_to_keep = init_iter_zern_modes[mode]
            else:
                """acquire projection image"""
                mmc.snapImage()
                image = mmc.getImage()
                    
                """Calculate metric."""
                metric = metric_shannon_dct(
                    image=image,
                    shannon_psf_radius_px=shannon_psf_radius_px,
                    crop_size=None
                    )
                    
                if metric==np.nan:
                    print("    Metric is NAN, setting to 0")
                    metric = float(np.nan_to_num(metric))
                
                if metric>=optimal_metric:
                    coeff_to_keep = coeff_opt
                    optimal_metric = metric
                    if verbose:
                        print(f"      Updating mirror with new optmimal mode coeff.: {coeff_to_keep:.4f} with metric: {metric:.4f}")
                else:
                    # if not keep the current mode coeff
                    if verbose:
                        print(
                            "    Metric not improved using previous iteration's mode coeff.",
                            f"\n     optimal metric: {optimal_metric:.6f}",
                            f"\n     rejected metric: {metric:.6f}"
                            )
                    coeff_to_keep = init_iter_zern_modes[mode]
            
            if save_results:
                optimal_metrics.append(optimal_metric)
                
            #---------------------------------------------#
            # Apply the kept optimized mirror modal coeffs
            #---------------------------------------------# 
            active_zern_modes[mode] = coeff_to_keep
            _ = aoMirror_local.set_modal_coefficients(active_zern_modes)
            """Loop back to top and do the next mode until all modes are done"""
            
        #---------------------------------------------#
        # After all modes, reduce the delta range for the next iteration
        #---------------------------------------------# 
        delta_range *= delta_range_alpha_per_iter
        if verbose:
            print(
                f"  Reduced sweep range to {delta_range:.4f}",
                f"  Current metric: {metric:.4f}"
                )
        
        if save_results: 
            optimal_coefficients.append(aoMirror_local.current_coeffs.copy())
            iteration_images.append(image)
            
        """Loop back to top and do the next iteration"""
    
    #---------------------------------------------#
    # After all the iterations the mirror state will be optimized
    #---------------------------------------------# 
    optimized_zern_modes = aoMirror_local.current_coeffs.copy()          
    if verbose:
        print(
            f"Starting Zernike mode amplitude:\n{initial_zern_modes}",
            f"\nFinal optimized Zernike mode amplitude:\n{optimized_zern_modes}"
            )
    
    # apply optimized Zernike mode coefficients to the mirror
    _ = aoMirror_local.set_modal_coefficients(optimized_zern_modes)
    aoMirror_local.save_mirror_positions(name="opm_current_flat")
    opmNIDAQ_local.stop_waveform_playback()
         


#-------------------------------------------------#
# Plotting functions
#-------------------------------------------------#

def plot_zernike_coeffs(optimal_coefficients: ArrayLike,
                        zernike_mode_names: ArrayLike,
                        save_path: Optional[Path] = None,
                        show_fig: Optional[bool] = False):
    """_summary_

    Parameters
    ----------
    optimal_coefficients : ArrayLike
        _description_
    save_path : Path
        _description_
    showfig : bool
        _description_
    """
    import matplotlib.pyplot as plt
    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 8))
    
    # Define colors and markers for each iteration
    colors = ['b', 'g', 'r', 'c', 'm']  
    markers = ['x', 'o', '^', 's', '*']  

    # populate plots
    for i in range(len(zernike_mode_names)):
        for j in range(optimal_coefficients.shape[0]):
            marker_style = markers[j % len(markers)]
            ax.scatter(optimal_coefficients[j, i], i, 
                       color=colors[j % len(colors)], s=125, marker=marker_style)  
        ax.axhline(y=i, linestyle="--", linewidth=1, color='k')
        
    # Plot a vertical line at 0 for reference
    ax.axvline(0, color='k', linestyle='-', linewidth=1)

    # Customize the plot
    ax.set_yticks(np.arange(len(zernike_mode_names)))
    ax.set_yticklabels(zernike_mode_names)
    ax.set_xlabel("Coefficient Value")
    ax.set_title("Zernike mode coefficients at each iteration")
    ax.set_xlim(-0.15, 0.15)

    # Add a legend for time points
    ax.legend([f'Iteration: {i+1}' for i in range(optimal_coefficients.shape[0])], loc='upper right')

    # Remove grid lines
    ax.grid(False)

    plt.tight_layout()
    if show_fig:
        plt.show()
    if save_path:
        fig.savefig(save_path)


def plot_metric_progress(optimal_metrics: ArrayLike,
                         modes_to_optimize: List[int],
                         zernike_mode_names: List[str],
                         save_path: Optional[Path] = None,
                         show_fig: Optional[bool] = False):
    """_summary_

    Parameters
    ----------
    metrics : ArrayLike
        _description_
    modes_to_optmize : List[int]
        _description_
    zernike_mode_names : List[str]
        _description_
    save_path : Optional[Path], optional
        _description_, by default None
    show_fig : Optional[bool], optional
        _description_, by default False
    """
    import matplotlib.pyplot as plt
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define colors and markers for each iteration
    colors = ['b', 'g', 'r', 'c', 'm']
    markers = ['x', 'o', '^', 's', '*']

    # Loop over iterations and plot each series
    for ii, series in enumerate(optimal_metrics):
        ax.plot(series, color=colors[ii], label=f"iteration {ii}", marker=markers[ii], linestyle="--", linewidth=1)

    # Set the x-axis to correspond to the modes_to_optimize
    mode_labels = [zernike_mode_names[i] for i in modes_to_optimize]
    ax.set_xticks(np.arange(len(mode_labels))) 
    ax.set_xticklabels(mode_labels, rotation=60, ha="right", fontsize=16) 

    # Customize the plot
    ax.set_ylabel("Metric", fontsize=16)
    ax.set_title("Optimal Metric Progress per Iteration", fontsize=18)

    ax.legend(fontsize=15)
    
    plt.tight_layout()
    
    if show_fig:
        plt.show()
    if save_path:
        fig.savefig(save_path)


def plot_2d_localization_fit_summary(
    fit_results,
    img,
    coords_2d,
    save_path: Path = None,
    showfig: bool = False
    ):
    """_summary_

    Parameters
    ----------
    fit_results : _type_
        _description_
    img : _type_
        _description_
    coords_2d : _type_
        _description_
    save_path : Path, optional
        _description_, by default None
    showfig : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    # imports
    from localize_psf.fit_psf import sxy2na
    from localize_psf.localize import plot_bead_locations
    import matplotlib.pyplot as plt
    
    to_keep = fit_results["to_keep"]
    sxy = fit_results["fit_params"][to_keep, 4]
    amp = fit_results["fit_params"][to_keep, 0]
    bg = fit_results["fit_params"][to_keep, 6]
    centers = fit_results["fit_params"][to_keep][:, (3, 2, 1)]
    cx = centers[:,2]
    cy = centers[:,1]

    width_ratios=[1,0.7,0.1,0.1,0.1]
    height_ratios=[1,0.1,0.5,0.5,0.5,0.5,0.5]
    figh_sum = plt.figure(figsize=(10,8))
    grid_sum = figh_sum.add_gridspec(nrows=len(height_ratios),
                                     ncols=len(width_ratios),
                                     width_ratios=width_ratios,
                                     height_ratios=height_ratios,
                                     hspace=0.2,
                                     wspace=0.3
                                     )

    ax_proj_sxy = figh_sum.add_subplot(grid_sum[0,:2])
    ax_cmap_i_sxy = figh_sum.add_subplot(grid_sum[0,2])
    ax_cmap_sxy = figh_sum.add_subplot(grid_sum[0,4])
    figh_sum = plot_bead_locations(img,
                                    centers,
                                    weights=[fit_results["fit_params"][to_keep, 4]],
                                    color_lists=["autumn"],
                                    color_limits=[[0.05,0.5]],
                                    cbar_labels=[r"$\sigma_{xy}$"],
                                    title="Max intensity projection with Sxy",
                                    coords=coords_2d,
                                    gamma=0.5,
                                    axes=[ax_proj_sxy, ax_cmap_i_sxy, ax_cmap_sxy]
                                    )
    ax_proj_sxy.set_title(f"Sxy: mean={np.mean(sxy):.3f}, median={np.median(sxy):.3f}; NA (median):{sxy2na(0.473, np.median(sxy)):.2f}")

    # Create axes for plotting x, y specific results
    ax_sxy_cx = figh_sum.add_subplot(grid_sum[3,0])
    ax_sxy_cy = figh_sum.add_subplot(grid_sum[3,1:],
                                     sharey=ax_sxy_cx)
    ax_amp_cx = figh_sum.add_subplot(grid_sum[4,0],sharex=ax_sxy_cx)
    ax_amp_cy = figh_sum.add_subplot(grid_sum[4,1:],
                                     sharey=ax_amp_cx,sharex=ax_sxy_cy)
    ax_bg_cx = figh_sum.add_subplot(grid_sum[5,0],sharex=ax_sxy_cx)
    ax_bg_cy = figh_sum.add_subplot(grid_sum[5,1:],
                                    sharey=ax_bg_cx,sharex=ax_sxy_cy)
    ax_sxy_cx.set_ylabel(r"$\sigma_{xy}$ ($\mu m$)")
    ax_amp_cx.set_ylabel("amplitude")
    ax_bg_cx.set_ylabel("background")
    ax_bg_cx.set_xlabel(r"$C_x$ $\mu m$")
    ax_bg_cy.set_xlabel(r"$C_y$ $\mu m$")
    for ax in [ax_sxy_cy,ax_amp_cy,ax_bg_cy]:
        ax.tick_params(labelleft=False)
    for ax in [ax_sxy_cx,ax_sxy_cy,ax_amp_cx,ax_amp_cy]:
        ax.tick_params(labelbottom=False)

    # Set limits for visualizing sz
    if max(amp)>65000:
        amp_max = 15000
    else:
        amp_max = np.max(amp)*1.1
    ax_sxy_cx.set_ylim(0,1.0)
    ax_sxy_cy.set_ylim(0,1.0)
    ax_amp_cx.set_ylim(0, amp_max)
    ax_amp_cy.set_ylim(0, amp_max)
    ax_bg_cx.set_ylim(0, amp_max)
    ax_bg_cy.set_ylim(0, amp_max)
    ax_sxy_cx.set_xlim(0,img.shape[1]*0.115)
    ax_sxy_cy.set_xlim(0,img.shape[0]*0.115)
    ax_amp_cx.set_xlim(0,img.shape[1]*0.115)
    ax_amp_cy.set_xlim(0,img.shape[0]*0.115)
    ax_bg_cx.set_xlim(0,img.shape[1]*0.115)
    ax_bg_cy.set_xlim(0,img.shape[0]*0.115)
    # Plot directional results
    ax_sxy_cx.plot(cx, sxy, c="b", marker=".", markersize=3, linestyle="none")
    ax_sxy_cy.plot(cy, sxy, c="b", marker=".", markersize=3, linestyle="none")
    ax_amp_cx.plot(cx, amp, c="b", marker=".", markersize=3, linestyle="none")
    ax_amp_cy.plot(cy, amp, c="b", marker=".", markersize=3, linestyle="none")
    ax_bg_cx.plot(cx, bg, c="b", marker=".", markersize=3, linestyle="none")
    ax_bg_cy.plot(cy, bg, c="b", marker=".", markersize=3, linestyle="none")

    if showfig:
        figh_sum.show()
        plt.show()
    else:
        plt.close(figh_sum)
    if save_path:
        figh_sum.savefig(save_path, dpi=150)
    
    figh_sum = None
    del figh_sum
    return None


#-------------------------------------------------#
# Functions for preparing data
#-------------------------------------------------#

def get_image_center(image: ArrayLike, threshold: float) -> Tuple[int, int]:
    """
    Calculate the center of an image using a thresh-holded binary mask.

    Parameters
    ----------
    image : ArrayLike
        2D image array.
    threshold : float
        Intensity threshold for binarization.

    Returns
    -------
    center : Tuple[int, int]
        Estimated center coordinates (x, y).
    """
    try:
        binary_image = image > threshold
        center = center_of_mass(binary_image)
        center = tuple(map(int, center))
    except Exception:
        center = (image.shape[1]//2, image.shape[0]//2)
    return center


def get_cropped_image(image: ArrayLike, crop_size: int, center: Tuple[int, int]) -> ArrayLike:
    """
    Extract a square region from an image centered at a given point.

    Parameters
    ----------
    image : ArrayLike
        Input 2D or 3D image.
    crop_size : int
        Half-width of the cropping region.
    center : Tuple[int, int]
        Center coordinates (x, y) of the crop.

    Returns
    -------
    cropped_image : ArrayLike
        Cropped region from the input image.
    """
    if len(image.shape) == 3:
        x_min, x_max = max(center[0] - crop_size, 0), min(center[0] + crop_size, image.shape[1])
        y_min, y_max = max(center[1] - crop_size, 0), min(center[1] + crop_size, image.shape[2])
        cropped_image = image[:, x_min:x_max, y_min:y_max]
    else:
        x_min, x_max = max(center[0] - crop_size, 0), min(center[0] + crop_size, image.shape[0])
        y_min, y_max = max(center[1] - crop_size, 0), min(center[1] + crop_size, image.shape[1])
        cropped_image = image[x_min:x_max, y_min:y_max]
    return cropped_image


#-------------------------------------------------#
# Functions for fitting and calculations
#-------------------------------------------------#

def gauss2d(coords_xy: ArrayLike, amplitude: float, center_x: float, center_y: float,
            sigma_x: float, sigma_y: float, offset: float) -> ArrayLike:
    """
    Generates a 2D Gaussian function for curve fitting.

    Parameters
    ----------
    coords_xy : ArrayLike
        Meshgrid coordinates (x, y).
    amplitude : float
        Peak intensity of the Gaussian.
    center_x : float
        X-coordinate of the Gaussian center.
    center_y : float
        Y-coordinate of the Gaussian center.
    sigma_x : float
        Standard deviation along the x-axis.
    sigma_y : float
        Standard deviation along the y-axis.
    offset : float
        Background offset intensity.

    Returns
    -------
    raveled_gauss2d : ArrayLike
        Flattened 2D Gaussian function values.
    """
    x, y = coords_xy
    raveled_gauss2d = (
        offset +
        amplitude * np.exp(
            -(((x - center_x)**2 / (2 * sigma_x**2)) + ((y - center_y)**2 / (2 * sigma_y**2)))
        )
    ).ravel()

    return raveled_gauss2d


def otf_radius(img: ArrayLike, shannon_psf_radius_px: float) -> int:
    """
    Computes the optical transfer function (OTF) cutoff frequency.

    Parameters
    ----------
    img : ArrayLike
        2D image.
    shannon_psf_radius_px : float
        Estimated point spread function (PSF) radius in pixels.

    Returns
    -------
    cutoff : int
        OTF cutoff frequency in pixels.
    """
    w = min(img.shape)
    shannon_psf_radius_px = max(1, np.ceil(shannon_psf_radius_px))  # clip all PSF radii below 1 px to 1.
    cutoff = np.ceil(w / (2 * shannon_psf_radius_px)).astype(int)

    return cutoff


def normL2(x: ArrayLike) -> float:
    """
    Computes the L2 norm of an n-dimensional array.

    Parameters
    ----------
    x : ArrayLike
        Input array.

    Returns
    -------
    l2norm : float
        L2 norm of the array.
    """
    l2norm = np.sqrt(np.sum(x.flatten() ** 2))

    return l2norm


def shannon(spectrum_2d: ArrayLike, otf_radius: int = 100) -> float:
    """
    Computes the Shannon entropy of an image spectrum within a given OTF radius.

    Parameters
    ----------
    spectrum_2d : ArrayLike
        2D spectrum of an image (e.g., from DCT or FFT).
    otf_radius : int, optional
        OTF support radius in pixels (default is 100).

    Returns
    -------
    entropy : float
        Shannon entropy of the spectrum.
    """
    h, w = spectrum_2d.shape
    y, x = np.ogrid[:h, :w]

    # Circular mask centered at (0,0) for DCT
    support = (x**2 + y**2) < otf_radius**2

    spectrum_values = np.abs(spectrum_2d[support])
    total_energy = np.sum(spectrum_values)

    if total_energy == 0:
        return 0  # Avoid division by zero

    probabilities = spectrum_values / total_energy
    entropy = -np.sum(probabilities * np.log2(probabilities, where=(probabilities > 0)))
    metric = np.log10(entropy)
    return metric


def dct_2d(image: ArrayLike, cutoff: int = 100) -> ArrayLike:
    """
    Computes the 2D discrete cosine transform (DCT) of an image with a cutoff.

    Parameters
    ----------
    image : ArrayLike
        2D image array.
    cutoff : int, optional
        OTF radius cutoff in pixels (default is 100).

    Returns
    -------
    dct_2d : ArrayLike
        Transformed image using DCT.
    """
    dct_2d = dct(dct(image.astype(np.float32), axis=0, norm='ortho'), axis=1, norm='ortho')

    return dct_2d


def quadratic(x: float, a: float, b: float, c: float) -> ArrayLike:
    """
    Quadratic function evaluation at x.

    Parameters
    ----------
    x : float
        Point to evaluate.
    a : float
        x^2 coefficient.
    b : float
        x coefficient.
    c : float
        Offset.

    Returns
    -------
    value : float
        a * x^2 + b * x + c
    """
    return a * x**2 + b * x + c


def quadratic_fit(x: ArrayLike, y: ArrayLike) -> Sequence[float]:
    """
    Quadratic function for curve fitting.

    Parameters
    ----------
    x : ArrayLike
        1D x-axis data.
    y : ArrayLike
        1D y-axis data.

    Returns
    -------
    coeffs : Sequence[float]
        Fitting parameters.
    """
    A = np.vstack([x**2, x, np.ones_like(x)]).T
    coeffs = np.linalg.lstsq(A, y, rcond=None)[0]

    return coeffs


#-------------------------------------------------#
# Localization methods to generate ROIs for fitting
#-------------------------------------------------#

def localize_2d_img(
    img, 
    dxy,
    localize_psf_filters = {
        "threshold":3000,
        "amp_bounds":(1000, 30000),
        "sxy_bounds":(0.100, 1.0)
        },
    save_dir_path: Path = None,
    label: str = "", 
    showfig: bool = False,
    verbose: bool = False):
    """_summary_

    Parameters
    ----------
    img : _type_
        _description_
    dxy : _type_
        _description_
    localize_psf_filters : dict, optional
        _description_, by default { "threshold":3000, "amp_bounds":(1000, 30000), "sxy_bounds":(0.100, 1.0) }
    save_dir_path : Path, optional
        _description_, by default None
    label : str, optional
        _description_, by default ""
    showfig : bool, optional
        _description_, by default False
    verbose : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    from localize_psf.fit_psf import gaussian3d_psf_model
    from localize_psf.localize import (
        localize_beads_generic,
        get_param_filter,
        get_coords
        )
    
    # Define fitting model and coordinates
    model = gaussian3d_psf_model() 
    coords_3d = get_coords((1,)+img.shape, (1, dxy, dxy))
    coords_2d = get_coords(img.shape, (dxy, dxy))
                           
    # Set fit bounds and parameter filters
    threshold = localize_psf_filters["threshold"]
    amp_bounds = localize_psf_filters["amp_bounds"]
    sxy_bounds = localize_psf_filters["sxy_bounds"]
    fit_dist_max_err = (0, dxy*2) 
    fit_roi_size = (1, dxy*9, dxy*9)
    min_spot_sep = (0, dxy*5)
    dist_boundary_min = (0, 1.0)
        
    param_filter = get_param_filter(
        coords_3d,
        fit_dist_max_err=fit_dist_max_err,
        min_spot_sep=min_spot_sep,
        amp_bounds=amp_bounds,
        dist_boundary_min=dist_boundary_min,
        sigma_bounds=((0,sxy_bounds[0]),(1,sxy_bounds[1]))
        )
        
    # Run localization function
    _, r, _ = localize_beads_generic(
        img,
        (1, dxy, dxy),
        threshold=threshold,
        roi_size=fit_roi_size,
        filter_sigma_small=None,
        filter_sigma_large=None,
        min_spot_sep=min_spot_sep,
        model=model,
        filter=param_filter,
        max_nfit_iterations=100,
        use_gpu_fit=False,
        use_gpu_filter=False,
        return_filtered_images=False,
        fit_filtered_images=False,
        verbose=verbose
        )
    
    if save_dir_path:
        plot_2d_localization_fit_summary(
            r, 
            img,
            coords_2d, 
            save_dir_path / Path(f"localize_psf_summary_{label}.png"),
            showfig
            )
        
    return r

#-------------------------------------------------#
# Functions to calculate image metrics
#-------------------------------------------------#

def metric_brightness(image: ArrayLike,
                      crop_size: Optional[int] = None,
                      threshold: Optional[float] = 100,
                      image_center: Optional[int] = None,
                      return_image: Optional[bool] = False
                      ) -> float:
    """
    Compute weighted metric for 2D Gaussian.

    Parameters
    ----------
    image : ArrayLike
        2D image.
    threshold : float, optional
        Initial threshold to find spot (default is 100).
    crop_size_px : int, optional
        Crop size in pixels, one side (default is 20).
    image_center : Optional[int], optional
        Center of the image to crop (default is None).
    return_image : Optional[bool], optional
        Whether to return the cropped image (default is False).

    Returns
    -------
    weighted_metric : float
        Weighted metric value.
    """
    if crop_size:
        if image_center is None:
            center = get_image_center(image, threshold)
        else:
            center = image_center
        image = get_cropped_image(image, crop_size, center)

    if len(image.shape) == 3:
        image = np.max(image, axis=0)

    image_perc = np.percentile(image, 90)
    max_pixels = image[image >= image_perc]

    if return_image:
        return np.mean(max_pixels), image
    else:
        return np.mean(max_pixels)


def metric_gauss2d(image: ArrayLike,
                   crop_size: Optional[int] = None,
                   threshold: Optional[float] = 100,
                   image_center: Optional[int] = None,
                   return_image: Optional[bool]= False
                   ) -> float:
    """Compute weighted metric for 2D gaussian.

    Parameters
    ----------
    image : ArrayLike
        2D image.
    threshold : float, optional
        Initial threshold to find spot (default is 100).
    crop_size_px : int, optional
        Crop size in pixels, one side (default is 20).
    image_center : Optional[int], optional
        Center of the image to crop (default is None).
    return_image : Optional[bool], optional
        Whether to return the cropped image (default is False).

    Returns
    -------
    weighted_metric : float
        Weighted metric value.
    """
    # Optionally crop the image
    if crop_size:    
        if image_center is None:
            center = get_image_center(image, threshold)
        else:
            center = image_center
        # crop image
        image = get_cropped_image(image, crop_size, center)
        
    # normalize image 0-1
    image = image / np.max(image)
    image = image.astype(np.float32)
    
    # create coord. grid for fitting 
    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    x, y = np.meshgrid(x, y)
    
    # fitting assumes a single bead in FOV....
    initial_guess = (image.max(), image.shape[1] // 2, 
                     image.shape[0] // 2, 5, 5, image.min())
    fit_bounds = [[0,0,0,1.0,1.0,0],
                  [1.5,image.shape[1],image.shape[0],100,100,5000]]
    try:
        popt, pcov = curve_fit(gauss2d, (x, y), image.ravel(), 
                               p0=initial_guess,
                               bounds=fit_bounds,
                               maxfev=1000)
        
        amplitude, center_x, center_y, sigma_x, sigma_y, offset = popt
        weighted_metric = ((1 - np.abs((sigma_x-sigma_y) / (sigma_x+sigma_y))) 
                           + (1 / (sigma_x+sigma_y)) 
                           + np.exp(-1 * (sigma_x+sigma_y-4)**2))
        
        if (weighted_metric <= 0) or (weighted_metric > 100):
            weighted_metric = 1e-12 
    except Exception:
        weighted_metric = 1e-12
        
        
    if return_image:
        return weighted_metric, image
    else:
        return weighted_metric


def metric_gauss3d(
    image: ArrayLike,
    metric_value: str = "mean",
    crop_size: Optional[int] = None,
    threshold: Optional[float] = 100,
    image_center: Optional[int] = None,
    verbose: Optional[bool] = False,
    return_image: Optional[bool] = False
    ):
    """Compute weighted metric for 3D Gaussian using LocalizePSF
    
    Parameters
    ----------
    image : ArrayLike
        2D image.
    metric_value: str
        Whether to average fit values or generate an average PSF and fit the result.
    threshold : float, optional
        Initial threshold to find spot (default is 100).
    crop_size_px : int, optional
        Crop size in pixels, one side (default is 20).
    image_center : Optional[int], optional
        Center of the image to crop (default is None).
    return_image : Optional[bool], optional
        Whether to return the cropped image (default is False).

    Returns
    -------
    weighted_metric : float
        Weighted metric value.
    """
    pass
    """
    Need to work on this:
    1. is there a conflict installing any of the localize_psf packages?
    """
    # # Optionally crop the image
    # if crop_size:    
    #     if image_center is None:
    #         center = get_image_center(image, threshold)
    #     else:
    #         center = image_center
    #     # crop image
    #     image = get_cropped_image(image, crop_size, center)
        
    # image = image / np.max(image)
    # image = image.astype(np.float32)
        
    # # Define coordinates to pass to localization, use pixel units
    # # Using pixel units, but assumes we are using 0.270 z-steps
    # dxy = 1 # 0.115
    # dz  = 0.250 / 0.115 # 0.250 
    # coords_3d = get_coords(image.shape, (dz, dxy, dxy))
    # # coords_2d = get_coords(cropped_image.shape[1:], (dxy, dxy))
    
    # # Prepare filter for localization
    # sigma_bounds = ((0.1, 0.1),(100, 100)) # [xy min, xy max, z min, z max]
    # amp_bounds = (0.1, 2.0) # [min / max]
    # param_filter = get_param_filter(coords_3d,
    #                                 fit_dist_max_err=(5, 5),
    #                                 min_spot_sep=(10, 6),
    #                                 amp_bounds=amp_bounds,
    #                                 dist_boundary_min=[3, 3],
    #                                 sigma_bounds=sigma_bounds
    #                             )
    # filter = param_filter  
     
    # # define roi sizes used in fitting, assumes a minimum 3um z-stack, dz=0.27um
    # fit_roi_size = [9, 7, 7]
    
    # # Run localization function
    # model = psf.gaussian3d_psf_model()
    # _, r, img_filtered = localize_beads_generic(
    #     image,
    #     drs=(dz, dxy, dxy),
    #     threshold=0.5,
    #     roi_size=fit_roi_size,
    #     filter_sigma_small=None,
    #     filter_sigma_large=None,
    #     min_spot_sep=(10,10),
    #     model=model,
    #     filter=filter,
    #     max_nfit_iterations=100,
    #     use_gpu_fit=False,
    #     use_gpu_filter=False,
    #     return_filtered_images=True,
    #     fit_filtered_images=False,
    #     verbose=True
    #     )
    
    # if r is None:
    #     print("no beads found!")
    #     return 0, image[image.shape[0]//2]
    # else:
    #     to_keep = r["to_keep"]
    #     fit_params = r["fit_params"]
    #     sz = fit_params[to_keep, 5]
    #     sxy = fit_params[to_keep, 4]
    #     amp = fit_params[to_keep, 0]
    #     # Use averages over fit results
    #     if metric_value=="mean":
    #         sz = np.mean(fit_params[to_keep, 5])
    #         sxy = np.mean(fit_params[to_keep, 4])
    #         amp = np.mean(fit_params[to_keep, 0])
    #     elif metric_value=="median":
    #         sz = np.median(fit_params[to_keep, 5])
    #         sxy = np.median(fit_params[to_keep, 4])
    #         amp = np.median(fit_params[to_keep, 0])
    #     elif metric_value=="average":
    #         # Generate average PSF
    #         fit_roi_size_pix = np.round(np.array(fit_roi_size) / np.array([dz, dxy, dxy])).astype(int)
    #         fit_roi_size_pix += (1 - np.mod(fit_roi_size_pix, 2))

    #         psfs_real = np.zeros((1) + tuple(fit_roi_size_pix))
    #         # otfs_real = np.zeros(psfs_real.shape, dtype=complex)
    #         fit_params_average = np.zeros((1, model.nparams))
    #         psf_coords = None

    #         # only use a percent of bead results based on the sxy
    #         percentile = 50
    #         if percentile:
    #             sigma_max = np.percentile(fit_params[:, 4][to_keep], percentile)
    #             to_use = np.logical_and(to_keep, fit_params[:, 4] <= sigma_max)

    #             # get centers
    #             centers = np.stack((fit_params[:, 3][to_use],
    #                                 fit_params[:, 2][to_use],
    #                                 fit_params[:, 1][to_use]), axis=1)

    #         # find average experimental psf/otf
    #         psfs_real, psf_coords = psf.average_exp_psfs(r["data"],
    #                                                      coords_3d,
    #                                                      centers,
    #                                                      fit_roi_size_pix,
    #                                                      backgrounds=fit_params[:, 5][to_use],
    #                                                      return_psf_coords=True)

    #         # fit average experimental psf
    #         def fn(p): return model.model(psf_coords, p)
    #         init_params = model.estimate_parameters(psfs_real, psf_coords)

    #         results = fit_model(psfs_real, fn, init_params, jac='3-point', x_scale='jac')
    #         fit_params_average = results["fit_params"]     

    #         sz = fit_params_average[5]
    #         sxy = fit_params_average[4]
    #         amp = fit_params_average[0]    
        
    #     # TODO: Refine weighted metric if needed
    #     weight_amp = 1 # scales amplitude to a value between 0-65
    #     # SJS: Normalize image and remove brightness
    #     weight_xy = 2 
    #     weight_z = 2
    #     weighted_metric = weight_amp * amp + weight_xy / sxy + weight_z / sz + np.exp(-1*(sxy+sz-6)**2)

    #     if return_image:
    #         return weighted_metric, image
    #     else:
    #         return weighted_metric


def metric_shannon_dct(
    image: ArrayLike, 
    shannon_psf_radius_px: float = 3,
    crop_size: Optional[int] = 501,
    threshold: Optional[float] = 100,
    image_center: Optional[int] = None,
    return_image: Optional[bool] = False
    ) -> float:
    """Compute the Shannon entropy metric using DCT.

    Parameters
    ----------
    image : ArrayLike
        2D image.
    shannon_psf_radius_px : float, optional
        Estimated point spread function (PSF) radius in pixels (default: 3).
    crop_size : Optional[int], optional
        Crop size for image (default: 501).
    threshold : Optional[float], optional
        Intensity threshold to find the center (default: 100).
    image_center : Optional[int], optional
        Custom image center (default: None).
    return_image : Optional[bool], optional
        Whether to return the image along with the metric (default: False).
    
    Returns
    -------
    entropy_metric : float
        Shannon entropy metric.
    """
    # Crop image if necessary
    if crop_size:
        if image_center is None:
            center = get_image_center(image, threshold)  # Ensure this function is defined
        else:
            center = image_center

        # Crop image (ensure get_cropped_image is correctly implemented)
        image = get_cropped_image(image, crop_size, center)

    # Compute the cutoff frequency based on OTF radius
    cutoff = otf_radius(image, shannon_psf_radius_px)

    # Compute DCT
    dct_result = dct_2d(image)

    # Compute Shannon entropy within the cutoff radius
    shannon_dct = shannon(dct_result, cutoff)

    if return_image:
        return shannon_dct, image
    else:
        return shannon_dct


def metric_localize_gauss2d(image: ArrayLike) -> float:
    """_summary_

    Parameters
    ----------
    image : ArrayLike
        _description_

    Returns
    -------
    float
        _description_
    """
    fit_results = localize_2d_img(
        image, 
        0.115,
        {"threshold":3000,
         "amp_bounds":(1000, 30000),
         "sxy_bounds":(0.100, 1.0)
         },
        save_dir_path = None,
        label = "", 
        showfig = False,
        verbose = False
        )
    
    to_keep = fit_results["to_keep"]
    sxy = fit_results["fit_params"][to_keep, 4]
    metric = np.median(sxy)
    
    return metric


#-------------------------------------------------#
# Helper functions for saving optmization results
#-------------------------------------------------#

# def save_optimization_results(iteration_images: ArrayLike,
#                               mode_delta_images: ArrayLike,
#                               optimal_coefficients: ArrayLike,
#                               optimal_metrics: ArrayLike,
#                               modes_to_optimize: List[int],
#                               results_save_path: Path):
#     """_summary_

#     Parameters
#     ----------
#     optimal_coefficients : ArrayLike
#         _description_
#     optimal_metrics : ArrayLike
#         _description_
#     modes_to_optimize : List[int]
#         _description_
#     results_save_path : Path
#         _description_
#     """
#     with h5py.File(str(results_save_path), "w") as f:
#                 f.create_dataset("optimal_images", data=iteration_images)
#                 f.create_dataset("mode_delta_images", data=mode_delta_images)
#                 f.create_dataset("optimal_coefficients", data=optimal_coefficients)
#                 f.create_dataset("optimal_metrics", data=optimal_metrics)
#                 f.create_dataset("modes_to_optimize", data=modes_to_optimize)
#                 f.create_dataset("zernike_mode_names", data=np.array(mode_names, dtype="S"))
    
    
# def load_optimization_results(results_path: Path):
#     """_summary_

#     Parameters
#     ----------
#     results_path : Path
#         _description_
#     """
#     # Load the mixed dictionary from HDF5
#     with h5py.File(str(results_path), "r") as f:
#         optimal_images = f["optimal_images"][:]
#         mode_delta_images = f["mode_delta_images"][:]
#         optimal_coefficients = f["optimal_coefficients"][:]
#         optimal_metrics = f["optimal_metrics"][:]
#         modes_to_optimize = f["modes_to_optimize"][:]
#         zernike_mode_names = [name.decode("utf-8") for name in f["zernike_mode_names"][:]]

#     return optimal_images, mode_delta_images, optimal_coefficients, optimal_metrics, modes_to_optimize, zernike_mode_names


#-------------------------------------------------#
# Run as script 'keeps mirror flat'
#-------------------------------------------------#
if __name__ == "__main__":
    """Keeps the mirror in it's flat position
    """
    wfc_config_file_path = Path(r"C:\Users\qi2lab\Documents\github\opm_ao\Configuration Files\WaveFrontCorrector_mirao52-e_0329.dat")
    wfc_correction_file_path = Path(r"C:\Users\qi2lab\Documents\github\opm_ao\OUT_FILES\correction_data_backup_starter.aoc")
    haso_config_file_path = Path(r"C:\Users\qi2lab\Documents\github\opm_ao\Configuration Files\WFS_HASO4_VIS_7635.dat")
    wfc_flat_file_path = Path(r"C:\Users\qi2lab\Documents\github\opm_ao\OUT_FILES\flat_actuator_positions.wcs")
    # wfc_calibrated_flat_path = Path(r"C:\Users\qi2lab\Documents\github\opm_ao\OUT_FILES\20250122_tilted_gauss2d_laser_actuator_positions.wcs")
    wfc_calibrated_flat_path = Path(r"C:\Users\qi2lab\Documents\github\opm_ao\OUT_FILES\20250215_tilted_brightness_laser_actuator_positions.wcs")
    # Load ao_mirror controller
    # ao_mirror puts the mirror in the flat_position state to start.
    ao_mirror = AOMirror(wfc_config_file_path = wfc_config_file_path,
                         haso_config_file_path = haso_config_file_path,
                         interaction_matrix_file_path = wfc_correction_file_path,
                         flat_positions_file_path = wfc_calibrated_flat_path)
    
    input("Press enter to exit . . . ")
    ao_mirror = None



