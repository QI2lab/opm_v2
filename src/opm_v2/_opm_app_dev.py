"""qi2lab modified version of the launching script for pymmcore-gui.

qi2lab specific changes start on ~ line 112.

Change Log:
2025/02: Initial version of the script.
"""
from __future__ import annotations
# TODO AO getting values from gui instead of json.
import argparse
import importlib
import importlib.util
import os
import sys
import traceback
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, cast

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QDockWidget
from superqt.utils import WorkerBase

from pymmcore_gui import MicroManagerGUI,WidgetAction, __version__
import json

import numpy as np

from opm_v2.hardware.OPMNIDAQ import OPMNIDAQ
from opm_v2.hardware.AOMirror import AOMirror
from opm_v2.hardware.ElveFlow import OB1Controller
from opm_v2.hardware.PicardShutter import PicardShutter
from opm_v2.engine.opm_engine_dev import OPMEngine
from opm_v2._update_config_widget import OPMSettings
from opm_v2.engine.setup_events_dev import (
    setup_stagescan,
    setup_mirrorscan,
    setup_projection,
    setup_optimizenow
)

use_mda_channels = False
DEBUGGING = True
if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

    ExcTuple = tuple[type[BaseException], BaseException, TracebackType | None]


APP_NAME = "Micro-Manager GUI"
APP_VERSION = __version__
ORG_NAME = "pymmcore-plus"
ORG_DOMAIN = "pymmcore-plus"
APP_ID = f"{ORG_DOMAIN}.{ORG_NAME}.{APP_NAME}.{APP_VERSION}"
RESOURCES = Path(__file__).parent / "resources"
ICON = RESOURCES / ("icon.ico" if sys.platform.startswith("win") else "logo.png")
IS_FROZEN = getattr(sys, "frozen", False)


class MMQApplication(QApplication):
    exceptionRaised = pyqtSignal(BaseException)

    def __init__(self, argv: list[str]) -> None:
        if sys.platform == "darwin" and not argv[0].endswith("mmgui"):
            # Make sure the app name in the Application menu is `mmgui`
            # which is taken from the basename of sys.argv[0]; we use
            # a copy so the original value is still available at sys.argv
            argv[0] = "napari"

        super().__init__(argv)
        self.setApplicationName("Micro-Manager GUI")
        self.setWindowIcon(QIcon(str(ICON)))

        self.setApplicationName(APP_NAME)
        self.setApplicationVersion(APP_VERSION)
        self.setOrganizationName(ORG_NAME)
        self.setOrganizationDomain(ORG_DOMAIN)
        if os.name == "nt" and not IS_FROZEN:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)  # type: ignore

        self.aboutToQuit.connect(WorkerBase.await_workers)


def parse_args(args: Sequence[str] = ()) -> argparse.Namespace:
    if not args:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(description="Enter string")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Config file to load",
        nargs="?",
    )
    return parser.parse_args(args)


def main() -> None:
    """Run the Micro-Manager GUI."""
    # args = parse_args()

    app = MMQApplication(sys.argv)
    # _install_excepthook()

    win = MicroManagerGUI()
    win.setWindowIcon(QIcon(str(ICON)))
    win.showMaximized()
    win.show()

    splsh = "_PYI_SPLASH_IPC" in os.environ and importlib.util.find_spec("pyi_splash")
    if splsh:  # pragma: no cover
        import pyi_splash  # pyright: ignore [reportMissingModuleSource]

        pyi_splash.update_text("UI Loaded ...")
        pyi_splash.close()

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    # ----------------Begin custom qi2lab code for running OPM control----------------
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------

    # load microscope configuration file
    config_path = Path(r"C:\Users\qi2lab\Documents\github\opm_v2\opm_config_20250312.json")
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
        
    # Load the widget for interacting with the configuration.
    opmSettings_widget = OPMSettings(config_path=config_path)
    dock_widget = QDockWidget("OPM Settings", win)  
    dock_widget.setWidget(opmSettings_widget)
    dock_widget.setObjectName("OPMConfigurator")
    win.addDockWidget(Qt.RightDockWidgetArea, dock_widget)  
    dock_widget.setFloating(False)
    
    # Start the mirror in the flat_position position.
    opmAOmirror = AOMirror(
        wfc_config_file_path = Path(config["AOMirror"]["wfc_config_path"]),
        haso_config_file_path = Path(config["AOMirror"]["haso_config_path"]),
        interaction_matrix_file_path = Path(config["AOMirror"]["wfc_correction_path"]),
        flat_positions_file_path = Path(config["AOMirror"]["wfc_flat_path"]),
        n_modes = 32,
        n_positions=1,
        modes_to_ignore = []
    )
    
    opmAOmirror.set_mirror_positions_flat()

    # load OPM NIDAQ and OPM AO mirror classes
    opmNIDAQ = OPMNIDAQ(
        name = str(config["NIDAQ"]["name"]),
        scan_type = str(config["NIDAQ"]["scan_type"]),
        exposure_ms = float(config["Camera"]["exposure_ms"]),
        laser_blanking = bool(config["NIDAQ"]["laser_blanking"]),
        image_mirror_calibration = float(str(config["NIDAQ"]["image_mirror_calibration"])),
        projection_mirror_calibration = float(str(config["NIDAQ"]["projection_mirror_calibration"])),
        image_mirror_neutral_v = float(str(config["NIDAQ"]["image_mirror_neutral_v"])),
        projection_mirror_neutral_v = float(str(config["NIDAQ"]["projection_mirror_neutral_v"])),
        image_mirror_step_size_um = float(str(config["NIDAQ"]["image_mirror_step_size_um"])),
        verbose = bool(config["NIDAQ"]["verbose"])
    )
    opmNIDAQ.reset()

    # Initialize ElveFlow OB1 Controller
    opmOB1 = OB1Controller(
        port=config["OB1"]["port"],
        to_OB1_pin=config["OB1"]["to_OB1_pin"],
        from_OB1_pin=config["OB1"]["from_OB1_pin"]
    )
    
    # Initialize and close alignment laser shutter
    opmPicardShutter = PicardShutter(int(config["O2O3-autofocus"]["shutter_id"]))
    opmPicardShutter.closeShutter()
    
    # grab mmc instance and load OPM config file
    mmc = win.mmcore
    mmc.loadSystemConfiguration(Path(config["OPM"]["mm_config_path"]))
    mda_widget = win.get_widget(WidgetAction.MDA_WIDGET)

    # Enforce config's default properties
    mmc.setProperty(
        config["Camera"]["camera_id"],
        "Exposure",
        float(config["Camera"]["exposure_ms"])
    )
    mmc.waitForDevice(str(config["Camera"]["camera_id"]))
    
    mmc.clearROI()
    mmc.waitForDevice(str(config["Camera"]["camera_id"]))
    
    mmc.setROI(
        config["Camera"]["roi_center_x"] - int(config["Camera"]["roi_crop_x"]//2),
        config["Camera"]["roi_center_y"] - int(config["Camera"]["roi_crop_y"]//2),
        config["Camera"]["roi_crop_x"],
        config["Camera"]["roi_crop_y"]
    )
    mmc.waitForDevice(str(config["Camera"]["camera_id"]))

    def update_config():
        """Updates config from file on disk
        """
        with open(config_path, "r") as config_file:
            new_config = json.load(config_file)
        config.update(new_config)
        
    def calculate_projection_crop(image_mirror_range_um: float):
        """Return the projection mode ROI y-crop

        Parameters
        ----------
        image_mirror_range_um : float
            image mirror scan range in um

        Returns
        -------
        roi_height_px: float
            ROI height in pixels
        """
        roi_height_um = image_mirror_range_um
        roi_height_px = int(roi_height_um / mmc.getPixelSizeUm())
        return roi_height_px

    def update_live_state(device_name=None, property_name=None):
        """Update microscope states and values upon changes to in the GUI

        Parameters
        ----------
        device_name : str, optional
            signal source device name, by default None
        property_name : str, optional
            signal source propery name, by default None
        """
        
        # if "ImageGalvoMirrorRange" in device_name:
        # elif "Laser" in device_name:
            
        #     is not
        update_config()
            
        # Ignore updates from AutoShutter
        if device_name == mmc.getShutterDevice() and property_name == "State":
            return
        
        if DEBUGGING:
            print("update_live_state:")
            print(f"    device name: {device_name}, property name: {property_name}")
        
        
        # Only 2d and projection modes are available in live mode
        opm_mode = mmc.getProperty("OPM-live-mode", "Label")
        if "Standard" in opm_mode:
            _scan_type = "2d"
        elif "Projection" in opm_mode:
            _scan_type = "projection"
        
        #--------------------------------------------------------------------#
        # Stop DAQ playback
        opmNIDAQ_update_state = OPMNIDAQ.instance()
        restart_sequence = False
        if mmc.isSequenceRunning():
            mmc.stopSequenceAcquisition()
            restart_sequence = True
        if opmNIDAQ_update_state.running():
            opmNIDAQ_update_state.stop_waveform_playback()

        if DEBUGGING:
            # Useful for calibration / debugging
            _projection_calibration = config["NIDAQ"]["projection_mirror_calibration"]
            _image_mirror_neutral_v = config["NIDAQ"]["image_mirror_neutral_v"]
            opmNIDAQ_update_state.projection_mirror_calibration = _projection_calibration
            opmNIDAQ_update_state._ao_neutral_positions[0] = _image_mirror_neutral_v
        
        #--------------------------------------------------------------------#
        # Grab gui values 
        _exposure_ms = round(float(mmc.getProperty(config["Camera"]["camera_id"], "Exposure")), 0)
        _image_mirror_range_um = np.round(float(mmc.getProperty("ImageGalvoMirrorRange", "Position")),0)
        _active_channel_id = mmc.getProperty("Laser", "Label")
        
        # Compile channel states for daq 
        _channel_states = [False] * len(config["OPM"]["channel_ids"])
        for ch_i, ch_str in enumerate(config["OPM"]["channel_ids"]):
            if _active_channel_id==ch_str:
                _channel_states[ch_i] = True
        
        #--------------------------------------------------------------------#
        # Enforce camera ROI
        if _scan_type=="projection":
            crop_y = calculate_projection_crop(_image_mirror_range_um)
            change_crop = True
        elif "Camera-CropY" in device_name:
            crop_y = int(mmc.getProperty("ImageCameraCrop","Label"))
            change_crop = True
        else: 
            change_crop = False
        
        if change_crop:
            if not (crop_y == mmc.getROI()[-1]): 
                current_roi = mmc.getROI()
                mmc.clearROI()
                mmc.waitForDevice(str(config["Camera"]["camera_id"]))
                mmc.setROI(
                    config["Camera"]["roi_center_x"] - int(config["Camera"]["roi_crop_x"]//2),
                    config["Camera"]["roi_center_y"] - int(crop_y//2),
                    config["Camera"]["roi_crop_x"],
                    crop_y
                )
                mmc.waitForDevice(str(config["Camera"]["camera_id"]))
                
        #--------------------------------------------------------------------#
        # Enforce the camera exposure
        if ("projection" in opm_mode) and (_exposure_ms<50):
            print("Exposure too low for projection mode! \n reverting to 500ms")
            # Set the camera exposure
            _exposure_ms = 500
        
        mmc.setProperty(str(config["Camera"]["camera_id"]),"Exposure",_exposure_ms)
        mmc.waitForDevice(str(config["Camera"]["camera_id"]))
        
        #--------------------------------------------------------------------#
        # update DAQ values and prepare new waveform, restart acq.
        opmNIDAQ_update_state.set_acquisition_params(
            scan_type=_scan_type,
            channel_states=_channel_states,
            image_mirror_range_um=_image_mirror_range_um,
            exposure_ms=_exposure_ms,
        )
        
        #--------------------------------------------------------------------#
        # Update mirror positions
        if "AO_mode" in device_name:
            ao_mirror_state = mmc.getProperty("AO-mode", "Label")
            AOMirror_update_state = AOMirror.instance()
            if "ystem" in ao_mirror_state:
                mirror_key = "system_flat"
            elif "optimized" in ao_mirror_state:
                mirror_key = "last_optimized"
            AOMirror_update_state.set_mirror_positions(
                AOMirror_update_state.wfc_positions[mirror_key]
            )
            
        if restart_sequence:
            mmc.startContinuousSequenceAcquisition()
            
    # Connect changes in gui fields to the update_live_state method.            
    # mmc.events.propertyChanged.connect(update_live_state)
    mmc.events.configSet.connect(update_live_state)
        
    def setup_preview_mode_callback():
        """Callback to intercept preview mode and setup the OPM.
        
        This function parses the various configuration groups and creates
        the appropriate NIDAQ waveforms for the selected OPM mode and channel.
        """
        # get instance of opmnidaq here
        opmNIDAQ_setup_preview = OPMNIDAQ.instance()

        if opmNIDAQ_setup_preview.running():
            opmNIDAQ_setup_preview.stop_waveform_playback()
        
        # check if any channels are active. If not, don't setup DAQ.
        if any(opmNIDAQ_setup_preview.channel_states):
            # Check OPM mode and set up NIDAQ accordingly
            opmNIDAQ_setup_preview.clear_tasks()
            opmNIDAQ_setup_preview.generate_waveforms()
            opmNIDAQ_setup_preview.program_daq_waveforms()
            opmNIDAQ_setup_preview.start_waveform_playback()

    # Connect the above callback to the event that a continuous sequence is starting
    mmc.events.continuousSequenceAcquisitionStarting.connect(setup_preview_mode_callback)
    
    def custom_execute_mda(output: Path | str | object | None) -> None:
        """Custom execute_mda method that modifies the sequence before running it.
        
        This function parses the various configuration groups and the MDA sequence.
        It then creates a new MDA sequence based on the configuration settings.
        Importantly, we add custom metadata events that trigger the custom parts 
        of our acquistion engine.

        Parameters
        ----------
        output : Path | str | object | None
            The output path for the MDA sequence.
        """

        #--------------------------------------------------------------------#
        # Get the acquisition settings
        #--------------------------------------------------------------------#
        
        update_config()
                       
        opm_mode = config["acq_config"]["opm_mode"]
        ao_mode = config["acq_config"]["AO"]["ao_mode"]
        o2o3_mode = config["acq_config"]["O2O3-autofocus"]["o2o3_mode"]
        fluidics_mode = config["acq_config"]["fluidics"]
                  
        #--------------------------------------------------------------------#
        # Validate acquisition settings
        #--------------------------------------------------------------------#
        
        if ("now" in ao_mode) or ("now" in o2o3_mode):
            optimize_now = True
        else: 
            optimize_now = False        
                    
        if not(optimize_now) and not(output):
            print("Must set acquisition path to excecute acquisition")
            return
        
        if "none" not in fluidics_mode:
            # load dialog to have user verify ESI is running.
            # TODO: ad an entry for the number of rounds.
            from PyQt6.QtWidgets import QMessageBox
            response = QMessageBox.information(
                mda_widget, 
                'WARNING ! ! ! ESI MUST BE RUNNING ! ! !',
                'IS ESI SEQUENCE LOADED AND STARTED?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            )
            if response is not QMessageBox.StandardButton.Yes:
                return
            else:
                print("ESI Sequence accepted")
        
        #--------------------------------------------------------------------#
        # Get event structure for requested acquisition type
        #--------------------------------------------------------------------#
        
        if optimize_now:
            opm_events, handler = setup_optimizenow(
                mmc = mmc,
                config = config
                )
        elif ("stage" in opm_mode):
            opm_events, handler = setup_stagescan(
                mmc = mmc,
                config = config,
                sequence = mda_widget.value(),
                output = output
                )
        elif ("mirror" in opm_mode):
            opm_events, handler = setup_mirrorscan(
                mmc = mmc,
                config = config,
                sequence = mda_widget.value(),
                output = output
                )
        elif ("projection" in opm_mode):
            opm_events, handler = setup_projection(
                mmc = mmc,
                config = config,
                sequence = mda_widget.value(),
                output = output
                )
        if not(optimize_now):
            # tell AO mirror class where to save mirror information
            opmAOmirror_local = AOMirror.instance()
            opmAOmirror_local.output_path = output.parents[0]

        if opm_events is None:
            print("MDA acquisition not started!")
            return
        #--------------------------------------------------------------------#
        # Run Qi2lab custom MDA acquisition
        #--------------------------------------------------------------------#
        mda_widget._mmc.run_mda(opm_events, output=handler)
            
    # modify the method on the instance
    mda_widget.execute_mda = custom_execute_mda

    # Register the custom OPM MDA engine with mmc
    mmc.mda.set_engine(OPMEngine(mmc, config_path))
    
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    # -----------------End custom qi2lab code for running OPM control-----------------
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------

    app.exec()


# ------------------- Custom excepthook -------------------

def _install_excepthook() -> None:
    """Install a custom excepthook that does not raise sys.exit().

    This is necessary to prevent the application from closing when an exception
    is raised.
    """
    if hasattr(sys, "_original_excepthook_"):
        return
    sys._original_excepthook_ = sys.excepthook  # type: ignore
    sys.excepthook = ndv_excepthook


def rich_print_exception(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_traceback: TracebackType | None,
) -> None:
    import psygnal
    from rich.console import Console
    from rich.traceback import Traceback

    tb = Traceback.from_exception(
        exc_type,
        exc_value,
        exc_traceback,
        suppress=[psygnal],
        max_frames=100 if IS_FROZEN else 10,
        show_locals=True,
    )
    Console(stderr=True).print(tb)


def _print_exception(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_traceback: TracebackType | None,
) -> None:
    try:
        rich_print_exception(exc_type, exc_value, exc_traceback)
    except ImportError:
        traceback.print_exception(exc_type, value=exc_value, tb=exc_traceback)


# This log list is used by the ExceptionLog widget
# Be aware that it's currently possible for that widget to clear this list.
# If an immutable record of exceptions is needed, additional logic will be required.
EXCEPTION_LOG: list[ExcTuple] = []


def ndv_excepthook(
    exc_type: type[BaseException], exc_value: BaseException, tb: TracebackType | None
) -> None:
    EXCEPTION_LOG.append((exc_type, exc_value, tb))
    _print_exception(exc_type, exc_value, tb)
    if sig := getattr(QApplication.instance(), "exceptionRaised", None):
        sig.emit(exc_value)
    if not tb:
        return

    # if we're running in a vscode debugger, let the debugger handle the exception
    if (
        (debugpy := sys.modules.get("debugpy"))
        and debugpy.is_client_connected()
        and ("pydevd" in sys.modules)
    ):  # pragma: no cover
        with suppress(Exception):
            import threading

            import pydevd  # pyright: ignore [reportMissingImports]

            if (py_db := pydevd.get_global_debugger()) is None:
                return

            py_db = cast("pydevd.PyDB", py_db)
            thread = threading.current_thread()
            additional_info = py_db.set_additional_thread_info(thread)
            additional_info.is_tracing += 1

            try:
                arg = (exc_type, exc_value, tb)
                py_db.stop_on_unhandled_exception(py_db, thread, additional_info, arg)
            finally:
                additional_info.is_tracing -= 1
    # otherwise, if MMGUI_DEBUG_EXCEPTIONS is set, drop into pdb
    elif os.getenv("MMGUI_DEBUG_EXCEPTIONS"):
        import pdb

        pdb.post_mortem(tb)

    # after handling the exception, exit if MMGUI_EXIT_ON_EXCEPTION is set
    if os.getenv("MMGUI_EXIT_ON_EXCEPTION"):
        print("\nMMGUI_EXIT_ON_EXCEPTION is set, exiting.")
        sys.exit(1)