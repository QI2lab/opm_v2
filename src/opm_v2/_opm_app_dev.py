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
from datetime import datetime

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QDockWidget
from superqt.utils import WorkerBase

from pymmcore_gui import MicroManagerGUI,WidgetAction, __version__
import json

import numpy as np
from useq import MDAEvent, CustomAction
from types import MappingProxyType as mappingproxy

from opm_v2.hardware.OPMNIDAQ import OPMNIDAQ
from opm_v2.hardware.AOMirror import AOMirror
from opm_v2.hardware.ElveFlow import OB1Controller
from opm_v2.hardware.PicardShutter import PicardShutter
from opm_v2.engine.opm_engine import OPMEngine
from opm_v2.handlers.opm_mirror_handler import OPMMirrorHandler
from opm_v2._update_config_widget import OPMSettings
from opm_v2.utils.opm_custom_actions import (O2O3_af_event,
                                             AO_optimize_event,
                                             FP_event,
                                             DAQ_event)

use_mda_channels = False

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
        
    def update_config():
        with open(config_path, "r") as config_file:
            new_config = json.load(config_file)
        config.update(new_config)
        
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

    def update_state(device_name=None, property_name=None):
        """Update microscope states and values upon changes to in the GUI

        Parameters
        ----------
        device_name : str, optional
            signal source device name, by default None
        property_name : str, optional
            signal source propery name, by default None
        """
        update_config()
        
        # print(f"device name: {device_name}, property name: {property_name}")
        # Ignore updates from AutoShutter
        if device_name == mmc.getShutterDevice() and property_name == "State":
            return
            
        # Only 2d and projection modes are available in live mode
        opm_mode = mmc.getProperty("OPM-mode", "Label")
        if "Standard" in opm_mode:
            _scan_type = "2d"
        elif "Stage" in opm_mode:
            _scan_type = "2d"
        elif "Projection" in opm_mode:
            _scan_type = "projection"
        elif "Mirror" in opm_mode:
            _scan_type = "2d"
            
        #--------------------------------------------------------------------#
        # Stop DAQ playback
        opmNIDAQ_update_state = OPMNIDAQ.instance()
        restart_sequence = False
        if mmc.isSequenceRunning():
            mmc.stopSequenceAcquisition()
            restart_sequence = True
        if opmNIDAQ_update_state.running():
            opmNIDAQ_update_state.stop_waveform_playback()

        # Useful for calibration / debugging
        _projection_calibration = config["NIDAQ"]["projection_mirror_calibration"]
        _image_mirror_neutral_v = config["NIDAQ"]["image_mirror_neutral_v"]
        opmNIDAQ_update_state.projection_mirror_calibration = _projection_calibration
        opmNIDAQ_update_state._ao_neutral_positions[0] = _image_mirror_neutral_v
        
        #--------------------------------------------------------------------#
        # Grab gui values 
        
        _exposure_ms = round(float(mmc.getProperty(config["Camera"]["camera_id"], "Exposure")), 0)
        _image_mirror_range_um = np.round(float(mmc.getProperty("ImageGalvoMirrorRange", "Position")),0)
        _image_mirror_step_um = np.round(float(mmc.getProperty("ImageGalvoMirrorStep", "Label").split("-")[0]),2)
        _active_channel_id = mmc.getProperty("LED", "Label")
        
        # Compile channel states for daq 
        _channel_states = [False] * len(config["OPM"]["channel_ids"])
        for ch_i, ch_str in enumerate(config["OPM"]["channel_ids"]):
            if _active_channel_id==ch_str:
                _channel_states[ch_i] = True
        if mmc.getProperty("LaserBlanking", "Label")=="On":
            _laser_blanking = True
        else:
            _laser_blanking = False
        
        #--------------------------------------------------------------------#
        # Enforce camera ROI
        if _scan_type=="projection":
            crop_y = calculate_projection_crop(_image_mirror_range_um)
        else :
            crop_y = int(mmc.getProperty("ImageCameraCrop","Label"))
        
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
            image_mirror_step_size_um=_image_mirror_step_um,
            image_mirror_range_um=_image_mirror_range_um,
            laser_blanking=_laser_blanking,
            exposure_ms=_exposure_ms,
        )
        
        if restart_sequence:
            opmNIDAQ_update_state.start_waveform_playback()
            mmc.startContinuousSequenceAcquisition()
            
    # Connect changes in gui fields to the update_state method.            
    mmc.events.propertyChanged.connect(update_state)
    mmc.events.configSet.connect(update_state)
    
    # grab handle to the Stage widget
    # stage_widget = win.get_widget(WidgetAction.STAGE_CONTROL)

    # grab handle to the MDA widget and define custom execute_mda method
    # in our method, the MDAEvents are modified before running the sequence
    mda_widget = win.get_widget(WidgetAction.MDA_WIDGET)
    
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

        update_config()
        
        opmAOmirror_local = AOMirror.instance()
        opmNIDAQ_custom = OPMNIDAQ.instance()
        
        #--------------------------------------------------------------------#
        #--------------------------------------------------------------------#
        # Get the acquisition parameters from config
        #--------------------------------------------------------------------#
        #--------------------------------------------------------------------#
        
        opm_mode = config["acq_config"]["opm_mode"]
        ao_mode = config["acq_config"]["AO"]["ao_mode"]
        o2o3_mode = config["acq_config"]["O2O3-autofocus"]["o2o3_mode"]
        fluidics_mode = config["acq_config"]["fluidics"]
                  
        #--------------------------------------------------------------------#
        # Validate acquisition entries
        #--------------------------------------------------------------------#
        
        if ("now" in ao_mode) or ("now" in o2o3_mode):
            optimize_now = True
            output = Path(str(config["AO-projection"]["optimize_now_path"]))
        else: 
            optimize_now = False        
                    
        if not(optimize_now) and not(output):
            print("Must set acquisition path to excecute acquisition")
            return
        
        if not("none" in fluidics_mode):
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
        
        try:
            offset = mmc.getProperty(
                config["Camera"]["camera_id"],
                "CONVERSION FACTOR OFFSET"
            )
            e_to_ADU = mmc.getProperty(
                config["Camera"]["camera_id"],
                "CONVERSION FACTOR COEFF"
            )
        except Exception:
            offset = 0.
            e_to_ADU = 1.
        
        #--------------------------------------------------------------------#
        #--------------------------------------------------------------------#
        # If optimize now, ignore MDA widget and run_mda now.
        #--------------------------------------------------------------------#
        #--------------------------------------------------------------------#
        
        if optimize_now:
            opm_events: list[MDAEvent] = []
             
            if "now" in o2o3_mode:
                o2o3_event = MDAEvent(**O2O3_af_event.model_dump())
                o2o3_event.action.data["camera"]["exposure_ms"] = float(config["O2O3-autofocus"]["exposure_ms"])
                o2o3_event.action.data["camera"]["camera_crop"] = [
                    config["Camera"]["roi_center_x"] - int(config["Camera"]["camera_crop_x"]//2),
                    config["Camera"]["roi_center_y"] - int(config["O2O3-autofocus"]["roi_crop_y"]//2),
                    config["Camera"]["roi_crop_x"],
                    config["O2O3-autofocus"]["roi_crop_y"]
                    ]
                opm_events.append(o2o3_event)
                
            if "now" in ao_mode:
                now = datetime.now()
                timestamp = f"{now.year:4d}{now.month:2d}{now.day:2d}_{now.hour:2d}{now.minute:2d}{now.second:2d}"
            
                # setup AO using values in the config widget, NOT the MDA widget
                AO_channel_states = [False] * len(config["OPM"]["channel_ids"]) 
                AO_channel_powers = [0.] * len(config["OPM"]["channel_ids"])
                AO_active_channel_id = config["acq_config"]["AO"]["active_channel_id"]
                AO_camera_crop_y = int(calculate_projection_crop(config["acq_config"]["AO"]["image_mirror_range_um"]))
                AO_save_path = Path(str(config["AO-projection"]["optimize_now_path"])) / Path(f"{timestamp}_ao_optimizeNOW")
                
                # Set the active channel in the daq channel list
                for chan_idx, chan_str in enumerate(config["OPM"]["channel_ids"]):
                    if AO_active_channel_id==chan_str:
                        AO_channel_states[chan_idx] = True
                        AO_channel_powers[chan_idx] = config["acq_config"]["AO"]["active_channel_power"]
                        
                # check to make sure there exist a laser power > 0
                if sum(AO_channel_powers)==0:
                    print("All AO lasers set to 0!")
                    return
                
                # Define AO optimization action data   
                ao_action_data = {
                    "AO" : {
                        "channel_states": AO_channel_states,
                        "channel_powers" : AO_channel_powers,
                        "modal_delta": float(config["acq_config"]["AO"]["mode_delta"]),
                        "modal_alpha":float(config["acq_config"]["AO"]["mode_alpha"]),                        
                        "iterations": int(config["acq_config"]["AO"]["num_iterations"]),
                        "metric": str(config["acq_config"]["AO"]["metric"]),
                        "image_mirror_range_um" : config["acq_config"]["AO"]["image_mirror_range_um"],
                        "blanking": bool(True),
                        "apply_existing": bool(False),
                        "output_path":AO_save_path
                    },
                    "Camera" : {
                        "exposure_ms": config["acq_config"]["AO"]["exposure_ms"],
                        "camera_crop" : [
                            config["acq_config"]["camera_roi"]["center_x"] - int(config["acq_config"]["camera_roi"]["crop_x"]//2),
                            config["acq_config"]["camera_roi"]["center_y"] - int(AO_camera_crop_y//2),
                            config["Camera"]["camera_crop_x"],
                            AO_camera_crop_y
                        ]
                    }
                    
                }
                ao_optimize_event = MDAEvent(**AO_optimize_event.model_dump())
                ao_optimize_event.action.data.update(ao_action_data)
                opm_events.append(ao_optimize_event)
                
            # Run optimize now events
            mda_widget._mmc.run_mda(opm_events, output=None)
            
            return

        #--------------------------------------------------------------------#
        #--------------------------------------------------------------------#
        # Run a custom OPM acquisition, use positions / channels from MDA
        # - get the opm mode and setup daq event data
        # - setup custom action event data for AO, o2o3 AF, and fluidics
        # - Setup stage positions from MDA
        # - Setup channel selection from MDA
        # - Setup daq event data
        #--------------------------------------------------------------------#
        #--------------------------------------------------------------------#
        
        else:
            #----------------------------------------------------------------#
            # Get the current MDAsequence and convert to dictionary 
            #----------------------------------------------------------------#
            sequence = mda_widget.value()
            sequence_dict = json.loads(sequence.model_dump_json())
            mda_stage_positions = sequence_dict["stage_positions"]
            mda_grid_plan = sequence_dict["grid_plan"]
            mda_channels = sequence_dict["channels"]        
            mda_time_plan = sequence_dict["time_plan"]
            mda_z_plan = sequence_dict["z_plan"]
            
            if not(use_mda_channels):
                #------------------------------------------------------------#
                # Get the acq. channel settings shared for each scan type from config
                #------------------------------------------------------------#
                laser_blanking = config["acq_config"][opm_mode+"_scan"]["laser_blanking"]
                channel_states = config["acq_config"][opm_mode+"_scan"]["channel_states"]
                channel_powers = config["acq_config"][opm_mode+"_scan"]["channel_powers"]
                channel_exposures_ms = config["acq_config"][opm_mode+"_scan"]["channel_exposures_ms"]
                
            else:
                #------------------------------------------------------------#
                # Generate channel selection from MDA widget
                #------------------------------------------------------------#
                
                # Validate mda channels is set to channel seletion
                if not(mda_channels):
                    print("Must select channels to use in MDA widget")
                    return
                elif not("Channel" in mda_channels["group"]):
                    print("Must select channels to use in MDA widget")
                    return
                
                channel_names = config["OPM"]["channel_ids"]
                channel_states = [False] * len(channel_names) 
                channel_exposures_ms = [0.] * len(channel_names) 
                channel_powers = [0.] * len(channel_names) 
                
                if mmc.getProperty("LaserBlanking", "Label")=="On":
                    laser_blanking = True
                else:
                    laser_blanking = False
                    
                # Validate mda channels is set to channels and not running optimize now.
                if not(mda_channels):
                    print("Must select channels to use in MDA widget")
                    return
                elif not("Channel" in mda_channels["group"]):
                    print("Must select channels to use in MDA widget")
                    return
            
                # Iterate through MDA checked channels and update active lasers list
                for mda_ch in mda_channels:
                    # Get the matching channel idx active lasers list
                    ch_id = mda_ch["config"]
                    ch_idx = config["OPM"]["channel_ids"].index(ch_id)
                    
                    # update active channel and powers
                    channel_states[ch_idx] = True
                    channel_exposures_ms[ch_idx] = mda_ch["exposure"]
                    channel_powers[ch_idx] = float(
                        mmc.getProperty(
                            "Coherent-Scientific Remote",
                            config["Lasers"]["laser_names"][ch_idx] + " - PowerSetpoint (%)"
                            )
                        )
            
            #----------------------------------------------------------------#
            # Compile the active channel info and validate powers
            #----------------------------------------------------------------#
            n_active_channels = sum(channel_states)
            active_channel_names = [_name for _, _name in zip(channel_states, channel_names) if _]
            
            # Interleave only available if all channels have the same exposure.
            active_channel_exps = [_exp for _, _exp in zip(channel_states, channel_exposures_ms) if _]
            if len(set(active_channel_exps))==1:
                interleaved_acq = True
            else:
                interleaved_acq = False
                    
            # Check to make sure there exist a laser power > 0
            if sum(channel_powers)==0:
                print("All lasers set to 0!")
                return

            #----------------------------------------------------------------#
            # Set the daq event data for the selected opm_mode
            #----------------------------------------------------------------#
            
            if "projection" in opm_mode:
                # Get the image mirror range
                image_mirror_range_um = np.round(
                    float(config["acq_config"][opm_mode+"_scan"]["image_mirror_range_um"]),0
                    )
                camera_crop_y = calculate_projection_crop(image_mirror_range_um)
                n_scan_steps = 1 
                interleaved_acq = False
                daq_mode = "projection"
                
                daq_action_data = {
                    "DAQ" : {
                        "mode" : opm_mode,
                        "image_mirror_range_um" : image_mirror_range_um,
                        "channel_states" : channel_states,
                        "channel_powers" : channel_powers,
                        "interleaved" : interleaved_acq,
                        "blanking" : laser_blanking, 
                    },
                    "Camera" : {
                        "exposure_channels" : channel_exposures_ms,
                        "camera_crop" : [
                            int(config["acq_config"]["camera_roi"]["center_x"] - int(config["acq_config"]["camera_roi"]["crop_x"]//2)),
                            int(config["acq_config"]["camera_roi"]["center_y"] - int(camera_crop_y//2)),
                            int(config["acq_config"]["camera_roi"]["crop_x"]),
                            int(camera_crop_y),
                        ]
                    }
                }
                
            elif "mirror" in opm_mode:
                # get image galvo mirror range and step size
                image_mirror_range_um = np.round(
                    float(config["acq_config"][opm_mode+"_scan"]["image_mirror_range_um"]),0
                    )
                image_mirror_step_um = np.round(
                    float(config["acq_config"][opm_mode+"_scan"]["image_mirror_range_um"]),0
                    )
                camera_crop_y = config["acq_config"]["camera_roi"]["crop_y"]
                    
                # use OPMNIDAQ class calculation for number of scan steps to ensure consistency
                opmNIDAQ_custom.set_acquisition_params(
                    scan_type="mirror",
                    channel_states = [False,False,False,False,False],
                    image_mirror_step_size_um=image_mirror_step_um,
                    image_mirror_range_um=image_mirror_range_um,
                    laser_blanking=True,
                    exposure_ms=100.
                )
                n_scan_steps = opmNIDAQ_custom.n_scan_steps 
                
                daq_action_data = {
                    "DAQ" : {
                        "mode" : opm_mode,
                        "image_mirror_step_um" : image_mirror_step_um,
                        "image_mirror_range_um" : image_mirror_range_um,
                        "channel_states" : channel_states,
                        "channel_powers" : channel_powers,
                        "interleaved" : interleaved_acq,
                        "blanking" : laser_blanking, 
                    },
                    "Camera" : {
                        "exposure_channels" : channel_exposures_ms,
                        "camera_crop" : [
                            int(config["acq_config"]["camera_roi"]["center_x"] - int(config["acq_config"]["camera_roi"]["crop_x"]//2)),
                            int(config["acq_config"]["camera_roi"]["center_y"] - int(camera_crop_y//2)),
                            int(config["acq_config"]["camera_roi"]["crop_x"]),
                            int(camera_crop_y),
                        ]
                    }
                }
                
            elif "stage" in opm_mode:
                # Get the stage scan range
                stage_scan_range_um = np.round(
                    float(config["acq_config"][opm_mode+"_scan"]["stage_scan_range_um"]),0
                    )
                camera_crop_y = config["acq_config"]["camera_roi"]["crop_y"]
                      
                daq_action_data = {
                    "DAQ" : {
                        "mode" : opm_mode,
                        "stage_scan_range_um": stage_scan_range_um,
                        "channel_states" : channel_states,
                        "channel_powers" : channel_powers,
                        "interleaved" : interleaved_acq,
                        "blanking" : laser_blanking, 
                    },
                    "Camera" : {
                        "exposure_channels" : channel_exposures_ms,
                        "camera_crop" : [
                            int(config["acq_config"]["camera_roi"]["center_x"] - int(config["acq_config"]["camera_roi"]["crop_x"]//2)),
                            int(config["acq_config"]["camera_roi"]["center_y"] - int(camera_crop_y//2)),
                            int(config["acq_config"]["camera_roi"]["crop_x"]),
                            int(camera_crop_y),
                        ]
                    }
                }
            
            else:
                print(f"No valid opm mode selected: {opm_mode}")
                return            
            
            # Create DAQ event to run before acquiring each 'image'
            daq_event = MDAEvent(**DAQ_event.model_dump())
            daq_event.action.data.update(daq_action_data)
            
            #----------------------------------------------------------------#
            # Set the AO event data
            #----------------------------------------------------------------#
            if not("none" in ao_mode):
                # setup AO using values in the config widget, NOT the GUI
                AO_channel_states = [False] * len(channel_names) 
                AO_channel_powers = [0.] * len(channel_names)
                AO_active_channel_id = config["acq_config"]["AO"]["active_channel_id"]
                AO_camera_crop_y = int(calculate_projection_crop(config["acq_config"]["AO"]["image_mirror_range_um"]))
                AO_save_path = Path(str(config["AO-projection"]["optimize_now_path"])) / Path(f"{timestamp}_ao_optimizeNOW")
                
                # Set the active channel in the daq channel list
                for chan_idx, chan_str in enumerate(config["OPM"]["channel_ids"]):
                    if AO_active_channel_id==chan_str:
                        AO_channel_states[chan_idx] = True
                        AO_channel_powers[chan_idx] = config["acq_config"]["AO"]["active_channel_power"]
                        
                # check to make sure there exist a laser power > 0
                if sum(AO_channel_powers)==0:
                    print("All AO lasers set to 0!")
                    return
                
                # Define AO optimization action data   
                ao_action_data = {
                    "AO" : {
                        "channel_states": AO_channel_states,
                        "channel_powers" : AO_channel_powers,
                        "modal_delta": float(config["acq_config"]["AO"]["mode_delta"]),
                        "modal_alpha":float(config["acq_config"]["AO"]["mode_alpha"]),                        
                        "iterations": int(config["acq_config"]["AO"]["num_iterations"]),
                        "metric": str(config["acq_config"]["AO"]["metric"]),
                        "image_mirror_range_um" : config["acq_config"]["AO"]["image_mirror_range_um"],
                        "blanking": bool(True),
                        "apply_existing": bool(False),
                        "pos_idx": int(0),
                        "output_path":AO_save_path
                    },
                    "Camera" : {
                        "exposure_ms": config["acq_config"]["AO"]["exposure_ms"],
                        "camera_crop" : [
                            config["acq_config"]["camera_roi"]["center_x"] - int(config["acq_config"]["camera_roi"]["crop_x"]//2),
                            config["acq_config"]["camera_roi"]["center_y"] - int(AO_camera_crop_y//2),
                            config["Camera"]["camera_crop_x"],
                            AO_camera_crop_y
                        ]
                    }
                    
                }
                ao_optimization_event = MDAEvent(**AO_optimize_event.model_dump())
                ao_optimization_event.action.data.update(ao_action_data)
            
            #----------------------------------------------------------------#
            # Set the o2o3 AF event data
            #----------------------------------------------------------------#
            
            if not("none" in o2o3_mode):
                o2o3_action_data = {
                    "Camera" : {                    
                        "exposure_ms" : config["O2O3-autofocus"]["exposure_ms"],
                        "camera_crop" : [
                            config["Camera"]["roi_center_x"] - int(config["Camera"]["roi_crop_x"]//2),
                            config["Camera"]["roi_center_y"] - int(config["O2O3-autofocus"]["roi_crop_y"]//2),
                            config["Camera"]["roi_crop_x"],
                            config["O2O3-autofocus"]["roi_crop_y"]
                            ]
                        }
                    }
                
                o2o3_event = MDAEvent(**O2O3_af_event.model_dump())
                o2o3_event.action.data.update(o2o3_action_data)

            #----------------------------------------------------------------#
            # Set the fluidics event data
            #----------------------------------------------------------------#
            if not("none" in fluidics_mode):
                fp_action_data = {
                    "Fluidics": {
                        "total_rounds": int(fluidics_mode),
                        "current_round": int(0)
                    }
                }
                
                fp_event = MDAEvent(**FP_event.model_dump())
                fp_event.action.data.update(fp_action_data)
                
                
            #----------------------------------------------------------------#
            #----------------------------------------------------------------#
            # Compile mda positions from active tabs, extract the relevant portions for qi2lab-OPM
            # -  if the ao or af is optimize now, no values from the mda are used.
            # -  if grid plan is none, try on the stage positions sequence.
            # -  if the stage positions sequence is also none, use the current stage position.
            # -  if running with fluidics, set the number of timepoints as the number of rounds.
            # -  if no time plan is selected, assume single timepoint.
            #----------------------------------------------------------------#
            #----------------------------------------------------------------#
                               
            #----------------------------------------------------------------#
            # Generate time points
            
            # If running a fluidics experiment, the number of time points are the number of rounds.
            if not("none" in fluidics_mode):
                n_time_steps = int(fluidics_mode)
                time_interval = 0
                
            # If the time plan is selected, generate a timelapse 
            elif mda_time_plan is not None:
                n_time_steps = mda_time_plan["loops"]
                time_interval = mda_time_plan["interval"]
            
            # If no time plan is given, assume a single time point
            else:
                n_time_steps = 1
                time_interval = 0
                
            #----------------------------------------------------------------#
            # Generate stage positions
            stage_positions = []
            
            # If only the stage positions tab is selected, use the MDA positions tab
            if (mda_stage_positions is not None) and (mda_grid_plan is None):
                for stage_pos in mda_stage_positions:
                    stage_positions.append({
                                'x': float(stage_pos['x']),
                                'y': float(stage_pos['y']),
                                'z': float(stage_pos['z'])
                            })
                    
            # If the grid tab is selected, generate positions from the grid plan
            elif mda_grid_plan is not None:
                #----------------------------------------------------------------#
                # Get the z positions
                # - First check the z-plan
                # - Then use the singular stage position in the stage plan if it exist
                # - Lastly use the current stage position
                #----------------------------------------------------------------#
                
                # Check if multiple z plans are setup in MDA
                if mda_z_plan is not None:
                    max_z_pos = float(mda_z_plan["top"])
                    min_z_pos = float(mda_z_plan["bottom"])
                    step_z = (
                        (1-2*config["OPM"]["tile_overlap_perc"]/100) 
                        * float(mmc.getProperty("ImageCameraCrop","Label") ) 
                        * np.sin((np.pi/180.)*float(config["OPM"]["angle_deg"])) 
                        * mmc.getPixelSizeUm()
                        )
                    # reverse z-directon if needed
                    if min_z_pos > max_z_pos:
                        step_z = -1 * step_z
                    num_z_pos = int(np.ceil(np.abs((max_z_pos - min_z_pos) / step_z)))
                    
                # if no z plan is selected, check if only 1 position exists in the stage position plan
                elif (mda_stage_positions is not None):
                    if len([_p for _p in mda_stage_positions])==1:
                        temp = []
                        for stage_pos in mda_stage_positions:
                            temp.append({
                                'x': float(stage_pos['x']),
                                'y': float(stage_pos['y']),
                                'z': float(stage_pos['z'])
                            })
                        min_z_pos = temp[0]["z"]
                        step_z = 0
                        num_z_pos = 1
                
                # If no other z positions are set, resort to the current z position 
                else:
                    min_z_pos = mmc.getZPosition()
                    step_z = 0
                    num_z_pos = 1
                
                #----------------------------------------------------------------# 
                # Generate stage position list using the grid's generated positions
                
                # iterate through sequence to grab stage positions from grid plan
                stage_positions_array = []
                for event in sequence:
                    json_event = json.loads(event.model_dump_json())
                    stage_positions_array.append([
                        float(json_event['z_pos']),
                        float(json_event['y_pos']),
                        float(json_event['x_pos'])]
                    )
                stage_positions_array = np.asarray(stage_positions_array, dtype=np.float32)

                # Define our own snake pattern, assumes a 20% overlap.
                max_y_pos = np.max(stage_positions_array[:,1])
                min_y_pos = np.min(stage_positions_array[:,1])
                max_x_pos = np.max(stage_positions_array[:,2])
                min_x_pos = np.min(stage_positions_array[:,2])
                num_x_pos = int(np.ceil(np.abs(max_x_pos - min_x_pos) / (0.6 * config["Camera"]["camera_crop_x"]*mmc.getPixelSizeUm())))
                num_y_pos = int(np.ceil(np.abs(max_y_pos - min_y_pos) / (0.6 * image_mirror_range_um)))
                step_x = (max_x_pos - min_x_pos) / num_x_pos
                step_y = (max_y_pos - min_y_pos) / num_y_pos
                
                # Generate stage positions in a snake like pattern
                for x_pos in range(num_x_pos):
                    # Even rows (left to right)
                    if x_pos % 2 == 0:                          
                        y_range = range(num_y_pos)
                    # Odd rows (right to left)
                    else:  
                        y_range = range(num_y_pos - 1, -1, -1)
                    
                    # populate x positions
                    for y_pos in y_range:
                        for z_pos in range(num_z_pos):
                            stage_positions.append({
                                "x": float(np.round(min_x_pos + x_pos * step_x, 2)),
                                "y": float(np.round(min_y_pos + y_pos * step_y, 2)),
                                "z": float(np.round(min_z_pos + z_pos * step_z, 2))
                            })
                
            # if neither positions or grid plan are selected, use the current position
            elif (mda_grid_plan is None) and (mda_stage_positions is None):
                stage_positions.append({
                    'x': float(mmc.getXPosition()),
                    'y': float(mmc.getYPosition()),
                    'z': float(mmc.getZPosition())
                })
                # TODO: Is this how we want to handle no positions?
                # return
            
            # update the wfc mirror positions array shape
            n_stage_pos = len(stage_positions)
            opmAOmirror_local.n_positions = n_stage_pos
            
            #----------------------------------------------------------------#
            #----------------------------------------------------------------#
            # Create MDA event structure
            #----------------------------------------------------------------#
            #----------------------------------------------------------------#
        
            opm_events: list[MDAEvent] = []
            
            #----------------------------------------------------------------#
            # Else populate a {t, p, c, z} event sequence
            # - Run fliudics, if present
            # - Run O2O3 AF before time points or positions
            # - Run AO before each acq or positions
            # - Program the daq
            # - Acquire images
            #----------------------------------------------------------------#
            
            # Flags to help ensure sequence-able events are kept together 
            need_to_setup_DAQ = True
            need_to_setup_stage = True
            
            # Check if run AF at start only
            if "start" in o2o3_mode:
                opm_events.append(o2o3_event)
                
            # check if run AO at start only
            if "start" in ao_mode:
                opm_events.append(ao_optimization_event)
                
            #----------------------------------------------------------------#
            # setup nD mirror-based AO-OPM acquisition event structure
            #----------------------------------------------------------------#
            
            for time_idx in range(n_time_steps):
                # TODO Clarify how the acquisition should run. 
                # Right now, the first round is run manually in ESI, and then the imaging is setup afterwards. 
                # This offsets the number of rounds and if running fluidics, we acquire the first round images then run fluidics at the second time point.
                
                # Run fluidic at start of each timepoint
                if not("none" in fluidics_mode) and not(time_idx==0):
                    current_FP_event = MDAEvent(**fp_event.model_dump())
                    current_FP_event.action.data["Fluidics"]["round"] = int(time_idx)
                    opm_events.append(current_FP_event)
                
                # Run AF before each timepoint
                if "time" in o2o3_mode:
                    opm_events.append(o2o3_event)
                    
                # Iterate through stage positions
                for pos_idx in range(n_stage_pos):
                    if need_to_setup_stage:
                        stage_event = MDAEvent(
                            action=CustomAction(
                                name="Stage-Move",
                                data = {
                                    "Stage" : {
                                        "x_pos" : stage_positions[pos_idx]["x"],
                                        "y_pos" : stage_positions[pos_idx]["y"],
                                        "z_pos" : stage_positions[pos_idx]["z"],
                                    }
                                }
                            )
                        )
                        opm_events.append(stage_event)
                        
                        if n_stage_pos > 1:
                            need_to_setup_stage = True
                        else:
                            need_to_setup_stage = False
                    
                    # Run AF before acquiring current position data
                    if "xyz" in o2o3_mode:
                        opm_events.append(o2o3_event)
                        
                    # Run AO optimization before acquiring current position
                    if ("xyz" in ao_mode) and (time_idx == 0):
                        need_to_setup_DAQ = True
                        current_AO_event = MDAEvent(**ao_optimization_event.model_dump())
                        current_AO_event.action.data["AO"]["output_path"] = Path(output).parent / Path(f"pos_{pos_idx}_ao_optimize") # TODO
                        current_AO_event.action.data["AO"]["pos_idx"] = int(pos_idx)
                        current_AO_event.action.data["AO"]["apply_existing"] = False
                        opm_events.append(current_AO_event)
                        
                    # Apply mirror correction for this position if time_idx > 0
                    elif ("xyz" in ao_mode) and (time_idx > 0):
                        need_to_setup_DAQ = True
                        current_AO_event = MDAEvent(**ao_optimization_event.model_dump())
                        current_AO_event.action.data["AO"]["pos_idx"] = int(pos_idx)
                        current_AO_event.action.data["AO"]["apply_existing"] = True
                        opm_events.append(current_AO_event)
                        
                    # Finally, handle acquiring images. 
                    # These events are passed through to the normal MDAEngine and *should* be sequenced. 
                    if interleaved_acq:
                        # The DAQ waveform can be repeated for all time and spatial positions
                        if need_to_setup_DAQ:
                            need_to_setup_DAQ = True
                            opm_events.append(daq_event)
                            
                        # create camera events at current stage position
                        for scan_idx in range(n_scan_steps):
                            for chan_idx in range(n_active_channels):
                                image_event = MDAEvent(
                                    index=mappingproxy({
                                        "t": time_idx, 
                                        "p": pos_idx, 
                                        "c": chan_idx, 
                                        "z": scan_idx
                                    }),
                                    metadata = {
                                        "DAQ" : {
                                            "mode" : daq_mode,
                                            "image_mirror_step_um" : float(image_mirror_step_um), # This needs to be updated based on the opm mode, only passed for mirror scans
                                            "image_mirror_range_um" : float(image_mirror_range_um),
                                            "active_channels" : channel_states,
                                            "exposure_channels_ms": channel_exposures_ms,
                                            "interleaved" : interleaved_acq,
                                            "laser_powers" : channel_powers,
                                            "blanking" : laser_blanking,
                                            "current_channel" : active_channel_names[chan_idx]
                                        },
                                        "Camera" : {
                                            "exposure_ms" : float(channel_exposures_ms[chan_idx]),
                                            "camera_center_x" : config["acq_config"]["camera_roi"]["center_x"] - int(config["acq_config"]["camera_roi"]["crop_x"]//2),
                                            "camera_center_y" : config["acq_config"]["camera_roi"]["center_y"] - int(camera_crop_y//2),
                                            "camera_crop_x" : config["acq_config"]["camera_roi"]["crop_x"],
                                            "camera_crop_y" : int(camera_crop_y),
                                            "offset" : float(offset),
                                            "e_to_ADU": float(e_to_ADU)
                                        },
                                        "OPM" : {
                                            "angle_deg" : float(config["OPM"]["angle_deg"]),
                                            "camera_Zstage_orientation" : str(config["OPM"]["camera_Zstage_orientation"]),
                                            "camera_XYstage_orientation" : str(config["OPM"]["camera_XYstage_orientation"]),
                                            "camera_mirror_orientation" : str(config["OPM"]["camera_mirror_orientation"])
                                        },
                                        "Stage" : {
                                            "x_pos" : stage_positions[pos_idx]["x"],
                                            "y_pos" : stage_positions[pos_idx]["y"],
                                            "z_pos" : stage_positions[pos_idx]["z"],
                                        }
                                    }
                                )
                                opm_events.append(image_event)
                    else:
                        # Mirror scan each channel separately
                        for chan_idx, chan_bool in enumerate(channel_states):
                            temp_channels = [False] * len(channel_states)
                            temp_exposures = [0] * len(channel_exposures_ms)
                            if chan_bool:
                                if need_to_setup_DAQ:
                                    # The DAQ has to be updated for every channel
                                    need_to_setup_DAQ = True
                                    current_DAQ_event = MDAEvent(**daq_event.model_dump())
                                    temp_channels[chan_idx] = True
                                    temp_exposures[chan_idx] = channel_exposures_ms[chan_idx]
                                    current_DAQ_event.action.data["DAQ"]["active_channels"] = temp_channels
                                    current_DAQ_event.action.data["Camera"]["exposure_channels"] = temp_exposures 
                                    opm_events.append(current_DAQ_event)
                                    
                                for scan_idx in range(n_scan_steps):
                                    image_event = MDAEvent(
                                        index=mappingproxy({
                                            "t": time_idx, 
                                            "p": pos_idx, 
                                            "c": chan_idx, 
                                            "z": scan_idx
                                        }),
                                        metadata = {
                                            "DAQ" : {
                                                "mode" : daq_mode,
                                                "image_mirror_step_um" : float(image_mirror_step_um),
                                                "image_mirror_range_um" : float(image_mirror_range_um),
                                                "active_channels" : channel_states,
                                                "exposure_channels_ms": channel_exposures_ms,
                                                "interleaved" : interleaved_acq,
                                                "laser_powers" : channel_powers,
                                                "blanking" : laser_blanking,
                                                "current_channel" : config["OPM"]["channel_ids"][chan_idx]
                                            },
                                            "Camera" : {
                                                "exposure_ms" : float(channel_exposures_ms[chan_idx]),
                                                "camera_center_x" : config["acq_config"]["camera_roi"]["center_x"] - int(config["acq_config"]["camera_roi"]["crop_x"]//2),
                                                "camera_center_y" : config["acq_config"]["camera_roi"]["center_y"] - int(camera_crop_y//2),
                                                "camera_crop_x" : config["acq_config"]["camera_roi"]["crop_x"],
                                                "camera_crop_y" : int(camera_crop_y),
                                                "offset" : float(offset),
                                                "e_to_ADU": float(e_to_ADU)
                                            },
                                            "OPM" : {
                                                "angle_deg" : float(config["OPM"]["angle_deg"]),
                                                "camera_Zstage_orientation" : str(config["OPM"]["camera_Zstage_orientation"]),
                                                "camera_XYstage_orientation" : str(config["OPM"]["camera_XYstage_orientation"]),
                                                "camera_mirror_orientation" : str(config["OPM"]["camera_mirror_orientation"])
                                            },
                                            "Stage" : {
                                                "x_pos" : stage_positions[pos_idx]["x"],
                                                "y_pos" : stage_positions[pos_idx]["y"],
                                                "z_pos" : stage_positions[pos_idx]["z"],
                                            }
                                        }
                                    )
                                    opm_events.append(image_event)

            # Check if path ends if .zarr. If so, use Qi2lab OutputHandler
            if len(Path(output).suffixes) == 1 and Path(output).suffix == ".zarr":
                indice_sizes = {
                    't' : int(np.maximum(1,n_time_steps)),
                    'p' : int(np.maximum(1,n_stage_pos)),
                    'c' : int(np.maximum(1,n_active_channels)),
                    'z' : int(np.maximum(1,n_scan_steps))
                }
                handler = OPMMirrorHandler(
                    path=Path(output),
                    indice_sizes=indice_sizes,
                    delete_existing=True
                    )
                
                print(f"Running acquisition using Qi2lab handler: \n{indice_sizes}")
                
            # If not, use built-in handler based on suffix
            else:
                handler = Path(output)

            # run MDA with our event structure and modified tensorstore handler 
            mda_widget._mmc.run_mda(opm_events, output=handler)

            # tell AO mirror class where to save mirror information
            opmAOmirror_local.output_path = output.parents[0]

    # modify the method on the instance
    mda_widget.execute_mda = custom_execute_mda

    # Register the custom OPM MDA engine with mmc
    mmc.mda.set_engine(OPMEngine(mmc,config_path))

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
    # Because callbacks are blocking, our custom setup code is called before the preview mode starts. 
    mmc.events.continuousSequenceAcquisitionStarting.connect(setup_preview_mode_callback)

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    # -----------------End custom qi2lab code for running OPM control-----------------
    # --------------------------------------------------------------------------------
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