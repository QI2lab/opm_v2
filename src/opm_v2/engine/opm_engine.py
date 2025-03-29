"""OPM pymmcore-plus MDA Engine

TO DO: Fix init so we only have one instance of OPMNIDAQ, OPMAOMIRROR, and config is not global.

Change Log:
2025-02-07: New version that includes all possible modes
"""

from pymmcore_plus.mda import MDAEngine
from useq import MDAEvent, MDASequence, CustomAction
from typing import TYPE_CHECKING, Iterable
from opm_v2.hardware.OPMNIDAQ import OPMNIDAQ
from opm_v2.hardware.AOMirror import AOMirror
from opm_v2.hardware.ElveFlow import OB1Controller
from opm_v2.utils.elveflow_control import run_fluidic_program
from pymmcore_plus.metadata import (
    FrameMetaV1,
    PropertyValue,
    SummaryMetaV1,
    frame_metadata,
    summary_metadata
)
from numpy.typing import NDArray
from opm_v2.utils.sensorless_ao import run_ao_optimization
from opm_v2.utils.autofocus_remote_unit import manage_O3_focus
import json
from pathlib import Path
import numpy as np
from time import sleep

class OPMEngine(MDAEngine):
    def __init__(self, mmc, config_path: Path, use_hardware_sequencing: bool = True) -> None:

        super().__init__(mmc, use_hardware_sequencing)

        self.opmDAQ = OPMNIDAQ.instance()
        self.AOMirror = AOMirror.instance()
        self.execute_stage_scan = False

        with open(config_path, "r") as config_file:
            self._config = json.load(config_file)

    def setup_sequence(self, sequence: MDASequence) -> SummaryMetaV1 | None:
        """Setup state of system (hardware, etc.) before an MDA is run.

        This method is called once at the beginning of a sequence.
        (The sequence object needn't be used here if not necessary)
        """

        self._mmc.setCircularBufferMemoryFootprint(16000)
        super().setup_sequence(sequence)

    def setup_event(self, event: MDAEvent) -> None:
        """Prepare state of system (hardware, etc.) for `event`.

        This method is called before each event in the sequence. It is
        responsible for preparing the state of the system for the event.
        The engine should be in a state where it can call `exec_event`
        without any additional preparation.
        """
        if isinstance(event.action, CustomAction):
            action_name = event.action.name
            data_dict = event.action.data

            if action_name == "O2O3-autofocus":
                # Stop DAQ playback
                if self.opmDAQ.running():
                    self.opmDAQ.stop_waveform_playback()
                
                # Setup camera properties
                if not (int(data_dict["Camera"]["camera_crop"][3]) == self._mmc.getROI()[-1]):
                    current_roi = self._mmc.getROI()
                    self._mmc.clearROI()
                    self._mmc.waitForDevice(str(self._config["Camera"]["camera_id"]))
                    self._mmc.setROI(
                        data_dict["Camera"]["camera_crop"][0],
                        data_dict["Camera"]["camera_crop"][1],
                        data_dict["Camera"]["camera_crop"][2],
                        data_dict["Camera"]["camera_crop"][3],
                    )
                    self._mmc.waitForDevice(str(self._config["Camera"]["camera_id"]))
                self._mmc.setProperty(
                    str(self._config["Camera"]["camera_id"]), 
                    "Exposure", 
                    np.round(float(data_dict["Camera"]["exposure_ms"]),0)
                )
                self._mmc.waitForDevice(str(self._config["Camera"]["camera_id"]))
                
            elif action_name == "Stage-Move":
                """TO DO
                Check if speed is normal or setup for stage scan
                if stage scan, cache the current speed, set to higher speed,
                move stage to position, then set back to stage scan speed
                """
                self._mmc.setProperty(self._config["Stage"]["name"],"OnlySendSerialCommandOnChange","No")
                
                # Stage speed
                command = "SPEED Y=.1 X=0.1"
                self._mmc.setProperty(self._config["Stage"]["name"],"SerialCommand",command)

                # make sure ASI controller is ready for next command
                ready='B'
                while(ready!='N'):
                    print("in while loop setting stage move speed")
                    command = 'STATUS'
                    self._mmc.setProperty(self._config["Stage"]["name"],"SerialCommand",command)
                    print('cmd sent')
                    ready = self._mmc.getProperty(self._config["Stage"]["name"],"SerialResponse")
                    print("cmd rec.")
                    sleep(.5)
                self._mmc.setProperty(self._config["Stage"]["name"],"OnlySendSerialCommandOnChange","Yes")

                # Move stage to position
                print("about to move z")
                self._mmc.setPosition(np.round(float(data_dict["Stage"]["z_pos"]),2))
                self._mmc.waitForDevice(self._mmc.getFocusDevice())
                print("about to move xy")
                self._mmc.setXYPosition(
                    np.round(float(data_dict["Stage"]["x_pos"]),2),
                    np.round(float(data_dict["Stage"]["y_pos"]),2)
                )
                print("stage moved")
                dev =self._mmc.getXYStageDevice()
                print(f"XY dev: {dev}")
                
                current_x, current_y = self._mmc.getXYPosition()
                print(current_x,current_y)
                
                while not(np.round(current_x/10,0) == np.round(float(data_dict["Stage"]["x_pos"])/10,0)) or not(np.round(current_y/10,0) == np.round(float(data_dict["Stage"]["y_pos"])/10,0)):
                    sleep(.5)
                    current_x, current_y = self._mmc.getXYPosition()
                    print(current_x,current_y)
                
            elif action_name == "ASI-setupscan":
                # ensure commands are sent to the stage controller
                self._mmc.setProperty(self._config["Stage"]["name"],"OnlySendSerialCommandOnChange","No")

                # Setup PLC controller for TTL output to stage sync signal
                plcName = self._config["PLC"]["name"] # 'PLogic:E:36'
                propPosition = self._config["PLC"]["position"] # 'PointerPosition'
                propCellConfig = self._config["PLC"]["cellconfig"] # 'EditCellConfig'
                addrOutputBNC1 = int(self._config["PLC"]["pin"]) # 33 # BNC1 on the PLC front panel
                addrStageSync = int(self._config["PLC"]["signalid"]) # 46  # TTL5 on Tiger backplane = stage sync signal
                
                self._mmc.setProperty(plcName, propPosition, addrOutputBNC1)
                
                # make sure ASI controller is ready for next command
                ready='B'
                while(ready!='N'):
                    command = 'STATUS'
                    self._mmc.setProperty(self._config["Stage"]["name"],"SerialCommand",command)
                    ready = self._mmc.getProperty(self._config["Stage"]["name"],"SerialResponse")
                    sleep(.5)
                
                # Setup PLC controller to emit stage sync signal
                self._mmc.setProperty(plcName, propCellConfig, addrStageSync)

                # make sure ASI controller is ready for next command
                ready='B'
                while(ready!='N'):
                    command = 'STATUS'
                    self._mmc.setProperty(self._config["Stage"]["name"],"SerialCommand",command)
                    ready = self._mmc.getProperty(self._config["Stage"]["name"],"SerialResponse")
                    sleep(.5)

                # Set tile axis speed
                command = "SPEED Y=.1"
                self._mmc.setProperty(self._config["Stage"]["name"],"SerialCommand",command)

                # make sure ASI controller is ready for next command
                ready='B'
                while(ready!='N'):
                    command = 'STATUS'
                    self._mmc.setProperty(self._config["Stage"]["name"],"SerialCommand",command)
                    ready = self._mmc.getProperty(self._config["Stage"]["name"],"SerialResponse")
                    sleep(.5)

                # Set scan axis speed
                command = "SPEED X=" + str(data_dict["ASI"]["scan_axis_speed_mm_s"])
                self._mmc.setProperty(self._config["Stage"]["name"],"SerialCommand",command)

                # make sure ASI controller is ready for next command
                ready='B'
                while(ready!='N'):
                    command = 'STATUS'
                    self._mmc.setProperty(self._config["Stage"]["name"],"SerialCommand",command)
                    ready = self._mmc.getProperty(self._config["Stage"]["name"],"SerialResponse")
                    sleep(.5)

                # Set scan axis to true 1D scan with no backlash
                #command = "1SCAN X? Y=0 Z=9 F=0"
                #self._mmc.setProperty(self._config["Stage"]["name"],"SerialCommand",command)
                self._mmc.setProperty(self._mmc.getXYStageDevice(),"ScanPattern","Raster")
                self._mmc.setProperty(self._mmc.getXYStageDevice(),"ScanSlowAxis","Null (1D scan)")
                self._mmc.setProperty(self._mmc.getXYStageDevice(),"ScanFastAxis","1st axis")

                # # make sure ASI controller is ready for next command
                # ready='B'
                # while(ready!='N'):
                #     command = 'STATUS'
                #     self._mmc.setProperty(self._config["Stage"]["name"],"SerialCommand",command)
                #     ready = self._mmc.getProperty(self._config["Stage"]["name"],"SerialResponse")
                #     sleep(.5)

                # Set scan range and return speed (10% of max speed) for scan axis
                # command = '1SCANR X='+str(np.round(data_dict["ASI"]["scan_axis_start_mm"],2))+' Y='+str(np.round(data_dict["ASI"]["scan_axis_end_mm"],2))+' R=10'
                # print("scanning command", command)
                self._mmc.setProperty(self._mmc.getXYStageDevice(),"ScanFastAxisStartPosition(mm)",np.round(data_dict["ASI"]["scan_axis_start_mm"],2))
                self._mmc.setProperty(self._mmc.getXYStageDevice(),"ScanFastAxisStopPosition(mm)",np.round(data_dict["ASI"]["scan_axis_end_mm"],2))
                self._mmc.setProperty(self._mmc.getXYStageDevice(),"ScanSettlingTime(ms)",500)
                
                # make sure ASI controller is ready for next command
                # ready='B'
                # while(ready!='N'):
                #     command = 'STATUS'
                #     self._mmc.setProperty(self._config["Stage"]["name"],"SerialCommand",command)
                #     ready = self._mmc.getProperty(self._config["Stage"]["name"],"SerialResponse")
                #     sleep(.5)

                # put controller back into standard communication mode
                self._mmc.setProperty(self._config["Stage"]["name"],"OnlySendSerialCommandOnChange","Yes")
                
                # put camera into external START trigger mode
                self._mmc.setProperty(self._config["Camera"]["camera_id"],"Trigger","START")
                self._mmc.waitForDevice(str(self._config["Camera"]["camera_id"]))
                self._mmc.setProperty(self._config["Camera"]["camera_id"],"TriggerPolarity","POSITIVE")
                self._mmc.waitForDevice(str(self._config["Camera"]["camera_id"]))
                self._mmc.setProperty(self._config["Camera"]["camera_id"],"TRIGGER SOURCE","EXTERNAL")
                self._mmc.waitForDevice(str(self._config["Camera"]["camera_id"]))

                # ready for a stage scan
                self.execute_stage_scan = True
            
            elif action_name == "AO-projection":
                
                if data_dict["AO"]["apply_existing"]:
                    print("AO: Load existing mirror positions\n")
                    pass
                else:
                    # Stop and clear DAQ tasks to re-program
                    self.opmDAQ.stop_waveform_playback()
                    self.opmDAQ.clear_tasks()
                    
                    # Setup camera properties
                    if not (int(data_dict["Camera"]["camera_crop"][3]) == self._mmc.getROI()[-1]):
                        current_roi = self._mmc.getROI()
                        self._mmc.clearROI()
                        self._mmc.waitForDevice(str(self._config["Camera"]["camera_id"]))
                        self._mmc.setROI(
                            data_dict["Camera"]["camera_crop"][0],
                            data_dict["Camera"]["camera_crop"][1],
                            data_dict["Camera"]["camera_crop"][2],
                            data_dict["Camera"]["camera_crop"][3],
                        )
                    self._mmc.setProperty(
                        str(self._config["Camera"]["camera_id"]), 
                        "Exposure", 
                        np.round(float(data_dict["Camera"]["exposure_ms"]),0)
                    )
                    self._mmc.waitForDevice(str(self._config["Camera"]["camera_id"]))
                    
                    # Set laser powers
                    for chan_idx, chan_bool in enumerate(data_dict["AO"]["channel_states"]):
                        if chan_bool:
                            self._mmc.setProperty(
                                self._config["Lasers"]["name"],
                                str(self._config["Lasers"]["laser_names"][chan_idx]) + " - PowerSetpoint (%)",
                                float(data_dict["AO"]["channel_powers"][chan_idx])
                            )
                        else:
                            self._mmc.setProperty(
                                self._config["Lasers"]["name"],
                                str(self._config["Lasers"]["laser_names"][chan_idx]) + " - PowerSetpoint (%)",
                                0.0
                            )
            
            elif "DAQ" in action_name:
                # Stop and clear waveform tasks
                self.opmDAQ.stop_waveform_playback()
                self.opmDAQ.clear_tasks()
                
                # Setup camera properties
                if not (int(data_dict["Camera"]["camera_crop"][3]) == self._mmc.getROI()[-1]):
                    current_roi = self._mmc.getROI()
                    self._mmc.clearROI()
                    self._mmc.waitForDevice(str(self._config["Camera"]["camera_id"]))
                    self._mmc.setROI(
                        data_dict["Camera"]["camera_crop"][0],
                        data_dict["Camera"]["camera_crop"][1],
                        data_dict["Camera"]["camera_crop"][2],
                        data_dict["Camera"]["camera_crop"][3],
                    )
                    self._mmc.waitForDevice(str(self._config["Camera"]["camera_id"]))
                
                # Set laser powers
                for chan_idx, chan_bool in enumerate(data_dict["DAQ"]["channel_states"]):
                    if chan_bool:
                        self._mmc.setProperty(
                            self._config["Lasers"]["name"],
                            str(self._config["Lasers"]["laser_names"][chan_idx]) + " - PowerSetpoint (%)",
                            float(data_dict["DAQ"]["channel_powers"][chan_idx])
                        )
                        exposure_ms = np.round(float(data_dict["Camera"]["exposure_channels"][chan_idx]),0)
                    else:
                        self._mmc.setProperty(
                            self._config["Lasers"]["name"],
                            str(self._config["Lasers"]["laser_names"][chan_idx]) + " - PowerSetpoint (%)",
                            0.0
                        )
                        
                self._mmc.setProperty(
                    str(self._config["Camera"]["camera_id"]), 
                    "Exposure", 
                    exposure_ms
                )
                
                # Update daq waveform values and setup daq for playback
                if str(data_dict["DAQ"]["mode"]) == "stage":
                    self.opmDAQ.set_acquisition_params(
                        scan_type = str(data_dict["DAQ"]["mode"]),
                        channel_states = data_dict["DAQ"]["channel_states"],
                        laser_blanking = bool(data_dict["DAQ"]["blanking"]),
                        exposure_ms = exposure_ms
                    )
                elif str(data_dict["DAQ"]["mode"]) == "projection":
                    self.opmDAQ.set_acquisition_params(
                        scan_type = str(data_dict["DAQ"]["mode"]),
                        channel_states = data_dict["DAQ"]["channel_states"],
                        image_mirror_range_um = float(data_dict["DAQ"]["image_mirror_range_um"]),
                        laser_blanking = bool(data_dict["DAQ"]["blanking"]),
                        exposure_ms = exposure_ms
                    )
                elif str(data_dict["DAQ"]["mode"]) == "mirror":
                    self.opmDAQ.set_acquisition_params(
                        scan_type = str(data_dict["DAQ"]["mode"]),
                        channel_states = data_dict["DAQ"]["channel_states"],
                        image_mirror_step_size_um = float(data_dict["DAQ"]["image_mirror_step_um"]),
                        image_mirror_range_um = float(data_dict["DAQ"]["image_mirror_range_um"]),
                        laser_blanking = bool(data_dict["DAQ"]["blanking"]),
                        exposure_ms = exposure_ms
                    )
                self.opmDAQ.generate_waveforms()
                self.opmDAQ.program_daq_waveforms()
                
                # Wait for MM core
                self._mmc.waitForSystem()

            elif action_name == "Fluidics":
                print("Triggering ESI fluidics sequence\n")
                
        else:
            super().setup_event(event)

    def post_sequence_started(self, event):

        # execute stage scan if requested
        if self.execute_stage_scan:
                           
            print("sending SCAN")
            self._mmc.setProperty(self._config["Stage"]["name"],"SerialCommand","1SCAN")
            
            self.execute_stage_scan = False

    def exec_event(self, event: MDAEvent) -> Iterable[tuple[NDArray, MDAEvent, FrameMetaV1]]:
        """Execute `event`.

        This method is called after `setup_event` and is responsible for
        executing the event. The default assumption is to acquire an image,
        but more elaborate events will be possible.
        """
        if isinstance(event.action, CustomAction):
   
            action_name = event.action.name
            data_dict = event.action.data

            if action_name == "O2O3-autofocus":
                manage_O3_focus(self._config["O2O3-autofocus"]["O3_stage_name"], verbose=True)
                
            elif action_name == "AO-projection":               
                if data_dict["AO"]["apply_existing"]:
                    wfc_positions_to_use = self.AOMirror.wfc_positions_array[int(data_dict["AO"]["pos_idx"])]
                    self.AOMirror.set_mirror_positions(wfc_positions_to_use)
                else:
                    self.AOMirror.output_path = data_dict["AO"]["output_path"]
                    run_ao_optimization(
                        metric_to_use=data_dict["AO"]["mode"],
                        image_mirror_range_um=float(data_dict["AO"]["image_mirror_range_um"]),
                        exposure_ms=float(data_dict["Camera"]["exposure_ms"]),
                        channel_states=data_dict["AO"]["channel_states"],
                        num_iterations=int(data_dict["AO"]["iterations"]),
                        save_dir_path=data_dict["AO"]["output_path"],
                        verbose=True
                    )
                    self.AOMirror.wfc_positions_array[int(data_dict["AO"]["pos_idx"]),:] = self.AOMirror.current_positions.copy()
                    
            elif "DAQ" in action_name:
                self.opmDAQ.start_waveform_playback()
                
            elif action_name == "Fluidics":
                run_fluidic_program(True)
                
        else:
            result = super().exec_event(event)
            return result

    def teardown_event(self, event):
        if isinstance(event.action, CustomAction):
            self._mmc.clearCircularBuffer()
        super().teardown_event(event)

    def teardown_sequence(self, sequence: MDASequence) -> None:
        print("Acq finished, tearing down.")
        # print(self._mmc.getRemainingImageCount())
        
        # Shut down DAQ
        self.opmDAQ.stop_waveform_playback()
        self.opmDAQ.clear_tasks()
        self.opmDAQ.reset()

        print("Daq reset")

        # Put camera back into internal mode
        self._mmc.setProperty(self._config["Camera"]["camera_id"],"TriggerPolarity","POSITIVE")
        self._mmc.waitForDevice(str(self._config["Camera"]["camera_id"]))
        self._mmc.setProperty(self._config["Camera"]["camera_id"],"TRIGGER SOURCE","INTERNAL")
        self._mmc.waitForDevice(str(self._config["Camera"]["camera_id"]))

        self._mmc.setProperty(self._mmc.getXYStageDevice(),"MotorSpeedX-S(mm/s)",0.1)
        self._mmc.setProperty(self._mmc.getXYStageDevice(),"MotorSpeedY-S(mm/s)",0.1)

        # Reset the stage speed
        # ready='B'
        # while(ready!='N'):
        #     command = 'STATUS'
        #     self._mmc.setProperty(self._config["Stage"]["name"],"SerialCommand",command)
        #     ready = self._mmc.getProperty(self._config["Stage"]["name"],"SerialResponse")
        #     sleep(.5)
            
        # command = "SPEED Y=.1 X=.1"
        # self._mmc.setProperty(self._config["Stage"]["name"],"SerialCommand",command)

        # ready='B'
        # while(ready!='N'):
        #     command = 'STATUS'
        #     self._mmc.setProperty(self._config["Stage"]["name"],"SerialCommand",command)
        #     ready = self._mmc.getProperty(self._config["Stage"]["name"],"SerialResponse")
        #     sleep(.5)       
            
        # print("stage position was set")
        
        # Set all lasers to zero emission
        for laser in self._config["Lasers"]["laser_names"]:
            self._mmc.setProperty(
                self._config["Lasers"]["name"],
                laser + " - PowerSetpoint (%)",
                0.0
            )
            
        # save mirror positions array
        self.AOMirror.save_wfc_positions_array()
        
        self._mmc.clearCircularBuffer()

        super().teardown_sequence(sequence)