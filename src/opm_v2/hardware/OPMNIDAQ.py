#!/usr/bin/python
"""
qi2lab-OPM interface class to run NIDAQ with camera as master using PyDAQMx.

Authors:
Peter Brown
Franky Djutanta
Steven Sheppard
Douglas Shepherd

contact: douglas.shepherd@asu.edu

Change Log:
2025/02: Refactor
2025/02: Projection mode
2021/12: Initial version

"""

import PyDAQmx as daqmx
import ctypes as ct
import numpy as np
from typing import Sequence
import warnings

# Suppress warning for stoping AO waveform before all samples are sent
warnings.filterwarnings("ignore", category=daqmx.DAQmxFunctions.StoppedBeforeDoneWarning)
_instance_daq = None

class OPMNIDAQ:
    """Class to control NIDAQ for the qi2lab-OPM.
    
    This class is specialized to run an OPM where the camera provides the external timing. The current config expects there to be 
    an analog output for a image scanning galvo mirror, an analog output for a projection galvo mirror, multiple digital output to laser control,
    and appropriate wiring for TTL triggering from the camera into a digital input.
    
    There are multiple modes possible. 
    
    The first three modes use the camera for all timing:
    
    scan_type = "2d"
        This mode snaps an image for each of the requested laser in sequential order, with both mirrors at the neutral position. 
        The images are a single oblique plane each. The "EXPOSURE OUT" timing of the camera controls when the lasers are active, if laser blanking
        is "True".
    scan_type = "mirror"
        This mode snaps image for each of the requested lasers in sequential order at "N" image galvo mirror positions to form an oblique volume. 
        The "EXPOSURE TIMING" of the camera controls when the lasers are active, if laser blanking is "True". The changing edge of the "EXPOSURE OUT"
        timing of the camera controls when the image galvo mirror advances to the next discrete position.
    scan_type = "stage"
        This mode is similar to "2d", except the camera now does not begin free running until the ASI stage controller sends a pulse that the scanning
        stage has passed the start position and is up to speed.
        
    The last mode uses the camera "EXPOSURE OUT" to initiate waveform playback at a set speed.
    
    scan_type = "projection"
        This mode snaps an image where both the image and projection galvo mirrors sweep through a set of voltages during a SINGLE camera exposure.
        When the timing is correct for this mode, the resulting image is a Z sum projection of the swept volume, with a 1/cos(30 degrees) shrink factor
        applied.
    
    Parameters
    ----------
    scan_type: str, default = "2d"
        scan type
    exposure_ms: float, default = 50.0
        exposure time in milliseconds
    laser_blanking: bool, default = True
        synchronize laser output to the camera rolling shutter being fully open. For fast imaging, this likely needs to be `False`
    image_mirror_calibration: float, default = .043
        image mirror calibration in V/um
    projection_mirror_calibration: float, default = .0052
        projection mirror calibration in V/um
    image_mirror_step_size_um: float, default = 0.4
        image mirror step size in um. default value is Nyquist sampled for our OPM.
    """

    @classmethod
    def instance(cls) -> 'OPMNIDAQ':
        """Return the global singleton instance of `OPMNIDAQ`.

        """
        global _instance_daq
        if _instance_daq is None:
            _instance_daq = cls()
        return _instance_daq

    def __init__(
        self,
        name = "Dev1",
        scan_type: str = "2d",
        exposure_ms: float = 50.,
        laser_blanking: bool = True,
        image_mirror_calibration: float = .0433,
        projection_mirror_calibration: float = .0052,
        image_mirror_neutral_v: float = 0.0,
        projection_mirror_neutral_v: float = 0.0,
        image_mirror_step_size_um = 0.4,
        verbose: bool=False
    ):
        
        # Set the first instance of this class as the global singleton
        global _instance_daq
        if _instance_daq is None:
            _instance_daq = self
        
        self.scan_type = scan_type
        self.exposure_ms = exposure_ms
        self.laser_blanking = laser_blanking
        self.image_mirror_calibration = image_mirror_calibration
        self.projection_mirror_calibration = projection_mirror_calibration
        self.image_mirror_step_size_um = image_mirror_step_size_um
        self.verbose = verbose
        
        # Define waveform generation parameters
        self._daq_sample_rate_hz = 10000
        self._do_ind = [0,1,2,3,4]
        self._channel_states = [False, False, False, False, False]
        self._active_channels_indices = None
        self._n_active_channels = 0
        self._num_do_channels = 8
        self._do_waveform = np.zeros((self._num_do_channels),dtype=np.uint8)
        self._ao_waveform = [np.zeros(1), np.zeros(1)]
        self._ao_neutral_positions = [image_mirror_neutral_v, projection_mirror_neutral_v]
        
        # Configure hardware pin addresses.
        self._dev_name = name
        # Program DO port0 using 8-bit waveforms
        self._address_channel_do = "/Dev1/port0/line0:7" # laser lines 0:4
        self._address_ao_mirrors = [
            "/Dev1/ao0", # image scanning galvo
            "/Dev1/ao1" # projection scanning galvo
        ]
        self._channel_di_trigger_from_camera = "/Dev1/PFI0" # camera trig port 0
        self._channel_di_start_trigger = "/Dev1/PFI1" # Empty PFI pin
        self._channel_di_change_trigger = "/Dev1/PFI2" # Empty PFI pin
        self._channel_ao_start_trigger = "/Dev1/PFI3" # Route channel_do_trigger
        
        # task handles
        self._task_do = None
        self._task_ao = None
        self._task_di = None

        # daq running
        self._running = False
        
    @property
    def scan_type(self) -> str:
        """Scan type.
        
        Returns
        -------
        scan_type: str
            scan type. One of "2d", "mirror", "projection", or "stage"
        """
        
        return getattr(self,"_scan_type",None)
    
    @scan_type.setter
    def scan_type(self, value: str):
        """Set the scan type.
        
        Parameters
        ----------
        value: str
            scan type. One of "2d", "mirror", "projection", or "stage
        """
        
        if not hasattr(self, "_scan_type") or self._scan_type is None:
            self._scan_type = value
        else:
            self._scan_type = value
            
    @property
    def exposure_ms(self) -> float:
        """Exposure time in milliseconds.
        
        Returns
        -------
        exposure_ms: float
            exposure time in units of milliseconds.
        """
        
        return getattr(self,"_exposure_s",None) * 1000.
    
    @exposure_ms.setter
    def exposure_ms(self, value: float):
        """Set the exposure time in milliseconds.
        
        Parameters
        ----------
        value: float
            exposure time in units of milliseconds.
        """
        
        if not hasattr(self, "_exposure_s") or self._exposure_s is None:
            self._exposure_s = value / 1000.
        else:
            self._exposure_s = value / 1000.
            
    @property
    def laser_blanking(self) -> bool:
        """Laser blanking state.
        
        Returns
        -------
        laser_blanking: bool
            Laser blanking state: Active (True) or Inactive (False) 
        """
        
        return getattr(self,"_laser_blanking",None)
    
    @laser_blanking.setter
    def laser_blanking(self, value: bool):
        """Set the exposure time in milliseconds.
        
        Parameters
        ----------
        value: bool
            Laser blanking state: Active (True) or Inactive (False) 
        """
        
        if not hasattr(self, "_laser_blanking") or self._laser_blanking is None:
            self._laser_blanking = value
        else:
            self._laser_blanking = value
            
    @property
    def image_mirror_calibration(self) -> float:
        """Image mirror calibration (V/um).
        
        Returns
        -------
        image_mirror_calibration: float
            Image mirror calibration in V/um
        """
        
        return getattr(self,"_image_mirror_calibration",None)
    
    @image_mirror_calibration.setter
    def image_mirror_calibration(self, value: float):
        """Set the image mirror calibration (V/um).
        
        Parameters
        ----------
        value: float
            Image mirror calibration in V/um
        """
        
        if not hasattr(self, "_image_mirror_calibration") or self._image_mirror_calibration is None:
            self._image_mirror_calibration = value
        else:
            self._image_mirror_calibration = value
            
    @property
    def projection_mirror_calibration(self) -> float:
        """Projection mirror calibration (V/um).
        
        Returns
        -------
        projection_mirror_calibration: float
            Projection mirror calibration in V/um
        """
        
        return getattr(self,"_projection_mirror_calibration",None)
    
    @projection_mirror_calibration.setter
    def projection_mirror_calibration(self, value: float):
        """Set the projection mirror calibration (V/um).
        
        Parameters
        ----------
        value: float
            Projection mirror calibration in V/um
        """
        
        if not hasattr(self, "_projection_mirror_calibration") or self._projection_mirror_calibration is None:
            self._projection_mirror_calibration = value
        else:
            self._projection_mirror_calibration = value
            
    @property
    def image_mirror_step_size_um(self) -> float:
        """Image mirror step size in microns.
        
        This is the lateral footprint along the coverslip.
        
        Returns
        -------
        image_mirror_step_size_um: float
            Image mirror step size in microns.
        """
        
        return getattr(self,"_image_mirror_step_size_um",None)
    
    @image_mirror_step_size_um.setter
    def image_mirror_step_size_um(self, value: float):
        """Set the image mirror step size in microns.
        
        Parameters
        ----------
        value: float
            Image mirror step size in microns.
        """
        
        if not hasattr(self, "_image_mirror_step_size_um") or self._image_mirror_step_size_um is None:
            self._image_mirror_step_size_um = value
        else:
            self._image_mirror_step_size_um = value

        self._image_axis_step_volts = self._image_mirror_step_size_um * self._image_mirror_calibration
        # temp = self.image_mirror_range_um.copy()
        # self.image_mirror_range_um = temp
            
    @property
    def image_mirror_range_um(self) -> float:
        """Image mirror sweep in microns.
        
        This is the lateral footprint of sweep, symmetric around the zero point, along the coverslip.
        
        Returns
        -------
        image_mirror_range_um: float
            Image mirror sweep in microns.
        """
        
        return getattr(self,"_image_mirror_range_um",None)
    
    @image_mirror_range_um.setter
    def image_mirror_range_um(self, value: float):
        """Set the image mirror sweep in microns.
        
        Parameters
        ----------
        value: float
            Image mirror sweep in microns.
        """
        
        if not hasattr(self, "_image_mirror_range_um") or self._image_mirror_range_um is None:
            self._image_mirror_range_um = value
        else:
            self._image_mirror_range_um = value

        # setup image galvo mirror
        self._image_mirror_min_volt = -(self._image_mirror_range_um * self._image_mirror_calibration) / 2. + self._ao_neutral_positions[0] # unit: volts
        self._image_axis_range_volts = self._image_mirror_range_um * self._image_mirror_calibration
        self._image_scan_steps = int(np.rint(self._image_axis_range_volts / self._image_axis_step_volts)) # galvo steps

        # setup projection galvo mirror
        self._projection_scan_range_volts = self._image_mirror_range_um * self._projection_mirror_calibration
        
    @property
    def n_scan_steps(self) -> int:
        """Image mirror scan steps.
        
        Returns
        -------
        n_scan_steps: int
            Number of scan steps for a mirror sweep.
            
        """
        
        return int(getattr(self,"_image_scan_steps",None))
            
    @property
    def channel_states(self) -> Sequence:
        """Active channel states.
        
        Returns
        -------
        channel_states: Sequence
            Boolean array of active laser lines
        """
        
        return getattr(self,"_channel_states",None)
    
    @channel_states.setter
    def channel_states(self, value: Sequence):
        """Set the active channel states.
        
        Parameters
        ----------
        value: Sequence
            Boolean array of active laser lines.
        """
        
        if not hasattr(self, "_channel_states") or self._channel_states is None:
            self._channel_states = value
        else:
            self._channel_states = value
        
        self._active_channels_indices = [ind for ind, st in zip(self._do_ind, self._channel_states) if st]
        self._n_active_channels = len(self._active_channels_indices)

    def running(self) -> bool:
        """Returns true is playback is active, false otherwise"""
        
        return self._running
    
    #-----------------------------------------------------#
    # Helper function for setting daq values
    #-----------------------------------------------------#
        
    def set_acquisition_params(
        self,
        scan_type: str = None,
        channel_states: Sequence[bool] = None,
        image_mirror_step_size_um: float = None,
        image_mirror_range_um: float = None,
        laser_blanking: bool = None,
        exposure_ms: float = None
    ):
        """Convenience function to set the DAQ up for an acquisition.
        
        
        Parameters
        ----------
        scan_type: str
            scan type. One of "2d", "mirror", "projection", or "stage
        channel_states: Sequence[bool]
            channel states, in order of [405nm, 488nm, 561nm, 637nm, 730nm].
        image_mirror_step_size_um: float
            Image mirror step size in microns. This is the lateral footprint along the coverslip.
        image_mirror_range_um: float
            Image mirror sweep in microns.
        laser_blanking: bool
            Laser blanking state: Active (True) or Inactive (False) 
        exposure_ms: float
            camera exposure time in milliseconds.
        """

        if scan_type:
            self.scan_type=scan_type
        if channel_states:
            self.channel_states = channel_states
        if laser_blanking:
            self.laser_blanking = laser_blanking
        if exposure_ms:
            self.exposure_ms = exposure_ms
        if image_mirror_step_size_um:
            self.image_mirror_step_size_um = image_mirror_step_size_um
        if image_mirror_range_um:
            self.image_mirror_range_um = image_mirror_range_um
            
    #-----------------------------------------------------#
    # Reset device methods
    #-----------------------------------------------------#
    
    def reset(self):
        """Reset the device."""

        daqmx.DAQmxResetDevice(self._dev_name)
        self.reset_ao_channels()
        self.reset_do_channels()

    def reset_ao_channels(self):
        """Stops any waveforms and deletes tasks, set analog lines to the mirror's neutral positions
        """
        self.clear_tasks()
        
        _ao_waveform = np.column_stack((np.full(2, self._ao_neutral_positions[0]),
                                        np.full(2, self._ao_neutral_positions[1])))

        samples_per_ch_ct = ct.c_int32()
        try:
            with daqmx.Task("ResetAO") as _task:
                _task.CreateAOVoltageChan(
                    self._address_ao_mirrors[0], 
                    "reset_ao0", 
                    -6.0, 6.0, daqmx.DAQmx_Val_Volts, None
                )
                _task.CreateAOVoltageChan(
                    self._address_ao_mirrors[1], 
                    "reset_ao1",
                    -1.0, 1.0, daqmx.DAQmx_Val_Volts, None
                )
                _task.WriteAnalogF64(
                    1, 
                    True, 
                    1, 
                    daqmx.DAQmx_Val_GroupByScanNumber, 
                    _ao_waveform, 
                    ct.byref(samples_per_ch_ct), None
                )
                _task.StopTask()
                _task.ClearTask()
        except (daqmx.DAQmxFunctions.InvalidTaskError, AttributeError):
            pass
      
    def reset_do_channels(self):
        """Stops any running waveforms and deletes task handlers. Then sets DO lines to 0."""
        
        self.clear_tasks()
        
        try:
            with daqmx.Task("ResetDO") as _task:
                _task.CreateDOChan(
                    self._address_channel_do,
                    "reset_do", 
                    daqmx.DAQmx_Val_ChanForAllLines
                )
                _task.WriteDigitalLines(
                    1, 
                    True, 
                    1.0, 
                    daqmx.DAQmx_Val_GroupByChannel, 
                    np.zeros((1, len(self._address_channel_do)), dtype=np.uint8),
                    None, 
                    None
                )    
                _task.StopTask()
                _task.ClearTask()
        except (daqmx.DAQmxFunctions.InvalidTaskError, AttributeError):
            pass
    
    #-----------------------------------------------------#
    # Methods for generating / starting / stopping waveform playback 
    #-----------------------------------------------------#
    
    def generate_waveforms(self):
        """Generate waveforms necessary to capture 1 'scan_type'.
        
           Waveforms run after receiving change detection from camera trigger.
           Possible 'scan_type':
           - '2d' for a 2d scan is a single image mirror position.
           - 'stage' for a 2d scan during a constant speed stage movement.
           - 'projection' for a projection scan is a linear ramp for both the image and projection mirrors.
           - 'mirror' for a mirror scan is n_scan_step frames x n_do channels x camera_roi
        """        

        if self.scan_type == '2d':
            """Only fire the active channel lasers, keep the mirrors in their neutral positions."""

            #-----------------------------------------------------#
            # The DO channel changes with changes in camera's trigger output,
            # There are 2 time steps per frame, except for first frame plus one final frame to reset voltage
            self.samples_per_do_ch = (2*self._n_active_channels - 1) + 1
            self.samples_per_do_ch = (2*self._n_active_channels - 1) + 1
 
            # Generate values for DO
            _do_waveform = np.zeros((self.samples_per_do_ch, self._num_do_channels), dtype=np.uint8)
            for ii, ind in enumerate(self._active_channels_indices):
                if self._laser_blanking:
                    _do_waveform[2*ii::2*self._n_active_channels, ind] = 1
                else:
                    _do_waveform[:,int(ind)] = 1
            
            if self._laser_blanking:
                _do_waveform[-1, :] = 0
            
            #-----------------------------------------------------#
            # Create ao waveform, keeping the mirrors in their neutral positions
            # In 2d mode, the first time point gets set.
            _ao_waveform = np.zeros((1, 2))
            _ao_waveform[:, 0] = self._ao_neutral_positions[0]
            _ao_waveform[:, 1] = self._ao_neutral_positions[1]
            
        elif self.scan_type == "projection":
            """Perform projection scan.
            
               - Fire active lasers
               - Synchronize image scanning mirror and projection mirror to rolling shutter.
               - Capture one frame per active channel.
               - Start camera in light sheet mode, apply linear ramp to each mirror?
            """
            #-----------------------------------------------------#
            # The DO channel changes with changes in camera's trigger output,
            # There are 2 time steps per frame, except for first frame plus one final frame to reset voltage
            self.samples_per_do_ch = (2*self._n_active_channels - 1) + 1
            self.samples_per_do_ch = (2*self._n_active_channels - 1) + 1
 
            # Generate values for DO
            _do_waveform = np.zeros((self.samples_per_do_ch, self._num_do_channels), dtype=np.uint8)
            for ii, ind in enumerate(self._active_channels_indices):
                if self._laser_blanking:
                    _do_waveform[2*ii::2*self._n_active_channels, ind] = 1
                else:
                    _do_waveform[:,int(ind)] = 1
            
            if self._laser_blanking:
                _do_waveform[-1, :] = 0
            
            #-----------------------------------------------------#
            # Create ao waveform, scan the image mirror and projection mirror voltages.
            # This array is written for both AO channels and runs at the camera di rising edge
            n_voltage_steps = int(self._exposure_s * self._daq_sample_rate_hz) + 1
            # n_return_steps = 1
                        
            # Generate projection mirror linear ramp
            self.proj_mirror_min_volt = self._projection_scan_range_volts/2
            self.proj_mirror_max_volt = - self._projection_scan_range_volts/2
            proj_mirror_sweep_volts = np.linspace(self.proj_mirror_min_volt, self.proj_mirror_max_volt, n_voltage_steps-1)
            # proj_mirror_return_volts = np.linspace(self.proj_mirror_max_volt, self.proj_mirror_min_volt, n_return_steps)
            # proj_mirror_volts = np.concatenate(
            #     (proj_mirror_sweep_volts,
            #      proj_mirror_return_volts)
            # )
            if self.verbose:
                print(self.proj_mirror_min_volt)
                print(self.proj_mirror_max_volt)
            
            # Generate image scanning mirror voltage steps
            image_mirror_max_volts = self._image_mirror_min_volt + self._image_axis_range_volts
            image_mirror_sweep_volts = np.linspace(self._image_mirror_min_volt, image_mirror_max_volts, n_voltage_steps-1)
            # image_mirror_return_volts = np.linspace(image_mirror_max_volts, self._image_mirror_min_volt, n_return_steps)
            # image_mirror_volts = np.concatenate(
            #     (image_mirror_sweep_volts,
            #      image_mirror_return_volts)
            # ) 
            # Combine image and proj mirror waveforms to program to daq
            # _ao_waveform = np.column_stack(
            #     (image_mirror_volts,
            #      proj_mirror_volts)
            # )
            
            # Create an empty AO waveform
            _ao_waveform = np.zeros((n_voltage_steps,2))
            self.samples_per_ao_ch = _ao_waveform.shape[0]
            
            # Set the last time point (when exp is off) to the first mirror positions.
            _ao_waveform[:-1, 0] = image_mirror_sweep_volts
            _ao_waveform[:-1, 1] = proj_mirror_sweep_volts
            
            _ao_waveform[-1, 0] = image_mirror_sweep_volts[0]
            _ao_waveform[-1, 1] = proj_mirror_sweep_volts[0]          

        elif self.scan_type == 'mirror':
            """Fire active lasers, advance image scanning galvo in a linear ramp,
               hold the projection galvo in it's neutral position."""
            
            #-----------------------------------------------------#
            # The DO channel changes with changes in camera's trigger output,
            # There are 2 time steps per frame, except for first frame plus one final frame to reset voltage
            # Collect one frame for each scan position
            n_voltage_steps = int(self._image_scan_steps)
            self.samples_per_do_ch = 2*n_voltage_steps*self._n_active_channels
            self.samples_per_do_ch = 2*n_voltage_steps*self._n_active_channels
            
            # Generate values for DO
            _do_waveform = np.zeros((self.samples_per_do_ch, self._num_do_channels), dtype=np.uint8)
            for ii, ind in enumerate(self._active_channels_indices):
                # Turn laser on in order for each image position
                if self._laser_blanking:
                    _do_waveform[2*ii::2*self._n_active_channels, ind] = 1
                else:
                    _do_waveform[:,int(ind)] = 1
            
            if self._laser_blanking:
                _do_waveform[-1, :] = 0
                
            #-----------------------------------------------------#
            # Create ao waveform, scan the image mirror voltage, keep the projection mirror neutral            
            # This array is written for both AO channels
            _ao_waveform = np.zeros((self.samples_per_do_ch, 2))
            
            # Generate image scanning mirror voltage steps
            max_volt = self._image_mirror_min_volt + self._image_axis_range_volts
            scan_mirror_volts = np.linspace(self._image_mirror_min_volt, max_volt, n_voltage_steps)
            
            # Set the last time point (when exp is off) to the first mirror positions.
            _ao_waveform[0:2*self._n_active_channels - 1, 0] = scan_mirror_volts[0]
            
            if len(scan_mirror_volts) > 1:
                # (2 * # active channels) voltage values for all other frames
                _ao_waveform[2*self._n_active_channels - 1:-1, 0] = np.kron(scan_mirror_volts[1:], np.ones(2 * self._n_active_channels))
            
            # set back to initial value at end
            _ao_waveform[-1,0] = scan_mirror_volts[0]
            _ao_waveform[:,1] = self._ao_neutral_positions[1]
        
        elif self.scan_type == 'stage':
            """Only fire the active channel lasers,keep the mirrors in their neutral positions"""

            #-----------------------------------------------------#
            # setup digital trigger buffer on DAQ
            self.samples_per_do_ch = 2 * int(self._n_active_channels)
            self.samples_per_do_ch = 2 * int(self._n_active_channels)

            # create DAQ pattern for laser strobing controlled via rolling shutter
            _do_waveform = np.zeros((self.samples_per_do_ch, self._num_do_channels), dtype=np.uint8)
            for ii, ind in enumerate(self._active_channels_indices):
                if self._laser_blanking:
                    _do_waveform[2*ii::2*int(self._n_active_channels), int(ind)] = 1
                else:
                    _do_waveform[:,int(ind)] = 1

            if self._laser_blanking:
                _do_waveform[-1, :] = 0
            
            #-----------------------------------------------------#
            # Create ao waveform, keeping the mirrors in their neutral positions
            # In stage scan mode, only the first time point gets set.
            _ao_waveform = np.zeros((1, 2))
            _ao_waveform[:, 0] = self._ao_neutral_positions[0]
            _ao_waveform[:, 1] = self._ao_neutral_positions[1]
               
        # Update daq waveforms
        self._do_waveform = _do_waveform
        self._ao_waveform = _ao_waveform
        
    def program_daq_waveforms(self):
        """Create DAQ tasks for synchronizing camera output triggers to lasers and galvo mirrors."""
        
        try:
            #-------------------------------------------------#
            # Create DI trigger from camera task
            # This should only be done at startup or after a reset occurs
            if self._task_di is None:
                self._task_di = daqmx.Task("TaskDI")
                self._task_di.CreateDIChan(
                    self._channel_di_trigger_from_camera,
                    "DI_CameraTrigger", 
                    daqmx.DAQmx_Val_ChanForAllLines
                )
                
                # Configure change detection timing (from wave generator)
                self._task_di.CfgInputBuffer(0)    # must be enforced for change-detection timing, i.e no buffer
                self._task_di.CfgChangeDetectionTiming(
                    self._channel_di_trigger_from_camera, 
                    self._channel_di_trigger_from_camera, 
                    daqmx.DAQmx_Val_ContSamps, 
                    0
                )

                # Set where the starting trigger 
                self._task_di.CfgDigEdgeStartTrig(self._channel_di_trigger_from_camera, daqmx.DAQmx_Val_Rising)
                
                # Export DI signal to unused PFI pins, for clock and start
                self._task_di.ExportSignal(daqmx.DAQmx_Val_ChangeDetectionEvent, self._channel_di_change_trigger)
                self._task_di.ExportSignal(daqmx.DAQmx_Val_StartTrigger, self._channel_di_start_trigger)
        
            #-------------------------------------------------#
            # Create DO laser control tasks
            if self._task_do is None:
                self._task_do = daqmx.Task("TaskDO")
                self._task_do.CreateDOChan(
                    self._address_channel_do,
                    "DO_LaserControl", 
                    daqmx.DAQmx_Val_ChanForAllLines
                )
                
                # Configure change timing from camera trigger task
                self._task_do.CfgSampClkTiming(
                    self._channel_di_change_trigger, 
                    self._daq_sample_rate_hz,
                    daqmx.DAQmx_Val_Rising,
                    daqmx.DAQmx_Val_ContSamps, 
                    self.samples_per_do_ch
                )
            

            # Write the output waveform
            samples_per_ch_ct_digital = ct.c_int32()
            self._task_do.WriteDigitalLines(
                self.samples_per_do_ch, 
                False, 
                10.0, 
                daqmx.DAQmx_Val_GroupByChannel, 
                self._do_waveform.astype(np.uint8), 
                ct.byref(samples_per_ch_ct_digital), 
                None
            )

            #-------------------------------------------------#
            # Create AO tasks, dependent on acquisition scan mode
            # first, set the scan and projection galvo to the initial point if it is not already
            samples_per_ch_ct = ct.c_int32()
            with daqmx.Task("TaskInitAO") as _task:
                # Create a 2d array that sets the initial AO voltage to the start of the scan.
                initial_ao_waveform = np.column_stack(
                    (np.full(2, self._ao_waveform[0,0]),
                    np.full(2, self._ao_waveform[0,1]))
                )
                _task.CreateAOVoltageChan(
                    self._address_ao_mirrors[0], 
                    "initialize_ao0", 
                    -6.0, 
                    6.0, 
                    daqmx.DAQmx_Val_Volts, 
                    None
                    )
                _task.CreateAOVoltageChan(
                    self._address_ao_mirrors[1], 
                    "initialize_ao1",
                    -1.0, 
                    1.0, 
                    daqmx.DAQmx_Val_Volts, 
                    None
                )
                _task.WriteAnalogF64(
                    1, 
                    True, 
                    1, 
                    daqmx.DAQmx_Val_GroupByScanNumber, 
                    initial_ao_waveform, 
                    ct.byref(samples_per_ch_ct), 
                    None
                )
                _task.StartTask()
                _task.StopTask()
                _task.ClearTask()
                
            # Create and configure timing for AO tasks
            if self._task_ao is None:
                self._task_ao = daqmx.Task("TaskAO")
                self._task_ao.CreateAOVoltageChan(
                    self._address_ao_mirrors[0], 
                    "AO_ImageScanning", 
                    -6.0, 
                    6.0, 
                    daqmx.DAQmx_Val_Volts, 
                    None
                )
                self._task_ao.CreateAOVoltageChan(
                    self._address_ao_mirrors[1], 
                    "AO_ProjectionScanning", 
                    -1.0, 
                    1.0, 
                    daqmx.DAQmx_Val_Volts, 
                    None
                )

                if self.scan_type=='mirror':
                    # Configure timing to change on di change trigger, matches _do_waveform shape
                    self._task_ao.CfgSampClkTiming(
                        self._channel_di_change_trigger,
                        self._daq_sample_rate_hz, 
                        daqmx.DAQmx_Val_Rising, 
                        daqmx.DAQmx_Val_ContSamps,
                        self.samples_per_do_ch
                    )
                elif self.scan_type=="projection":
                    # Configure to run on internal clock, _ao_waveform has shape dictated by camera exposure
                    self._task_ao.CfgSampClkTiming(
                        "",
                        self._daq_sample_rate_hz,  # Define how fast the samples are output
                        daqmx.DAQmx_Val_Rising,
                        daqmx.DAQmx_Val_FiniteSamps,  # Output finite number of samples
                        self._ao_waveform.shape[0] # Total samples
                    )  

                    # Configure AO to start on the rising edge of DI signal
                    self._task_ao.CfgDigEdgeStartTrig(
                        self._channel_di_trigger_from_camera,
                        daqmx.DAQmx_Val_Rising
                    )
                                    
                    # Make the task retriggerable, for every exposure
                    self._task_ao.SetStartTrigRetriggerable(True)
                   
            # Write waveforms
            if self.scan_type=='mirror':
                # Write the output waveform
                self._task_ao.WriteAnalogF64(
                    self.samples_per_do_ch,
                    False, 
                    10.0, 
                    daqmx.DAQmx_Val_GroupByScanNumber, 
                    self._ao_waveform, 
                    ct.byref(samples_per_ch_ct), 
                    None
                )
            elif self.scan_type=="projection":                
                # Write the output waveform
                samples_per_ch_ct = ct.c_int32()
                self._task_ao.WriteAnalogF64(
                    self.samples_per_ao_ch,
                    False, 
                    10.0, 
                    daqmx.DAQmx_Val_GroupByScanNumber, 
                    self._ao_waveform, 
                    ct.byref(samples_per_ch_ct), 
                    None
                )                
        except (daqmx.DAQmxFunctions.InvalidTaskError, AttributeError):
            pass
     
    def start_waveform_playback(self):
        """Starts any tasks that exist."""
        
        # if already running, stop playback
        if self._running:
            self.stop_waveform_playback()
        try:
            for _task in [self._task_di, self._task_do, self._task_ao]:
                if _task:
                    _task.StartTask()
            # set running flag
            self._running = True
        except (daqmx.DAQmxFunctions.InvalidTaskError, AttributeError):
            pass

    def stop_waveform_playback(self):
        """Stop any tasks that exist."""
        if self._running:
            try:
                for _task in [self._task_di, self._task_do, self._task_ao]:
                    if _task:
                        _task.StopTask()
                self._running = False
            except (daqmx.DAQmxFunctions.InvalidTaskError, AttributeError):
                pass
      
    def clear_tasks(self):
        """Stop, Clear and remove task handlers."""
        
        # if already running, stop playback
        if self._running:
            self.stop_waveform_playback()
        try:
            for task_name in ["_task_di", "_task_do", "_task_ao"]:
                if hasattr(self, task_name):
                    task = getattr(self, task_name)
                    if task is not None:
                        # task.StopTask()
                        task.ClearTask()
                    setattr(self, task_name, None) 
            self._running = False
        except (daqmx.DAQmxFunctions.InvalidTaskError, AttributeError):
            pass
        
    def __del__(self):
        """Set DO to 0s, AO to neutral positions, clear tasks."""
        
        self.reset()