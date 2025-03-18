
from useq import MDAEvent, CustomAction
from types import MappingProxyType as mappingproxy

O2O3_af_event = MDAEvent(
    action=CustomAction(
        name="O2O3-autofocus",
        data = {
            "Camera" : {                    
                "exposure_ms" : None,
                "camera_crop" : [
                    None,
                    None,
                    None,
                    None
                ]
            }
        }
    )
)

AO_optimize_event = MDAEvent(
    exposure = None,
    action=CustomAction(
        name="AO-optimize",
        data = {
            "AO" : {
                "opm_mode": str("projection"),
                "channel_states": None,
                "channel_powers" : None,
                "mode": None,
                "iterations": None,
                "image_mirror_range_um" : None,
                "blanking": bool(True),
                "apply_existing": bool(False),
                "pos_idx":int(0),
                "output_path":None
            },
            "Camera" : {
                "exposure_ms": None,
                "camera_crop" : [
                    None,
                    None,
                    None,
                    None
                ]
            }
        }
    )
)

# Create Fluidics program event
FP_event = MDAEvent(
    # exposure = AO_exposure_ms,
    action=CustomAction(
        name="Fluidics",
        data = {
            "Fluidics" : {
                "mode" : None,
                "round" : None
            }
            }
    )
)

# create DAQ hardware setup event
DAQ_event = MDAEvent(
    action=CustomAction(
        name="DAQ",
        data = {
            "DAQ" : {
                "mode" : None,
                "image_mirror_step_um" : None,
                "image_mirror_range_um" : None,
                "stage_scan_range_um": None,
                "channel_states" : None,
                "channel_powers" : None,
                "interleaved" : None,
                "blanking" : None, 
            },
            "Camera" : {
                "exposure_channels" : None,
                "camera_crop" : [
                    None,
                    None,
                    None,
                    None,
                ]
            }
        }
    )
)
