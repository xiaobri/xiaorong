"""EEG adaptation of I2MoE for five anatomical brain regions."""

from .model import EEGRegionI2MoE, I2MoEOutput
from .regions import EEG_CHANNEL_NAMES, REGION_CHANNELS, REGION_NAMES

__all__ = [
    "EEGRegionI2MoE",
    "I2MoEOutput",
    "EEG_CHANNEL_NAMES",
    "REGION_CHANNELS",
    "REGION_NAMES",
]
