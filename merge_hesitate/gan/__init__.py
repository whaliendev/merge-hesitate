"""
Merge-Hesitate-GAN: Enhanced merge conflict resolution using GAN-based confidence estimation.

This subpackage extends the base Merge-Hesitate model with adversarial training
to improve confidence estimation and solution quality.
"""

from .model import MergeHesitateGAN
from .discriminator import MergeDiscriminator 