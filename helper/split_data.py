import os
import splitfolders

# Input base directory and output directory
base_dir = ""
output_dir = ""

# Splitting the dataset folder
splitfolders.ratio(base_dir, output_dir, seed=1217, ratio=(0.7, 0.15, 0.15), group_prefix=None)
