import argparse
import numpy as np
import scipy
import jsi
from pathlib import Path
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Generate heatmap using trained parameters')
parser.add_argument('-i', '--input', help='Input previously trained file', type=Path, required=True)
parser.add_argument('-o', '--output', help='Path for the output .png file', type=Path, required=True)

args = parser.parse_args()

with np.load(args.input) as input:
    data = dict(input)

jsi.data2HM(data)
plt.savefig(args.output.resolve())
