import numpy as np
import argparse
import matplotlib.pyplot as plt
from helpers import *

pathToTimings = "../results/ablation_fr2_xyz/fullmode/timings.txt"
parser = argparse.ArgumentParser(description = 'extract framerate')
parser.add_argument('-i','--input',
    default="./timings.txt", help='path to input timings file')
cmdArgs = parser.parse_args()

timings = ParseTimings(cmdArgs.input)

tFull = np.array(timings["FullLoop"])
tDraw = np.array(timings["Draw2D"])-np.array(timings["Draw3D"])

print np.mean(tFull-tDraw)
