#TODO(fyordan): update xml parsing to use tag names instead of indices
#TODO(fyordan): figure out way to take all images at once
#TODO(fyordan): keep constants in single json file
#TODO(fyordan): add sanity check to ensure directories and base files exist
#TODO(fyordan): generalise to multiple cameras and initial configurations
#TODO(fyordan): add usage/documentation to script
#TODO(fyordan): shell=True is used but discouraged, make sure it is safe
#TODO(fyordan): use better conventions for filenames

from ast import literal_eval
from subprocess import call
from time import sleep
import json
import os
import sys
import xml.etree.ElementTree as ET


PATH_TO_TDP = os.path.expanduser("~/tdp/")
PATH_TO_CALIB_EXE = PATH_TO_TDP + "build/experiments/calib/calib"
PATH_TO_SIMPLE_GUI= PATH_TO_TDP + "build/experiments/simpleGui/simpleGui"
CONFIG_DIR = PATH_TO_TDP + "config/"
TMP_DIR = CONFIG_DIR + "tmp_data/"
INITIAL_DATA_DIR = CONFIG_DIR + "initial_camera_poses/"
BASE_JSON = INITIAL_DATA_DIR + "base.json"

#string corresponding to calibration command for two cameras
calib_command = lambda two_pairs, id1, id2: (PATH_TO_CALIB_EXE + \
                                             " -p -f -grid-spacing 0.0165 "
                                             "-grid-seed 76 -grid-rows 12 "
                                             "-grid-cols 16 -cameras " + \
                                             TMP_DIR + two_pairs + ".xml " + \
                                             "-o " + TMP_DIR + "cameras.xml" + \
                                             " files://" + TMP_DIR + \
                                             two_pairs + "/capture_stream[" + \
                                             id1 + "," + id2 + "]_*_8bit.png")


def calls_calibration_for(pair, id1, id2):
    """
    input parameters are all strings
    calls the calibration command for a particular camera pair
    outputs a tuple of poses (T_wc) corresponding to both cameras
    """
    shellCommand = calib_command(pair, id1, id2)
    call(shellCommand, shell=True)
    xmlRoot = ET.parse(TMP_DIR + "cameras.xml").getroot()
    pose1 = xmlRoot[0][1][0]
    pose2 = xmlRoot[1][1][0]
    return (pose1, pose2)


def run_calibration(leftId, topId, bottomId, rightId):
    """
    input parameters are all strings
    return poses for topId, bottomId, and rightId in that order
    (note the left pose is assumed to be identity at origin)
    """
    # first copy the initial xml files to tmp_data
    call("cp " + INITIAL_DATA_DIR + "left-top.xml " + TMP_DIR, shell=True)
    call("cp " + INITIAL_DATA_DIR + "left-bottom.xml " + TMP_DIR, shell=True)
    call("cp " + INITIAL_DATA_DIR + "bottom-right.xml " + TMP_DIR, shell=True)
    # find the top and bottom pose with respect to left camera
    topPose = calls_calibration_for("left-top", leftId, topId)[1]
    bottomPose = calls_calibration_for("left-bottom", leftId, bottomId)[1]
    # create new xml file bottom-right.xml from new calibration values
    bottom_right_xml = ET.parse(TMP_DIR + "bottom-right.xml")
    bottom_right_xml.getroot()[0][1][0].text = bottomPose.text
    bottom_right_xml.write(TMP_DIR + "bottom-right.xml")
    # find the right pose
    rightPose = calls_calibration_for("bottom-right", bottomId, rightId)[1]
    return (topPose, bottomPose, rightPose)


def transform_xml_pose_to_lists(pose):
    """
    takes a pose in xml format and outputs a tuple (rotation, translation)
    """
    poseList = literal_eval(pose.text.replace(";", ",").replace(" ", ""))
    R = poseList[0:3] + poseList[4:7] + poseList[8:11]
    T = [poseList[3], poseList[7], poseList[11]]
    return (R, T)


def create_json_file_from_poses(topPose, bottomPose, rightPose, baseJson):
    """
    creates a new json file from a base_json and poses(tuple of list format)
    note that the left pose is assumed to be non changing
    """
    with open(baseJson, "r") as baseFile:
        data = json.load(baseFile)
    data[2]["camera"]["T_rc"]["R_3x3"] = bottomPose[0]
    data[2]["camera"]["T_rc"]["t_xyz"] = bottomPose[1]
    data[3]["camera"]["T_rc"]["R_3x3"] = bottomPose[0]
    data[3]["camera"]["T_rc"]["t_xyz"] = bottomPose[1]
    data[4]["camera"]["T_rc"]["R_3x3"] = rightPose[0]
    data[4]["camera"]["T_rc"]["t_xyz"] = rightPose[1]
    data[5]["camera"]["T_rc"]["R_3x3"] = rightPose[0]
    data[5]["camera"]["T_rc"]["t_xyz"] = rightPose[1]
    data[6]["camera"]["T_rc"]["R_3x3"] = topPose[0]
    data[6]["camera"]["T_rc"]["t_xyz"] = topPose[1]
    data[7]["camera"]["T_rc"]["R_3x3"] = topPose[0]
    data[7]["camera"]["T_rc"]["t_xyz"] = topPose[1]

    with open(CONFIG_DIR + "output.json", "r+") as outputJson:
        json.dump(data, outputJson, indent=4, separators=(', ', ': '))
        # Add comma between } and ]
        outputJson.seek(-2, os.SEEK_END)
        outputJson.truncate()
        outputJson.write(",\n]")
    return None


def calibrate_all(leftId, topId, bottomId, rightId):
    """
    Assuming the images for calibration already exist,
    runs calibration for 3 camera pairs and generates output.json file
    """
    (top, bottom, right) = run_calibration(leftId, topId, bottomId, rightId)
    poses = [transform_xml_pose_to_lists(i) for i in (top, bottom, right)]
    create_json_file_from_poses(poses[0], poses[1], poses[2], BASE_JSON)
    return None


def take_images(leftId, topId, bottomId, rightId):
    """
    Calls SimpleGui thrice to take the images of left-top, left-bottom,
    and bottom-right in that order.
    GreyScales them and puts them in appropriate folder
    """
    print ("Take images of left-top")
    shellCommand = PATH_TO_SIMPLE_GUI + " realsense:[register=true]//"
    call(shellCommand, shell=True)
    call("rm -r " + TMP_DIR + "left-top", shell=True)
    call("mkdir " + TMP_DIR + "left-top", shell=True)
    call("mv capture_stream* " + TMP_DIR + "left-top/", shell=True)
    call("cd " + TMP_DIR + "left-top && " + \
         "python " + PATH_TO_TDP + "scripts/convertRgbToGrey.py", shell=True)

    print ("Take images of left-bottom")
    call(shellCommand, shell=True)
    call("rm -r " + TMP_DIR + "left-bottom", shell=True)
    call("mkdir " + TMP_DIR + "left-bottom", shell=True)
    call("mv capture_stream* " + TMP_DIR + "left-bottom/", shell=True)
    call("cd " + TMP_DIR + "left-bottom && " + \
         "python " + PATH_TO_TDP + "scripts/convertRgbToGrey.py", shell=True)

    print ("Take images of bottom-right")
    call(shellCommand, shell=True)
    call("rm -r " + TMP_DIR + "bottom-right", shell=True)
    call("mkdir " + TMP_DIR + "bottom-right", shell=True)
    call("mv capture_stream* " + TMP_DIR + "bottom-right", shell=True)
    call("cd " + TMP_DIR + "bottom-right && " + \
         "python " + PATH_TO_TDP + "scripts/convertRgbToGrey.py", shell=True)
    return None


if __name__ == "__main__":
    take_images(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    calibrate_all(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

