import io
import os
import pydicom
import numpy as np
import pandas as pd
from scipy import ndimage as nd
from pylinac import FieldAnalysis, Centering
from pylinac.core.image import DicomImage
from skimage.feature import canny
from skimage.transform import hough_circle_peaks, hough_circle
from zipfile import ZipFile

# helper functions
def _field_details_from_dicom(fname=""):
    """
    Read RT Image and return geometry info

    Args:
        fname (str): file path

    Returns:
        fname (str): file path
        g (float): gantry angle
        c (float): collimator angle
        ds (object): pydicom object
    """

    f = zf.open(fname)
    f = io.BytesIO(f.read())
    # read dicom file
    ds = pydicom.dcmread(f)
    # record gantry and collimator angles
    g = float(round(ds.GantryAngle, 1))
    if round(g) == 360.0:
        g = 0.0
    c = float(round(ds.BeamLimitingDeviceAngle, 1))
    if c == 360:
        c = 0.0
    _, fname = os.path.split(fname)
    return fname, g, c, ds


def _file_tally(results={}, test_name=""):
    '''
    Basic verification of RTImage files and expected beam geometries for each session in the test cycle.
    Months: 1a, 1b, 2a, 2b and 3

    Args:
        results (dict): radiation field size analysis results
        test_name (str): month name, e.g. 1a

    Returns:
        msg (str): verification status message
        fault_flag (bool): True if verfication passes, False if verification fails

    '''
    df = pd.DataFrame.from_dict(results)
    msg = "Files processed successfully"
    fault_flag = False
    machines = len(df["MachineName"].unique())
    if machines != 1:
        msg = "Files from more than one linac detected, re-upload data from a single machine."
        return msg, fault_flag
    if test_name.lower() == "1a":
        lst = [
            len(df[(df["Energy"] == "6X") & (df["G"] == 0) & (df["C"] == 0)]),
            len(df[(df["Energy"] == "6X") & (df["G"] == 0) & (df["C"] == 90)]),
            len(df[(df["Energy"] == "6X") & (df["G"] == 0) & (df["C"] == 270)]),
            len(df[(df["Energy"] == "6X") & (df["G"] == 90) & (df["C"] == 0)]),
            len(df[(df["Energy"] == "6X") & (df["G"] == 90) & (df["C"] == 90)]),
            len(df[(df["Energy"] == "6X") & (df["G"] == 90) & (df["C"] == 270)]),
            len(df[(df["Energy"] == "10X") & (df["G"] == 0) & (df["C"] == 0)]),
            len(df[(df["Energy"] == "10X") & (df["G"] == 0) & (df["C"] == 90)]),
            len(df[(df["Energy"] == "10X") & (df["G"] == 0) & (df["C"] == 270)]),
            len(df[(df["Energy"] == "10X") & (df["G"] == 270) & (df["C"] == 0)]),
            len(df[(df["Energy"] == "10X") & (df["G"] == 270) & (df["C"] == 90)]),
            len(df[(df["Energy"] == "10X") & (df["G"] == 270) & (df["C"] == 270)]),
        ]
    elif test_name.lower() == "1b":
        lst = [
            len(df[(df["Energy"] == "6X") & (df["G"] == 0) & (df["C"] == 0)]),
            len(df[(df["Energy"] == "6X") & (df["G"] == 0) & (df["C"] == 90)]),
            len(df[(df["Energy"] == "6X") & (df["G"] == 0) & (df["C"] == 270)]),
            len(df[(df["Energy"] == "10X") & (df["G"] == 90) & (df["C"] == 0)]),
            len(df[(df["Energy"] == "10X") & (df["G"] == 90) & (df["C"] == 90)]),
            len(df[(df["Energy"] == "10X") & (df["G"] == 90) & (df["C"] == 270)]),
            len(df[(df["Energy"] == "10X") & (df["G"] == 0) & (df["C"] == 0)]),
            len(df[(df["Energy"] == "10X") & (df["G"] == 0) & (df["C"] == 90)]),
            len(df[(df["Energy"] == "10X") & (df["G"] == 0) & (df["C"] == 270)]),
            len(df[(df["Energy"] == "6X") & (df["G"] == 270) & (df["C"] == 0)]),
            len(df[(df["Energy"] == "6X") & (df["G"] == 270) & (df["C"] == 90)]),
            len(df[(df["Energy"] == "6X") & (df["G"] == 270) & (df["C"] == 270)]),
        ]
    elif test_name.lower() == "2a":
        lst = [
            len(df[(df["Energy"] == "6X") & (df["G"] == 0) & (df["C"] == 0)]) - 1,
            len(df[(df["Energy"] == "6X") & (df["G"] == 0) & (df["C"] == 90)]),
            len(df[(df["Energy"] == "6X") & (df["G"] == 0) & (df["C"] == 270)]),
            len(df[(df["Energy"] == "6X") & (df["G"] == 90) & (df["C"] == 0)]),
            len(df[(df["Energy"] == "6X") & (df["G"] == 90) & (df["C"] == 90)]),
            len(df[(df["Energy"] == "6X") & (df["G"] == 90) & (df["C"] == 270)]),
            len(df[(df["Energy"] == "6X") & (df["G"] == 270) & (df["C"] == 0)]),
            len(df[(df["Energy"] == "6X") & (df["G"] == 270) & (df["C"] == 90)]),
            len(df[(df["Energy"] == "6X") & (df["G"] == 270) & (df["C"] == 270)]),
        ]
    elif test_name.lower() == "2b":
        lst = [
            len(df[(df["Energy"] == "10X") & (df["G"] == 0) & (df["C"] == 0)]) - 1,
            len(df[(df["Energy"] == "10X") & (df["G"] == 0) & (df["C"] == 90)]),
            len(df[(df["Energy"] == "10X") & (df["G"] == 0) & (df["C"] == 270)]),
            len(df[(df["Energy"] == "10X") & (df["G"] == 90) & (df["C"] == 0)]),
            len(df[(df["Energy"] == "10X") & (df["G"] == 90) & (df["C"] == 90)]),
            len(df[(df["Energy"] == "10X") & (df["G"] == 90) & (df["C"] == 270)]),
            len(df[(df["Energy"] == "10X") & (df["G"] == 270) & (df["C"] == 0)]),
            len(df[(df["Energy"] == "10X") & (df["G"] == 270) & (df["C"] == 90)]),
            len(df[(df["Energy"] == "10X") & (df["G"] == 270) & (df["C"] == 270)]),
        ]
    elif test_name.lower() == "3":
        lst = [
            len(df[(df["Energy"] == "6X") & (df["G"] == 0) & (df["C"] == 0)]) - 1,
            len(df[(df["Energy"] == "6X") & (df["G"] == 0) & (df["C"] == 90)]),
            len(df[(df["Energy"] == "6X") & (df["G"] == 0) & (df["C"] == 270)]),
            len(df[(df["Energy"] == "10X") & (df["G"] == 0) & (df["C"] == 0)]) - 1,
            len(df[(df["Energy"] == "10X") & (df["G"] == 0) & (df["C"] == 90)]),
            len(df[(df["Energy"] == "10X") & (df["G"] == 0) & (df["C"] == 270)]),
        ]
    lst_tally = (set(lst), lst[0])

    if len(lst_tally[0]) != 1 or lst_tally[1] != 1:
        msg = (
            "Incorrect files uploaded for this month's test, check before re-uploading."
        )
        fault_flag = True
        return msg, fault_flag

    return msg, fault_flag


def _bb_detect(
    ds=None,
    BB_size=3.0,
    size_range=2.0,
    roi=[(-40.0, 40.0, 10.0), (0.0, 0.0, 10.0)],
):
    """
    Read RT Image pydicom object and detect BBs in predetermined ROIs

    Args:
        ds (object): pydicom image object
        BB_size (float): expected BB diameter (mm)
        size_range (float): possible BB size variation (+/-mm)
        roi (list): list containing a tuple for each roi centre in mm (y,x,size)

    Returns:
        roi_flags (list): list of bools (True if BB detected in ROI)
        BB_info (list): list of tuples denoting BB coordinates in pixels (y,x,radius)
    """

    roi_flags = [False] * len(roi)
    BB_info = [([], [], [])] * len(roi)

    # preprocess image
    img = ds.pixel_array
    img = img.astype("float")
    img = nd.median_filter(img, 3)

    # image array gemoetry
    sid = float(ds.RTImageSID)
    px = 1000 / sid * float(ds.ImagePlanePixelSpacing[0])
    imdim = np.array(img.shape)
    cnt = imdim / 2

    # BB radius
    radius = BB_size / 2
    radius_px = np.round(radius / px)
    rng_px = np.ceil(size_range / 2 / px)
    if radius_px - rng_px < 2:
        r_lo = 2.0
    else:
        r_lo = radius_px - rng_px
    radii = np.arange(r_lo, radius_px + rng_px)

    # BB search
    for i, r in enumerate(roi):
        # convert roi to px
        y = int(cnt[0] + np.round(r[0] / px))
        x = int(cnt[1] + np.round(r[1] / px))
        half_roi = int(np.round(r[2] / px / 2))
        # crop image
        img_roi = img[y - half_roi : y + half_roi, x - half_roi : x + half_roi]
        # canny edge filter and hough circle detection
        magnitude = nd.gaussian_gradient_magnitude(img_roi, 0.25)
        magnitude = canny(magnitude, sigma=5)
        hspaces = hough_circle(magnitude, radii)
        accum, cx, cy, rad = hough_circle_peaks(hspaces, radii, total_num_peaks=1)
        # flag if BB detected
        if accum.size > 0 and accum > 0.6:
            roi_flags[i] = True
            BB_info[i] = (y - half_roi + cy[0], x - half_roi + cx[0], rad[0])

    return roi_flags, BB_info


def _find_field_centre(ds=None, bb_flag="None", bb_loc=(0, 0)):
    """
    Calculate field centre from BB location

    Args:
      ds (object): pydicom image object
      bb_flag (str): expected BB location. "centre" at image centre; "suncheck" 4cm offset (y & x); "None" ignore BB and default to image centre
      bb_loc (tuple): BB location in pixels (y,x)

    Returns:
      prcnt_cnt (tuple): beam central axis as a percent of image size (y,x)
      cnt_shift (tuple): beam central axis shift in pixels (y,x)
    """
    # image array gemoetry
    sid = float(ds.RTImageSID)
    px = 1000 / sid * float(ds.ImagePlanePixelSpacing[0])
    imdim = np.array(ds.pixel_array.shape)
    cnt = imdim / 2
    bb_loc = np.array(bb_loc)
    # BB location
    if bb_flag.lower() == "suncheck":
        expected_cnt = np.array(
            [cnt[0] - np.round(40.0 / px), cnt[1] + np.round(40.0 / px)]
        )
        prct_cnt = bb_loc / expected_cnt * 0.5
        cnt_shift = (bb_loc - expected_cnt) * px
    elif bb_flag.lower() == "centre":
        prct_cnt = bb_loc / cnt * 0.5
        cnt_shift = (bb_loc - cnt) * px
    else:
        prct_cnt = None
        cnt_shift = None
    return prct_cnt, cnt_shift


def _analyse_img(
    fname=None, collimator_angle=None, centering="None", center_pos=(0.5, 0.5)
):
    """
    Measure field edges relative to central axis

    Args:
        fname (str): file path
        collimator_angle (int): must be a cardinal angle (0, 90, 180, 270, 360)
        centering (str): "manual" requires specification of beam centre as a fraction of image array; otherwise defaults to automatic centering
        center_pos (tuple): beam centre location as a fraction (y,x)

    Returns:
        Y1 (float): Y1 distance from CAX (mm)
        Y2 (float): Y2 distance from CAX (mm)
        X1 (float): X1 distance from CAX (mm)
        X2 (float): X2 distance from CAX (mm)
        Y (float): Y field width (mm)
        X (float): X field width (mm)
    """
    # define field orientations relative to collimator angle
    field_orientations = {
        "c_angle": [0, 90, 180, 270, 360],
        "cax_names": [
            "cax_to_top_mm",
            "cax_to_bottom_mm",
            "cax_to_left_mm",
            "cax_to_right_mm",
        ],
        "coll_names": [
            ["Y2", "Y1", "X1", "X2"],
            ["X1", "X2", "Y1", "Y2"],
            ["Y1", "Y2", "X2", "X1"],
            ["X2", "X1", "Y2", "Y1"],
            ["Y2", "Y1", "X1", "X2"],
        ],
    }
    fname = zf.open(fname)
    fname = io.BytesIO(fname.read())
    # pylinac field analysis
    img = FieldAnalysis(fname)
    if centering.lower() == "manual":
        img.analyze(
            centering=Centering.MANUAL,
            vert_position=center_pos[0],
            horiz_position=center_pos[1],
        )
    else:
        img.analyze()
    results = img.results_data(as_dict=True)
    # estimate actual field sizes
    c = int(collimator_angle)
    if c == 0 or c == 360:
        X = round(results["field_size_horizontal_mm"], 2)
        Y = round(results["field_size_vertical_mm"], 2)
    elif c == 90 or c == 270:
        Y = round(results["field_size_horizontal_mm"], 2)
        X = round(results["field_size_vertical_mm"], 2)
    else:
        Y = False
        X = False
    # extract measured field sizes
    i = field_orientations["c_angle"].index(int(collimator_angle))
    coll_order = field_orientations["coll_names"][i]
    cax_dict = {}
    for j, k in enumerate(coll_order):
        cax_dict[k] = round(results[field_orientations["cax_names"][j]], 2)
    return cax_dict["Y1"], cax_dict["Y2"], cax_dict["X1"], cax_dict["X2"], Y, X


# Identify test name
test_list_name = META["test_list_name"]
for idx, s in enumerate(test_list_name):
    if s == " ":
        test_month = test_list_name[idx + 1 :]


# extract dicom file data from zip
with ZipFile(BIN_FILE, "r") as zf:
    file = zf.namelist()
    list_dcm_files = [s for s in file if ".dcm" in s]
    results = {
        "G": [],
        "C": [],
        "X": [],
        "Y": [],
        "X1": [],
        "X2": [],
        "Y1": [],
        "Y2": [],
        "XOffset": [],
        "YOffset": [],
        "CAXOffset": [],
        "BBLoc": [],
        "Energy": [],
        "MachineName": [],
        "FileName": [],
        "JawPositionX1": [],
        "JawPositionX2": [],
        "JawPositionY1": [],
        "JawPositionY2": [],
    }
    for l in list_dcm_files:
        # field geometry
        dcm_name, gantry_angle, coll_angle, ds = _field_details_from_dicom(l)
        _, dcm_fname = os.path.split(dcm_name)
        img_desc = ds.RTImageDescription[0 : ds.RTImageDescription.index(" [")].upper()
        jawsx = np.round(
            ds.ExposureSequence[0].BeamLimitingDeviceSequence[0].LeafJawPositions, 2
        )
        jawsy = np.round(
            ds.ExposureSequence[0].BeamLimitingDeviceSequence[1].LeafJawPositions, 2
        )

        # record initial results
        results["Energy"].append(img_desc)
        results["MachineName"].append(ds.StationName)
        results["FileName"].append(dcm_fname)
        results["G"].append(gantry_angle)
        results["C"].append(coll_angle)

        # detect BB in image
        loc_flag, bb_loc = _bb_detect(ds)
        if loc_flag[0]:
            bb_type = "suncheck"
            bb_loc = bb_loc[0][:2]
        elif loc_flag[1]:
            bb_type = "centre"
            bb_loc = bb_loc[1][:2]
        else:
            bb_type = "none"
            bb_loc = None
            cnt_shift = (0, 0)
        # find field centre
        prcnt_cnt, cnt_shift = _find_field_centre(ds, bb_type, bb_loc)
        # field edge analysis
        if bb_loc:
            y1, y2, x1, x2, y_size, x_size = _analyse_img(
                l, coll_angle, "manual", prcnt_cnt
            )
        else:
            y1, y2, x1, x2, y_size, x_size = _analyse_img(l, coll_angle)
        # field results
        results["BBLoc"].append(bb_type)
        results["X"].append(x_size)
        results["Y"].append(y_size)
        results["X1"].append(x1)
        results["X2"].append(x2)
        results["Y1"].append(y1)
        results["Y2"].append(y2)
        results["CAXOffset"].append(cnt_shift)
        results["XOffset"].append(round(-x1 + x_size / 2, 2))  # +ve = offset towards X2
        results["YOffset"].append(round(-y1 + y_size / 2, 2))  # +ve = offset towards Y2
        results["JawPositionX1"].append(jawsx[0])
        results["JawPositionX2"].append(jawsx[1])
        results["JawPositionY1"].append(jawsy[0])
        results["JawPositionY2"].append(jawsy[1])


msg, rslt_flag = _file_tally(results, test_month)
UTILS.set_comment(msg)
zz_lrc_fsupload = results
