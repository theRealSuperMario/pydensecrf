import ipywidgets as widgets
import cv2
import glob
import os
import numpy as np
import pydensecrf.densecrf as dcrf
from supermariopy import plotting
from pydensecrf.utils import (
    unary_from_labels,
    create_pairwise_bilateral,
    create_pairwise_gaussian,
)
from matplotlib import pyplot as plt


image_files = sorted(
    glob.glob("MSRC_ObjCategImageDatabase_v2/SegmentationsGTHighQuality/*_s.bmp")
)


def append_fname(x, suffix):
    pardir = os.path.dirname(x)
    basename = os.path.basename(x)
    fname, ext = os.path.splitext(basename)
    return os.path.join(pardir, fname + suffix + ext)


def run_crf(theta_alpha, theta_beta, theta_gamma, compat, pairwise, image_select):
    """ run crf on image from MSRC-21 dataset """
    fn_im = image_select
    fn_gt = append_fname(fn_im, "_HQGT")
    fn_prior_anno = append_fname(fn_im, "_GT")
    fn_prior_anno = fn_prior_anno.replace("SegmentationsGTHighQuality", "GroundTruth")

    img = cv2.cvtColor(cv2.imread(fn_im), cv2.COLOR_BGR2RGB)

    # Convert the annotation's RGB color to a single 32-bit integer color 0xBBGGRR
    prior_anno_rgb = cv2.cvtColor(cv2.imread(fn_prior_anno), cv2.COLOR_BGR2RGB).astype(
        np.uint32
    )
    prior_anno_lbl = (
        prior_anno_rgb[:, :, 0]
        + (prior_anno_rgb[:, :, 1] << 8)
        + (prior_anno_rgb[:, :, 2] << 16)
    )

    gt_rgb = cv2.cvtColor(cv2.imread(fn_gt), cv2.COLOR_BGR2RGB).astype(np.uint32)
    gt_lbl = gt_rgb[:, :, 0] + (gt_rgb[:, :, 1] << 8) + (gt_rgb[:, :, 2] << 16)

    # Convert the 32bit integer color to 1, 2, ... labels.
    # Note that all-black, i.e. the value 0 for background will stay 0.
    colors, labels = np.unique(prior_anno_lbl, return_inverse=True)
    _, gt_labels = np.unique(gt_lbl, return_inverse=True)
    gt_labels = gt_labels.reshape((img.shape[0], img.shape[1]))

    # But remove the all-0 black, that won't exist in the MAP!
    HAS_UNK = 0 in colors
    if HAS_UNK:
        print(
            "Found a full-black pixel in annotation image, assuming it means 'unknown' label, and will thus not be present in the output!"
        )
        print(
            "If 0 is an actual label for you, consider writing your own code, or simply giving your labels only non-zero values."
        )
        colors = colors[1:]
    # else:
    #    print("No single full-black pixel found in annotation image. Assuming there's no 'unknown' label!")

    # And create a mapping back from the labels to 32bit integer colors.
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:, 0] = colors & 0x0000FF
    colorize[:, 1] = (colors & 0x00FF00) >> 8
    colorize[:, 2] = (colors & 0xFF0000) >> 16

    # Compute the number of classes in the label image.
    # We subtract one because the number shouldn't include the value 0 which stands
    # for "unknown" or "unsure".
    n_labels = len(set(labels.flat)) - int(HAS_UNK)
    print(
        n_labels,
        " labels",
        (' plus "unknown" 0: ' if HAS_UNK else ""),
        set(labels.flat),
    )

    ###########################
    ### Setup the CRF model ###
    ###########################

    # get unary potentials (neg log probability)
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)

    img = cv2.cvtColor(cv2.imread(fn_im), cv2.COLOR_BGR2RGB)
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)
    d.setUnaryEnergy(U)
    d.addPairwiseBilateral(
        sxy=(theta_alpha, theta_alpha),
        srgb=(theta_beta, theta_beta, theta_beta),
        rgbim=img,
        compat=10,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )
    if pairwise:
        d.addPairwiseGaussian(sxy=theta_gamma, compat=compat)
    Q = d.inference(10)
    map_ = np.argmax(Q, axis=0).reshape(img.shape[:2])[::-1, ...]

    map_rgb = colorize[map_, :]
    return img, gt_rgb, prior_anno_rgb, map_rgb
