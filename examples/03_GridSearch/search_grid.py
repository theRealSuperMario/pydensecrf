import sys
from sklearn.model_selection import ParameterGrid
import numpy as np
import pandas as pd

sys.path.insert(0, "../interactive")
import utils

from matplotlib import pyplot as plt
from supermariopy import plotting
import os
from supermariopy import imageutils

import multiprocessing
from contextlib import closing


# change these values here
theta_alpha = [31]
theta_beta = [31]
theta_gamma = np.arange(1, 20, 2)
compat = np.arange(1, 20, 2)
pairwise = [True]


# x is the value that is incremented first
# y is the value that is incremented last
x_ticks = theta_gamma
x_label = r"$\theta_\gamma$"
y_ticks = compat
y_label = r"compat"


def unsynchronized(fn_image, param_grid):
    canvas = []
    for param in ParameterGrid(param_grid):
        img, gt_rgb, prior_anno_rgb, map_rgb = utils.run_crf(
            **param, image_select=fn_image
        )
        canvas.append(map_rgb)
    canvas = imageutils.batch_to_canvas(np.stack(canvas, axis=0))
    fig = plt.figure(figsize=(20, 15))
    ax_img = plt.subplot2grid((3, 3), (0, 0))
    ax_gt = plt.subplot2grid((3, 3), (0, 1))
    ax_prior = plt.subplot2grid((3, 3), (0, 2))
    ax_grid = plt.subplot2grid((3, 3), (1, 0), colspan=3, rowspan=2)

    plotting.plot_canvas(canvas, img.shape[1], img.shape[0], fig=fig, ax=ax_grid)
    ax_grid.set_xlabel(x_label)
    ax_grid.set_xticklabels(x_ticks)

    ax_grid.set_ylabel(y_label)
    ax_grid.set_yticklabels(y_ticks)
    ax_grid.set_title("MAP")

    ax_img.imshow(img)
    ax_img.set_title("Image")

    ax_gt.imshow(gt_rgb)
    ax_gt.set_title("GT")

    ax_prior.imshow(prior_anno_rgb)
    ax_prior.set_title("Unary")

    plotting.set_all_axis_off([ax_img, ax_gt, ax_prior])
    plotting.set_all_fontsize([ax_img, ax_gt, ax_prior, ax_grid], 22)

    plt.savefig("grid_" + os.path.basename(fn_image).replace("bmp", "png"))


def main():
    # when done figuring the grid together, convert notebook to script and run

    # # Dont change the following lines

    # In[3]:

    param_grid = {
        "theta_alpha": theta_alpha,
        "theta_beta": theta_beta,
        "theta_gamma": theta_gamma,
        "compat": compat,
        "pairwise": pairwise,
    }

    # In[4]:

    df = pd.DataFrame(list(ParameterGrid(param_grid)))
    df.to_csv("grid.csv", index=False)
    print(df)
    param_list = list(ParameterGrid(param_grid))

    import functools

    func = functools.partial(unsynchronized, param_grid=param_grid)
    with closing(multiprocessing.Pool(4)) as p:
        for o in p.imap_unordered(func, utils.image_files):
            pass


if __name__ == "__main__":
    main()
