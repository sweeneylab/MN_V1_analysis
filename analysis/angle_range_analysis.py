import os
import pathlib
from collections import defaultdict

import cv2
import h5py
import yaml
import numpy as np
import pandas as pd
import roifile
import seaborn as sns
import tadpose
from matplotlib import pyplot as plt
from matplotlib import colors as mpl_colors
from skimage import draw
from tqdm.auto import tqdm, trange
from scipy.ndimage import gaussian_filter1d
from shared import settings


def angle_range(all_movs, stg, nodes, cfg):
    cfgs = cfg[stg]

    tab_all = []
    for fn in tqdm(all_movs, desc=f"Angle range {nodes[1]}"):
        gen = fn.parent.stem
        stg = fn.parent.parent.stem

        tail_a, tail_b, tail_c = nodes

        tad = tadpose.Tadpole.from_sleap(str(fn))
        track_okay_idx = np.nonzero(
            np.atleast_1d(
                tad.parts_detected(parts=(tail_b,), track_idx=None).sum(0) / tad.nframes
                > cfgs["TRACK_SELECT_THRES"]
            )
        )[0]

        file_path = tad.video_fn
        base_file = os.path.basename(file_path)[:-4]

        # aligner = tadpose.alignment.RotationalAligner(
        #     central_part=cfgs["ALIGN_CENTRAL"], aligned_part=cfgs["ALIGN_TOP"]
        # )

        # aligner.tracks_to_align = track_okay_idx

        # tad.aligner = aligner

        for tid in track_okay_idx:
            ang = tadpose.analysis.angles(
                tad, (tail_a, tail_b), (tail_b, tail_c), track_idx=tid
            )
            ang_smooth = gaussian_filter1d(ang, cfgs["FREQ_TEMP_ANGLE_SMOOTH"])

            speed_px_per_frame = tad.speed(
                cfgs["ANGLE_RANGE_MOVING_NODE"],
                track_idx=tid,
                pre_sigma=cfgs["LOCOMOTION_SPATIAL_SIGMA"],
                sigma=cfgs["LOCOMOTION_TEMPORAL_SIGMA"],
            )
            speed_calib = (
                speed_px_per_frame
                * cfg["FPS"]
                * tadpose.utils.calibrate_by_dish(
                    tad,
                    dish_diamter_in_cm=14,
                )
            )

            moving_bin = speed_calib > cfgs["ANGLE_RANGE_MOVING_NODE_THRESH"]

            if moving_bin.sum() > 0:
                ang_std_mov = ang_smooth[moving_bin].std()
                ang_p05_mov, ang_p95_mov = np.percentile(
                    ang_smooth[moving_bin], (5, 95)
                )
                ang_mean_mov = ang_smooth[moving_bin].mean()

                ang_speeds = np.gradient(ang_smooth)

                ang_mov_speed_pos_mean = ang_speeds[ang_speeds > 0].mean()
                ang_mov_speed_pos_std = ang_speeds[ang_speeds > 0].std()
                ang_mov_speed_pos_p95 = np.percentile(ang_speeds[ang_speeds > 0], 95)

                ang_mov_speed_neg_mean = -ang_speeds[ang_speeds < 0].mean()
                ang_mov_speed_neg_std = -ang_speeds[ang_speeds < 0].std()
                ang_mov_speed_neg_p95 = np.percentile(-ang_speeds[ang_speeds < 0], 95)

            else:
                ang_std_mov = np.nan
                ang_p05_mov = np.nan
                ang_mean_mov = np.nan
                ang_p05_mov = np.nan

                ang_mov_speed_pos_mean = np.nan
                ang_mov_speed_pos_std = np.nan
                ang_mov_speed_pos_p95 = np.nan
                ang_mov_speed_neg_mean = np.nan
                ang_mov_speed_neg_std = np.nan
                ang_mov_speed_neg_p95 = np.nan

            not_moving_bin = ~moving_bin
            if not_moving_bin.sum() > 0:
                ang_std_not_mov = ang_smooth[not_moving_bin].std()
                ang_p05_not_mov, ang_p95_not_mov = np.percentile(
                    ang_smooth[not_moving_bin], (5, 95)
                )
                ang_mean_not_mov = ang_smooth[not_moving_bin].mean()
            else:
                ang_std_not_mov = np.nan
                ang_p05_not_mov = np.nan
                ang_mean_not_mov = np.nan
                ang_p95_not_mov = np.nan

            tab_all.append(
                [
                    base_file,
                    stg,
                    gen,
                    tid,
                    #
                    ang_std_mov,
                    ang_p05_mov,
                    ang_mean_mov,
                    ang_p95_mov,
                    ang_std_not_mov,
                    ang_p05_not_mov,
                    ang_mean_not_mov,
                    ang_p95_not_mov,
                    ang_mov_speed_pos_mean,
                    ang_mov_speed_pos_std,
                    ang_mov_speed_pos_p95,
                    ang_mov_speed_neg_mean,
                    ang_mov_speed_neg_std,
                    ang_mov_speed_neg_p95,
                ]
            )

        tab_ar = pd.DataFrame(
            tab_all,
            columns=[
                "Movie",
                "Stage",
                "Genotype",
                "Track_idx",
                "angle_moving_std",
                "angle_moving_p05",
                "angle_moving_mean",
                "angle_moving_p95",
                "angle_non-moving_std",
                "angle_non-moving_p05",
                "angle_non-moving_mean",
                "angle_non-moving_p95",
                "angular_speed_mov_pos_mean",
                "angular_speed_mov_pos_std",
                "angular_speed_mov_pos_p95",
                "angular_speed_mov_neg_mean",
                "angular_speed_mov_neg_std",
                "angular_speed_mov_neg_p95",
            ],
        )
    return tab_ar


def run_stage(STAGE, cfg):
    cfgs = cfg[STAGE]
    ROOT_DIR = pathlib.Path(cfgs["ROOT_DIR"])

    all_movs = list(ROOT_DIR.rglob("*.mp4"))
    print(f" - Processing Stage {STAGE} with {len(all_movs)} movies")

    tab_ar_dict = {}
    if "ANGLE_RANGE_FOR" in cfgs:
        for name, nodes in cfgs["ANGLE_RANGE_FOR"].items():
            tab_ar = angle_range(all_movs, STAGE, nodes, cfg)

            tab_ar.to_csv(
                f"{cfg['ANGLE_RANGE_OUTDIR']}/angle_range_{STAGE}_{name}_res.tab",
                sep="\t",
            )
            tab_ar_dict[name] = tab_ar
    else:
        print("  -- no ANGLE_RANGE_FOR set")

    return tab_ar_dict


def run(STAGES, cfg):
    tab_all = []

    for STAGE in STAGES:
        tab_ar_dict = run_stage(STAGE, cfg)
        tab_all.append(tab_ar_dict)

    tab_collect = []
    for tab_dict in tab_all:
        for name, tab in tab_dict.items():
            tab["feature_for"] = name
            tab_collect.append(tab)

    if len(tab_collect) == 0:
        return

    tab_collect = pd.concat(tab_collect, axis=0, ignore_index=True)
    tab_collect.to_csv(f"{cfg['ANGLE_RANGE_OUTDIR']}/angle_range_res.tab", sep="\t")
    return tab_collect


def merge_results():
    cfg = settings()
    files = [
        "angle_range_52-54_tail_stem_res.tab",
        "angle_range_57-58_leg_ankle_res.tab",
        "angle_range_57-58_tail_stem_res.tab",
        "angle_range_59-62_leg_ankle_res.tab",
        "angle_range_59-62_tail_stem_res.tab",
        "angle_range_63-64_leg_ankle_res.tab",
        "angle_range_63-64_tail_stem_res.tab",
        "angle_range_44-48_tail_stem_res.tab",
    ]

    TAB = []
    for fn in files:
        tab = pd.read_csv(cfg["ANGLE_RANGE_OUTDIR"] + "/" + fn, sep="\t", index_col=0)
        name = "_".join(fn.split("_")[3:5])
        tab["feature_for"] = name
        TAB.append(tab)

    TAB = pd.concat(TAB, axis=0, ignore_index=True)
    TAB.to_csv(cfg["ANGLE_RANGE_OUTDIR"] + "/" + "angle_range_merged.tab", sep="\t")


def main(cfg=None):
    if cfg is None:
        cfg = settings()

    os.makedirs(cfg["ANGLE_RANGE_OUTDIR"], exist_ok=True)

    STAGES = cfg["STAGES"]
    run(STAGES, cfg)


if __name__ == "__main__":
    main()
