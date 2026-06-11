import os
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import skew
import tadpose
from matplotlib import pyplot as plt
from matplotlib import colors as mpl_colors
from tqdm.auto import tqdm
from scipy.ndimage import gaussian_filter1d

from shared import settings


def angle_correlation(all_movs, stg, name, nodes, cfg):
    cfgs = cfg[stg]

    tab_all = []
    for fn in tqdm(all_movs, desc=f"Angle correlation {name}"):
        gen = fn.parent.stem
        stg = fn.parent.parent.stem

        a_nodes, b_nodes = nodes

        tad = tadpose.Tadpole.from_sleap(str(fn))

        track_okay_idx = np.nonzero(
            np.atleast_1d(
                tad.parts_detected(
                    parts=(cfgs["LOCOMOTION_NODE"],), track_idx=None
                ).sum(0)
                / tad.nframes
                > cfgs["TRACK_SELECT_THRES"]
            )
        )[0]

        file_path = tad.video_fn
        base_file = os.path.basename(file_path)[:-4]

        for tid in track_okay_idx:
            a_ang = tadpose.analysis.angles(
                tad,
                (a_nodes[0], a_nodes[1]),
                (a_nodes[1], a_nodes[2]),
                track_idx=tid,
            )
            b_ang = tadpose.analysis.angles(
                tad,
                (b_nodes[0], b_nodes[1]),
                (b_nodes[1], b_nodes[2]),
                track_idx=tid,
            )

            a_ang_smooth = gaussian_filter1d(
                a_ang, cfgs["ANGLE_CORR_TEMP_ANGLE_SMOOTH"]
            )
            a_ang_zscore = (a_ang_smooth - a_ang_smooth.mean()) / a_ang_smooth.std()

            b_ang_smooth = gaussian_filter1d(
                b_ang, cfgs["ANGLE_CORR_TEMP_ANGLE_SMOOTH"]
            )
            b_ang_zscore = (b_ang_smooth - b_ang_smooth.mean()) / b_ang_smooth.std()

            if name.startswith("LR"):
                b_ang_zscore *= -1

            corr = tadpose.utils.corr_roll(
                a_ang_zscore, b_ang_zscore, win=cfgs["ANGLE_CORR_WIN_SIZE"]
            )
            a_ang_grad_mag = gaussian_filter1d(
                np.abs(np.gradient(a_ang_zscore)), cfgs["ANGLE_CORR_ACTIVE_SMOOTH"]
            )
            b_ang_grad_mag = gaussian_filter1d(
                np.abs(np.gradient(b_ang_zscore)), cfgs["ANGLE_CORR_ACTIVE_SMOOTH"]
            )

            ang_grad_mag = np.maximum(a_ang_grad_mag, b_ang_grad_mag)

            active_bin = ang_grad_mag > cfgs["ANGLE_CORR_ACTIVE_THRESH"]
            corr_active = corr[active_bin]

            corr_active = corr_active[~np.isnan(corr_active)]

            if len(corr_active) > 0:
                tab_all.append(
                    [
                        base_file,
                        stg,
                        gen,
                        tid,
                        name,
                        np.median(corr_active),
                        np.percentile(corr_active, 5),
                        np.percentile(corr_active, 95),
                        skew(corr_active),
                        corr_active.std(),
                        len(corr_active) / tad.nframes,
                    ]
                )

                #### Plots
                f, ax = plt.subplots(figsize=(6, 6))
                ax.hist(corr_active, bins=100, range=[-1, 1])
                ax.set_title(f"{name} {stg} {gen}\n{base_file}")
                ax.set_xlabel("Correlation {name}")
                ax.set_ylabel("Count")
                plt.tight_layout()
                plt.savefig(
                    f"{cfg['ANGLE_CORR_OUTDIR']}/imgs/{stg}_{gen}_{name}_{base_file}.png"
                )
                plt.close(f)

            else:
                tab_all.append(
                    [
                        base_file,
                        stg,
                        gen,
                        tid,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        0,
                    ]
                )

    tab_ar = pd.DataFrame(
        tab_all,
        columns=[
            "Movie",
            "Stage",
            "Genotype",
            "Track_idx",
            "corr_median",
            "corr_p05",
            "corr_p95",
            "corr_skewness",
            "corr_std",
            "computed_of_ratio",
        ],
    )
    return tab_ar


def run_stage(STAGE, cfg):
    cfgs = cfg[STAGE]
    ROOT_DIR = pathlib.Path(cfgs["ROOT_DIR"])

    all_movs = list(ROOT_DIR.rglob("*.mp4"))
    print(f" - Processing Stage {STAGE} with {len(all_movs)} movies")

    tab_ar_dict = {}
    if "ANGLE_CORR_FOR" in cfgs:
        for name, nodes_tuple in cfgs["ANGLE_CORR_FOR"].items():
            tab_ar = angle_correlation(all_movs, STAGE, name, nodes_tuple, cfg)

            tab_ar.to_csv(
                f"{cfg['ANGLE_CORR_OUTDIR']}/angle_correlation_{STAGE}_{name}_res.tab",
                sep="\t",
            )
            tab_ar_dict[name] = tab_ar
    else:
        print("  -- no ANGLE_CORR_FOR set")

    return tab_ar_dict


def run(STAGES, cfg):
    tab_all = []

    for STAGE in STAGES:
        tab_ac_dict = run_stage(STAGE, cfg)
        tab_all.append(tab_ac_dict)

    tab_collect = []
    for tab_dict in tab_all:
        for name, tab in tab_dict.items():
            tab["feature_for"] = name
            tab_collect.append(tab)

    if len(tab_collect) == 0:
        return

    tab_collect = pd.concat(tab_collect, axis=0, ignore_index=True)
    tab_collect.to_csv(
        f"{cfg['ANGLE_CORR_OUTDIR']}/angle_correlation_res.tab", sep="\t"
    )
    return tab_collect


def main(cfg=None):
    if cfg is None:
        cfg = settings()

    os.makedirs(cfg["ANGLE_CORR_OUTDIR"], exist_ok=True)
    os.makedirs(cfg["ANGLE_CORR_OUTDIR"] + "/imgs", exist_ok=True)

    STAGES = cfg["STAGES"]
    run(STAGES, cfg)


if __name__ == "__main__":
    main()
