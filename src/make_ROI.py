"""
Use for standalone freesurfer roi labels or ROI masks in volumetric space
Using freesurfer roi labels generally allows for easier drawing, since the
mask will be binaryself.
with volumetric masks, pycortex sampling results in a non-binary mask, which
must be outlined by hand.
"""

import cortex
import cortex.polyutils
import numpy as np
import os
import numpy as np
import mne
import nibabel as nib
import argparse

np.random.seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument("--roi_dir", type=str, default="./roi_data")
parser.add_argument("--subj", type=int, default=1)
parser.add_argument("--xfm", type=str, default="func1pt8_to_anat0pt8_autoFSbbr")
parser.add_argument("--compute-boundary", action="store_true")
parser.add_argument("--roi-format", default="vol", choices=["vol", "surf", "label"])
parser.add_argument(
    "--rois", nargs="*", type=str, default=["floc-faces", "floc-places", "floc-bodies", "prf-visualrois", "prf-eccrois", "floc-words"]
)
args = parser.parse_args()

surfs = [cortex.polyutils.Surface(*d) for d in cortex.db.get_surf("subj%02d" % args.subj, "fiducial")]
num_verts = np.array([surfs[0].pts.shape[0], surfs[1].pts.shape[0]])

for roi in args.rois:
    if args.roi_format == "vol":
        roi_dat = nib.load("%s/subj%02d/%s.nii.gz" % (args.roi_dir, args.subj, roi))
        overlay_dat = roi_dat.get_fdata().swapaxes(0, 2)
        V = cortex.Volume(
        overlay_dat,
        "subj%02d" % args.subj,
        args.xfm,
        mask=cortex.utils.get_cortical_mask(
            "subj%02d" % args.subj, args.xfm
        ),
        vmin=0,
        vmax=np.max(overlay_dat),
    )
    cortex.utils.add_roi(V, name=roi, open_inkscape=True, add_path=True, with_colorbar=True)
