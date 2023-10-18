cd code
declare -a rois=('EarlyVis' 'LOC' 'PPA' 'RSC' 'OPA')

for subnum in {2..3}; do
  sub=CSI${subnum}

  #first create the mask in anatomical space. binarize all voxels in ribbon: lh and rh
  mri_binarize --i $SUBJECTS_DIR/sub-$sub/mri/ribbon.mgz \
  --o $SUBJECTS_DIR/sub-$sub/mri/cortex.mgz --match 3 --match 42

  for folder in ${BOLD5000}/derivatives/fmriprep/sub-${sub}/ses*; do
    ses=${folder: -2}
    if [ ! -f ${SUBJECTS_DIR}/sub-${sub}/mri/cortex_func_ses-${ses}.nii ]; then
      #convert to functional space
      mri_vol2vol --mov $SUBJECTS_DIR/sub-$sub/mri/cortex.mgz \
      --regheader \
      --targ ${BOLD5000}/derivatives/fmriprep/sub-${sub}/ses-${ses}/func/sub-${sub}_ses-${ses}_task-5000scenes_run-01_bold_space-T1w_preproc.nii.gz \
      --o ${SUBJECTS_DIR}/sub-${sub}/mri/cortex_func_ses-${ses}.nii
    fi
    #create session-specific register.dat file: func -> freesurfer vol
    if [ ! -f $SUBJECTS_DIR/sub-$sub/mri/ses-${ses}_register.dat ]; then
      tkregister2 --mov $SUBJECTS_DIR/sub-$sub/mri/cortex_func_ses-${ses}.nii \
      --fstarg --regheader --reg $SUBJECTS_DIR/sub-$sub/mri/ses-${ses}_register.dat --s sub-$sub --noedit
    fi
  done

  # reslice ROIs in functional space and also project to surface.
  for roi in ${rois[@]}; do

    mri_vol2vol --mov ${BOLD5000}/derivatives/spm/sub-${sub}/sub-${sub}_mask-LH${roi}.nii.gz \
    --regheader \
    --targ ${BOLD5000}/derivatives/fmriprep/sub-${sub}/ses-01/func/sub-${sub}_ses-01_task-5000scenes_run-01_bold_space-T1w_preproc.nii.gz \
    --o ${SUBJECTS_DIR}/sub-${sub}/mri/${roi}_LH_aligned.nii

    mri_vol2vol --mov ${BOLD5000}/derivatives/spm/sub-${sub}/sub-${sub}_mask-RH${roi}.nii.gz \
    --regheader \
    --targ ${BOLD5000}/derivatives/fmriprep/sub-${sub}/ses-01/func/sub-${sub}_ses-01_task-5000scenes_run-01_bold_space-T1w_preproc.nii.gz \
    --o ${SUBJECTS_DIR}/sub-${sub}/mri/${roi}_RH_aligned.nii

    mri_vol2surf --mov ${SUBJECTS_DIR}/sub-${sub}/mri/${roi}_LH_aligned.nii \
    --reg $SUBJECTS_DIR/sub-${sub}/mri/spm_register.dat --hemi lh --surf midthickness \
    --out $SUBJECTS_DIR/sub-${sub}/label/lh.${roi}.gii

    mri_vol2surf --mov ${SUBJECTS_DIR}/sub-${sub}/mri/${roi}_RH_aligned.nii \
    --reg $SUBJECTS_DIR/sub-${sub}/mri/spm_register.dat --hemi rh --surf midthickness \
    --out $SUBJECTS_DIR/sub-${sub}/label/rh.${roi}.gii
  done
done
