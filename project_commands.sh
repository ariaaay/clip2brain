MODELS="
clip \
resnet50_bottleneck \
clip_visual_resnet \
clip_text \
bert_layer_13 \
YFCC_clip \
YFCC_simclr \
YFCC_slip \
laion2b_clip \
laion400m_clip"

for subj in {1..8}; do
    echo "processing subj $subj"
    # extract trial ID list
    python src/extract_image_list.py --subj $subj --type trial
    python src/extract_image_list.py --subj $subj --type cocoId

    # prepare brain voxels for encoding models:
    #   - extract cortical mask;
    #   - mask volume metric data;
    #   - zscore data by runs

    python src/extract_cortical_voxel.py --zscore_by_run --subj $subj

    # extract ROI mask to apply on cortical data
    python src/extract_cortical_voxel.py --subj $subj --mask_only --roi prf-eccrois
    python src/extract_cortical_voxel.py --subj $subj --mask_only --roi prf-visualrois
    python src/extract_cortical_voxel.py --subj $subj --mask_only --roi floc-faces
    python src/extract_cortical_voxel.py --subj $subj --mask_only --roi floc-words
    python src/extract_cortical_voxel.py --subj $subj --mask_only --roi floc-places
    python src/extract_cortical_voxel.py --subj $subj --mask_only --roi floc-bodies
    python src/extract_cortical_voxel.py --subj $subj --mask_only --roi Kastner2015
    python src/extract_cortical_voxel.py --subj $subj --mask_only --roi HCP_MMP1


    # computer explainable variance for the data and output data averaged by repeats
    python src/compute_ev.py --subj $subj --zscored_input

    # extract model features
    python src/extract_clip_features.py --subj $subj
    python src/extract_features_across_models.py --dataset YFCC --model simclr --subj $subj
    python src/extract_features_across_models.py --dataset YFCC --model clip --subj $subj
    python src/extract_features_across_models.py --dataset YFCC --model slip --subj $subj
    python src/extract_features_across_models.py --dataset laion2b --model clip --subj $subj
    python src/extract_features_across_models.py --dataset laion400m --model clip --subj $subj

    # run encoding model and their permutation test
    for model in $MODELS; do
        python src/run_modeling.py --model $model --subj $subj --fix_testing --test
    done

    # run joint model for variance partitioning
    python src/run_modeling.py --model "clip_visual_resnet" "resnet50_bottleneck" --subj $subj --fix_testing --test
    python src/run_modeling.py --model "clip_text" "bert_layer_13" --subj $subj --fix_testing --test
    python src/run_modeling.py --model "YFCC_slip" "YFCC_simclr" --subj $subj --fix_testing --test
    python src/run_modeling.py --model "laion2b_clip" "laion400m_clip" --subj 5 --fix_testing --test
    python src/run_modeling.py --model "clip" "laion400m_clip" --subj 5 --fix_testing --test

    # layerwise results
    for layer in {0..11}; do
        echo "running "visual_layer_${layer}" on subject ${subj}"
        python src/run_modeling.py --model "visual_layer_${layer}" --subj $subj --fix_testing
    done

    #processing bootstrap test results
    python src/analyze_clip_results.py --process_bootstrap_results --subj $subj

    # visualizing results
    python code/visualize_in_pycortex.py --subj $subj --mask_sig --sig_method fdr --vis_method quickflat
    python code/visualize_in_pycortex.py --subj $subj --mask_sig --sig_method fdr --vis_method 3d_views

done

python src/analyze_clip_results.py --performance_analysis_by_roi --group_analysis_by_roi --summary_statistics --clip_rsq_across_subject
python src/analyze_clip_results_with_PCA.py --best_voxel_n 20000 --model clip --group pca_analysis --pc-image_visualization

