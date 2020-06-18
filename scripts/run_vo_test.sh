DATASET_DIR=/media/roit/hard_disk_2/Datasets/kitti_odometry_color/sequences/
OUTPUT_DIR=/home/roit/Desktop/mc_out/

POSE_NET=/home/roit/models/SCBian/exp_pose_model_best.pth.tar

# save the visual odometry results to "results_dir/09.txt"
python test_vo.py \
--img-height 256 --img-width 832 \
--sequence 09 \
--pretrained-posenet $POSE_NET --dataset-dir $DATASET_DIR --output-dir $OUTPUT_DIR

# show the trajectory with gt. note that use "-s" for global scale alignment
evo_traj kitti -s /home/roit/Desktop/sc_out09.txt --ref=./kitti_eval/09.txt -p --plot_mode=xz

