#!/bin/bash
#SBATCH --qos=unkillable                      # Ask for unkillable job
#SBATCH --cpus-per-task=1                     # Ask for 2 CPUs
#SBATCH --gres=gpu:2                          # Ask for 4 GPU
#SBATCH --mem=12G                             # Ask for 10 GB of RAM
#SBATCH --time=18:00:00                       # The job will run for 
#SBATCH -o /network/tmp1/<user>/slurm-%j.out  # Write the log on tmp1

# 1. Load your environment
# conda activate maskrcnn_benchmark

cd /network/home/bhattdha/maskrcnn-benchmark

echo "COPYING DATA."
# 2. Copy your dataset on the compute node
cp -r /network/tmp1/bhattdha/coco_dataset_new/ $SLURM_TMPDIR

# 3. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
export NGPUS=2
ln -s $SLURM_TMPDIR/coco_dataset_new/val2017_modified/ datasets/coco/val2017_new
ln -s $SLURM_TMPDIR/coco_dataset_new/train2017_modified/ datasets/coco/train2017_new
ln -s $SLURM_TMPDIR/coco_dataset_new/annotations_modified/instances_val2017_modified.json datasets/coco/annotations/instances_val2017_new.json
ln -s $SLURM_TMPDIR/coco_dataset_new/annotations_modified/instances_train2017_modified.json datasets/coco/annotations/instances_train2017_new.json
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "configs/new_train_models/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x_caffe2.yaml" SOLVER.IMS_PER_BATCH 2 SOLVER.MAX_ITER 360000 SOLVER.STEPS "(120000, 360000)" TEST.IMS_PER_BATCH 2 MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 2000 OUTPUT_DIR /network/tmp1/bhattdha/coco_dataset_new/

# 4. Copy whatever you want to save on $SCRATCH
# cp $SLURM_TMPDIR/ /network/tmp1/<user>/