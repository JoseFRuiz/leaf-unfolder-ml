#!/bin/bash
#SBATCH --job-name=leaf-unfolding
#SBATCH --output=leaf-unfolding.out
#SBATCH --error=leaf-unfolding.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jfruizmu@unal.edu.co
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --mem-per-cpu=20000mb
#SBATCH --distribution=cyclic:cyclic
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=96:00:00

echo "Date      = $(date)"
echo "host      = $(hostname -s)"
echo "Directory = $(pwd)"

module load cuda/11.1

source activate audio-classification

# Run the U-Net experiment
python run_unet_experiment.py \
    --learning_rate 1e-4 \
    --batch_size 8 \
    --max_epochs 100 \
    --img_size 256 \
    --padding 300 \
    --save_examples_every_n_epochs 10 \
    --resume_from_checkpoint unet_checkpoints/unet-leaf-unfolding-epoch=99-val_loss=0.15.ckpt

conda deactivate
