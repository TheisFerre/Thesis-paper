#!/bin/sh
#BSUB -J METALEARN #The name the job will get
#BSUB -q gpuv100 #The queue the job will be committed to, here the GPU enabled queue
#BSUB -gpu "num=1:mode=exclusive_process" #How the job will be run on the VM, here I request 1 GPU with exclusive access i.e. only my c #BSUB -n 1 How many CPU cores my job request
#BSUB -W 12:00 #The maximum runtime my job have note that the queuing might enable shorter jobs earlier due to scheduling.
#BSUB -R "span[hosts=1]" #How many nodes the job requests
#BSUB -R "rusage[mem=12GB]" #How much RAM the job should have access to
#BSUB -R "select[gpu32gb]" #For requesting the extra big GPU w. 32GB of VRAM
#BSUB -o logs/OUTPUT.%J #Log file
#BSUB -e logs/ERROR.%J #Error log file
echo "Starting:"

cd ~/Thesis/metalearning
#cd /Users/theisferre/Documents/SPECIALE/Thesis/src/models

source ~/Thesis/venv-thesis/bin/activate

DATA_DIR=/zhome/2b/7/117471/Thesis/data/processed/ablation-augmented
TRAIN_SIZE=0.9
BATCH_TASK_SIZE=10
K_SHOT=5
ADAPTATION_STEPS=10
EPOCHS=150
ADAPT_LR=0.05
META_LR=0.001
EXCLUDE=citibike-tripdata-GRID,GM,TLC2018-FHV-aug-GRID,TLC2018-FHV-REGION,UBER2015-jan-june-GRID,citibike2014-tripdata-REGION,citibike2014-tripdata-GRID,UBER2015-jan-june-REGION,green,yellow-taxi2020-nov-REGION,LYFT
LOG_DIR=/zhome/2b/7/117471/Thesis/ablation-study/augmented
HIDDEN_SIZE=46
DROPOUT_P=0.2
NODE_OUT_FEATURES=10

# citibike-tripdata-GRID,GM,TLC2018-FHV-aug-GRID,TLC2018-FHV-REGION,UBER2015-jan-june-GRID,citibike2014-tripdata-REGION,citibike2014-tripdata-GRID,UBER2015-jan-june-REGION,green,yellow-taxi2020-nov-REGION,LYFT,yellow-taxi2020-nov-GRID


python /zhome/2b/7/117471/Thesis/src/models/train_meta.py --data_dir $DATA_DIR --train_size $TRAIN_SIZE --batch_task_size $BATCH_TASK_SIZE \
--k_shot $K_SHOT --adaptation_steps $ADAPTATION_STEPS --epochs $EPOCHS --adapt_lr $ADAPT_LR --meta_lr $META_LR --log_dir $LOG_DIR \
--hidden_size $HIDDEN_SIZE --dropout_p $DROPOUT_P --node_out_features $NODE_OUT_FEATURES --exclude $EXCLUDE --gpu

