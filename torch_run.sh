export CUDA_VISIBLE_DEVICES=0,2

###################################
# User Configuration Section
###################################
RUN_PYTHON_PATH="/data3/yuzhuoyi/miniconda3/envs/diffusion_planner/bin/python" # python path (e.g., "/home/xxx/anaconda3/envs/diffusion_planner/bin/python")

# Set training data path
TRAIN_SET_PATH="/data3/yuzhuoyi/AD/DiffusionPlanner/Diffusion-Planner/nuplan_diffusionplanner_large" # preprocess data using data_process.sh
TRAIN_SET_LIST_PATH="/data3/yuzhuoyi/AD/DiffusionPlanner/Diffusion-Planner/diffusion_planner_training.json"
###################################

$RUN_PYTHON_PATH -m torch.distributed.run --nnodes 1 --nproc_per_node 2 --standalone train_predictor.py \
--train_set       $TRAIN_SET_PATH \
--train_set_list  $TRAIN_SET_LIST_PATH \
--vocab_path      /data3/yuzhuoyi/AD/DiffusionPlanner/Diffusion-Planner/vocab/ego_vocab_1024.npz \
--nbr_vocab_path  /data3/yuzhuoyi/AD/DiffusionPlanner/Diffusion-Planner/vocab/nbr_vocab_1024.npz \
--save_dir        /data3/yuzhuoyi/AD/DiffusionPlanner/Diffusion-Planner/yzy_output \
--name            training_divide_v4_128D \
--save_utd        40 \
--batch_size      128 \
--num_workers     44 \
--train_epochs    1000 \
--token_emb_dim   128 \
--alpha_planning_loss 5.0 \
--ddp True



