export CUDA_VISIBLE_DEVICES=0,1,2
export HYDRA_FULL_ERROR=1

# Add PyTorch bundled CUDA libs to LD_LIBRARY_PATH (fixes missing libnvrtc.so)
_torch_lib=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null)
if [ -n "$_torch_lib" ]; then
    export LD_LIBRARY_PATH="$_torch_lib:${LD_LIBRARY_PATH}"
fi

###################################
# User Configuration Section
###################################
# Set environment variables
export NUPLAN_DEVKIT_ROOT="/data3/yuzhuoyi/AD/DiffusionPlanner/nuplan-devkit"   # e.g. "/data3/yuzhuoyi/nuplan-devkit"
export NUPLAN_DATA_ROOT="/data3/yuzhuoyi/AD/DiffusionPlanner/nuplan-devkit/nuplan/dataset"               # e.g. "/data3/yuzhuoyi/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/data3/yuzhuoyi/AD/DiffusionPlanner/nuplan-devkit/nuplan/dataset/maps"               # e.g. "/data3/yuzhuoyi/nuplan/dataset/maps"
export NUPLAN_EXP_ROOT="/data3/yuzhuoyi/AD/DiffusionPlanner/nuplan-devkit/nuplan/dataset/exp"                 # e.g. "/data3/yuzhuoyi/nuplan/exp"

# Dataset split to use
# Options:
#   - "test14-random"
#   - "test14-hard"
#   - "val14"
SPLIT="val14"

# Challenge type
# Options:
#   - "closed_loop_nonreactive_agents"   (NR: 周围车辆不对 ego 做出反应)
#   - "closed_loop_reactive_agents"      (R:  周围车辆会对 ego 做出反应)
CHALLENGE="closed_loop_nonreactive_agents"
###################################


BRANCH_NAME=diffusion_planner_release
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

###################################
# Checkpoint Configuration
# 每次评估前修改 CKPT_DIR 指向对应的训练结果目录
CKPT_NAME="training_divide_v3_128D/2026-04-22-18:13:32"
CKPT_DIR="$SCRIPT_DIR/yzy_output/training_log/$CKPT_NAME"
###################################

ARGS_FILE=$CKPT_DIR/args.json
CKPT_FILE=$CKPT_DIR/latest.pth

if [ "$SPLIT" == "val14" ]; then
    SCENARIO_BUILDER="nuplan"
else
    SCENARIO_BUILDER="nuplan_challenge"
fi
echo "Processing $CKPT_FILE..."
FILENAME=$(basename "$CKPT_FILE")
FILENAME_WITHOUT_EXTENSION="${FILENAME%.*}"

PLANNER=diffusion_planner

python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
    +simulation=$CHALLENGE \
    planner=$PLANNER \
    planner.diffusion_planner.config.args_file=$ARGS_FILE \
    planner.diffusion_planner.ckpt_path=$CKPT_FILE \
    scenario_builder=$SCENARIO_BUILDER \
    scenario_filter=$SPLIT \
    experiment_uid=$PLANNER/$SPLIT/$BRANCH_NAME/$(echo $CKPT_NAME | tr '/: ' '___')_$(TZ='Asia/Shanghai' date "+%Y-%m-%d-%H-%M-%S") \
    verbose=true \
    worker=ray_distributed \
    worker.threads_per_node=48 \
    distributed_mode='SINGLE_NODE' \
    number_of_gpus_allocated_per_simulation=0.15 \
    enable_simulation_progress_bar=true \
    hydra.searchpath="[pkg://diffusion_planner.config.scenario_filter, pkg://diffusion_planner.config, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments  ]"
