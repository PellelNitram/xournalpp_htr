# TODO: Store log into file as script itself does not capture the log itself

set -e # To catch all errors

# ========
# Settings
# ========

BASE_PATH=experiments

# ============
# Experiment 1
# ============

# Question: General hyperparameter tuning

EPOCH_MAX=3

for LEARNING_RATE in 0.0005 0.001 0.002
do
    for BATCH_SIZE in 16 32 64 128
    do

        echo "LR=${LEARNING_RATE}, BS=${BATCH_SIZE}"

        python run_training.py \
            --learning_rate ${LEARNING_RATE} \
            --batch_size ${BATCH_SIZE} \
            --output_path ${BASE_PATH}/experiment1/lr${LEARNING_RATE}_bs${BATCH_SIZE} \
            --epoch_max ${EPOCH_MAX} \

    done
done

# ==================
# Future experiments
# ==================

# Other questions to answer by conducting additional experiments:
# - Do different model seeds change results?
# - Does batch size matter? -> already covered in general hyperparameter tuning
# - Longer training help w/ more patience?
# - Cheap as k-fold on data to get good estimate of true performance