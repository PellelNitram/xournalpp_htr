# ========
# SETTINGS
# ========

ENVIRONMENT_NAME="xournalpp_htr_testing"
HTR_PIPELINE_PATH="external/htr_pipeline"

# ====================
# Installation process
# ====================

eval "$(conda shell.bash hook)" # enable `conda activate`, see
                                # https://stackoverflow.com/a/56155771

# conda create --name ${ENVIRONMENT_NAME} python=3.10.11 -y
conda activate ${ENVIRONMENT_NAME}
# pip install -r requirements.txt
# pip install -e .

# Install HTRPipeline
mkdir -p ${HTR_PIPELINE_PATH}
cd ${HTR_PIPELINE_PATH}
git clone https://github.com/githubharald/HTRPipeline.git
cd HTRPipeline
pip install .
# 3. Install [HTRPipelines](https://github.com/githubharald/HTRPipeline) package using [its installation guide](https://github.com/githubharald/HTRPipeline/tree/master#installation).

# ========
# Feedback
# ========

# TODO: Activate w/ "conda activate ${ENVIRONMENT_NAME}"
# TODO: Download weights from HTRPipeline