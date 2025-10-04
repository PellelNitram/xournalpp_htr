# ========
# SETTINGS
# ========

ENVIRONMENT_NAME="xournalpp_htr"
HTR_PIPELINE_PATH="external/htr_pipeline"

# ================
# Helper functions
# ================

install_htr_pipeline () {

    mkdir -p ${HTR_PIPELINE_PATH}
    cd ${HTR_PIPELINE_PATH}
    git clone https://github.com/githubharald/HTRPipeline.git
    cd HTRPipeline
    cd htr_pipeline/models
    wget https://www.dropbox.com/s/j1hl6bppecug0sz/models.zip
    unzip -o models.zip
    cd ../../
    pip install .
    # 3. Install [HTRPipelines](https://github.com/githubharald/HTRPipeline) package using [its installation guide](https://github.com/githubharald/HTRPipeline/tree/master#installation).

}

CURRENT_DIR=$(pwd)

# ====================
# Installation process
# ====================

rm -rf ${HTR_PIPELINE_PATH}

eval "$(conda shell.bash hook)" # enable `conda activate`, see
                                # https://stackoverflow.com/a/56155771

conda create --name ${ENVIRONMENT_NAME} python=3.10.11 -y
conda activate ${ENVIRONMENT_NAME}
install_htr_pipeline
cd ${CURRENT_DIR}
pip install -r requirements.txt
pip install -e .
pre-commit install

cd plugin
bash copy_to_plugin_folder.sh

# ========
# Feedback
# ========

echo
echo "==========================================="
echo "==========================================="
echo "==========================================="
echo
echo "Installation complete"
echo
echo "Activate environment with:"
echo "\"conda activate ${ENVIRONMENT_NAME}\""
echo