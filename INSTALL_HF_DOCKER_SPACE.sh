# Based on `INSTALL_LINUX.sh` file.

# ========
# SETTINGS
# ========

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

install_htr_pipeline
cd ${CURRENT_DIR}
pip install -r requirements.txt
pip install gradio # TODO: Move to optional package in `pyproject.toml` once I use this setup.
pip install -e .

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