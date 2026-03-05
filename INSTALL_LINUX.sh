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

}

CURRENT_DIR=$(pwd)

# ====================
# Installation process
# ====================

rm -rf ${HTR_PIPELINE_PATH}

install_htr_pipeline
cd ${CURRENT_DIR}
uv sync
uv run xournalpp-htr-configure
uv run pre-commit install

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
echo "Run commands in the environment with:"
echo "\"uv run <command>\""
echo
