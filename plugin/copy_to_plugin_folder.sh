# ========
# SETTINGS
# ========

TARGET_FOLDER=~/.config/xournalpp/plugins/xournalpp_htr/
# TARGET_FOLDER=/usr/share/xournalpp/plugins/xournalpp_htr # requires `sudo`

# ============
# COPY PROCESS
# ============

mkdir -p ${TARGET_FOLDER}

cp plugin.ini ${TARGET_FOLDER}
cp main.lua ${TARGET_FOLDER}
cp config.lua ${TARGET_FOLDER}