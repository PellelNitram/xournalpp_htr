# Potentially needs to be executed using `sudo`

TARGET_FOLDER=~/.config/xournalpp/plugins/xournalpp_htr/
# TARGET_FOLDER=/usr/share/xournalpp/plugins/xournalpp_htr # requires `sudo`

mkdir -p ${TARGET_FOLDER}

cp plugin.ini ${TARGET_FOLDER}
cp main.lua ${TARGET_FOLDER}