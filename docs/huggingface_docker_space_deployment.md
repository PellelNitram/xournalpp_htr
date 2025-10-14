## Local Docker image building

1. Build the Docker image: `docker build -t xournalpp_htr .`
2. Run Docker image: `docker run -d -p 7860:7860 xournalpp_htr`
    - Interactively for debugging: `docker run -it --entrypoint bash xournalpp_htr`
3. Run Docker image for interactive development
    - Start docker container: `docker run -it -p 7860:7860 -v $(pwd):/temp_code_mount --entrypoint bash xournalpp_htr`
    - Call Python code inside the container: `python /temp_code_mount/scripts/demo.py`

Generally, tidy up Docker caches with `docker system prune` if your system is full.

## looking into adding xournalpp to the image b/c i need that for the prediction (to convert xoj/xopp to pdf):

now cross compiled on M4
- build image: `docker buildx build --platform linux/amd64 -t xournalpp_htr .`
- interactively entering: `docker run -it --platform linux/amd64 -p 7860:7860 -v $(pwd):/temp_code_mount --entrypoint bash xournalpp_htr`
- dl deb file: `wget --no-check-certificate https://github.com/xournalpp/xournalpp/releases/download/v1.2.8/xournalpp-1.2.8-Debian-bookworm-x86_64.deb`
    - there're issues!!
- alternative: use appimage:
    - `wget --no-check-certificate https://github.com/xournalpp/xournalpp/releases/download/v1.2.8/xournalpp-1.2.8-x86_64.AppImage`
