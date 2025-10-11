## Local Docker image building

1. Build the Docker image: `docker build -t xournalpp_htr .`
2. Run Docker image: `docker run -d -p 7860:7860 xournalpp_htr`
    - Interactively for debugging: `docker run -it --entrypoint bash xournalpp_htr`
3. Run Docker image for interactive development
    - Start docker container: `docker run -it -p 7860:7860 -v $(pwd):/temp_code_mount --entrypoint bash xournalpp_htr`
    - Call Python code inside the container: `python /temp_code_mount/scripts/demo.py`