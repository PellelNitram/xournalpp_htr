## Local Docker image building

1. Build the Docker image: `docker build -t xournalpp_htr .`
2. Run Docker image: `docker run -d -p 7860:7860 xournalpp_htr`
    - Interactively for debugging: `docker run -it --entrypoint bash xournalpp_htr`