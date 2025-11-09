Our [online demo](https://huggingface.co/spaces/PellelNitram/xournalpp_htr) allows users to experiment with Xournal++ HTR without any installation and thereby lowers the entry barrier.

We use a Hugging Face Docker Space setup to deliver the [online demo](https://huggingface.co/spaces/PellelNitram/xournalpp_htr). The deployment is automated using Github Actions, see [this workflow](https://github.com/PellelNitram/xournalpp_htr/blob/master/.github/workflows/deploy-demo-to-hf-space.yml).

This page outlines how to set up and develop the demo locally, ensuring compatibility with Hugging Face Docker Space deployment.

## Local development using Docker

1. Build the Docker image from the repository root folder: `docker build -t xournalpp_htr .`
2. Run Docker image: `docker run -d -p 7860:7860 xournalpp_htr`
3. Run Docker image for interactive development
    - Start docker container: `docker run -it -p 7860:7860 -v $(pwd):/temp_code_mount --entrypoint bash xournalpp_htr`
    - Call Python code inside the container: `python /temp_code_mount/scripts/demo.py`

Generally, tidy up Docker caches with `docker system prune` if your system is full.

## Production deployment using Github Actions

Once code has been pushed to the `master` branch, it is picked up by CI/CD in the form of Github Actions ([see code here](https://github.com/PellelNitram/xournalpp_htr/blob/master/.github/workflows/deploy-demo-to-hf-space.yml)) and the code is automatically deployed to a Hugging Face Docker Space [here](https://huggingface.co/spaces/PellelNitram/xournalpp_htr).

## Environment variables

The demo needs a number of environment variables to work correctly, see below and in the `.env.example` file: 

```bash
DEMO=1
SB_URL="https://<add here>.supabase.co"
SB_KEY="<add here>"
SB_BUCKET_NAME="xournalpp_htr_hf_space"
SB_SCHEMA_NAME="public"
SB_TABLE_NAME="xournalpp_htr_hf_space_events"
```

Here, demo mode should be disabled for the production environment, i.e. `DEMO=0`.

## Supabase setup

Supabase stores both analytics data and donated data samples.

Create the events table for analytics:

```sql
create table public.xournalpp_htr_hf_space_events (
  id bigserial primary key,
  timestamp timestamptz not null,
  demo boolean not null,
  session_id text not null,
  donate_data bool not null,
  interaction text not null
);
```

Create bucket with following name to store donated data samples:

```
xournalpp_htr_hf_space
```