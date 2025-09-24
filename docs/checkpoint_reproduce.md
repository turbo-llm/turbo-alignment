## ♻️ Checkpointing & Reproducibility

### 1️⃣ Configure the trainer to save *complete* checkpoints  
Add the block below to your experiment JSON (or keep these keys unchanged
if they already exist). Setting `save_only_model` to **false** guarantees
that the *entire* training state—model, optimizer, scheduler, RNG seeds—is
written to disk.

```json
{
  {
    // … other trainer settings
    "save_only_model": false
  },
  "checkpoint_uploader_callback_parameters": {
      "directory": "test/reproduce-ckpt-dir-test"
  },
}
```

### 2️⃣ Mount a local folder as /checkpoints inside the container
During training the library will write sub-folders named checkpoint-*
(identical to Hugging Face format) into this directory.
Make sure you mount a persistent volume so the files survive container restarts:

```bash
# Host → Container
/local/ckpt/dir  ➜  /checkpoints
```
Expected layout after a few save steps with example `"save_total_limit": 2` and `"save_steps": 4`:

```bash
/checkpoints/
├─ checkpoint-52/
└─ checkpoint-56/
```

### 3️⃣ Provide S3 (or MinIO) credentials via environment variables

```bash
export S3_CHECKPOINTS_BUCKET="bucket"          # target bucket
export S3_CHECKPOINTS_HOST="endpoint"          # e.g. https://s3.us-west-1.amazonaws.com
export S3_CHECKPOINTS_AWS_ACCESS_KEY_ID="key"  # access key
export S3_CHECKPOINTS_AWS_SECRET_ACCESS_KEY="key"  # secret key
```
