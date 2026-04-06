---

title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
-------------

# Email Triage OpenEnv

LLM-powered email triage agent for the OpenEnv Hackathon.

## Overview

Processes emails in the `email-triage-v1` environment and outputs structured decisions.

## Output Format

```text id="fmt1"
[START] task=<task_name> env=email-triage-v1 model=<model_name>
[STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>
```

## Environment Variables

* API_BASE_URL=https://router.huggingface.co/v1
* MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
* HF_TOKEN=your_token_here

## Docker

```bash id="fmt2"
docker build -t email-triage-env .
docker run -p 7860:7860 email-triage-env
```
