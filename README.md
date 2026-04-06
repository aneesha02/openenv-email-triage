---

title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: docker
sdk_version: "1.0"
app_file: app.py
pinned: false
-------------

# 📧 Email Triage OpenEnv

This project implements an LLM-powered email triage agent for the OpenEnv Hackathon.

## Overview

The system processes emails in the email-triage-v1 environment and outputs structured decisions per step.

## Output Format

```
[START] task=<task_name> env=email-triage-v1 model=<model_name>
[STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>
```

## Environment Variables

* API_BASE_URL=https://router.huggingface.co/v1
* MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
* HF_TOKEN=your_token_here

HF_TOKEN is required for execution.

## Docker

```
docker build -t email-triage-env .
docker run -p 7860:7860 email-triage-env
```

Open: http://localhost:7860
