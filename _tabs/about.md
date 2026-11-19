---
title: About
icon: fas fa-info-circle
order: 4
description: About Karthik Balasubramanian, a Software Engineer at Adobe moving into LLM inference and GPU kernel engineering.
---

<!-- Avatar: /assets/img/avatar.jpg (candidate headshot, pending approval). -->

I'm **Karthik Balasubramanian**, a Software Engineer at Adobe. I've worked at Adobe since 2017, starting at Marketo
(an Adobe company), with a background in **site reliability, tools and infrastructure**.

My infrastructure work includes running thousands of scheduled, containerized ETL jobs on Kubernetes and Mesos, building GitOps
CI/CD with Jenkins and Helm, and setting up logging, monitoring and alerting with Prometheus, Grafana, OpenTelemetry
and Splunk. I'm a **Certified Kubernetes Administrator**. More recently I've shipped agentic systems:
LangGraph agents, agent harnesses and loops, research into isolated sandboxes for executing agent code, and agent
evaluation using the *analyze → measure → improve* method from Hamel Husain and Shreya Shankar.

## What I'm working on now

I'm moving down the stack to where LLM cost and latency are decided: **GPU kernels and inference engines**.

- **AI Kernel Engineering** with Vizuara AI (in progress): Triton and CUDA kernels, the GPU memory hierarchy, and FlashAttention-style fusion.
- **Build a Large Language Model (From Scratch)** by Sebastian Raschka, with all my code in
  [pytorch_tutorials](https://github.com/karthikBalasubramanian/pytorch_tutorials).
- **A weekly blog** on kernels, attention, KV cache, continuous batching and speculative decoding, and on serving with
  **vLLM** and **SGLang**. I benchmark each topic the way an SRE would, with p50/p99 latency, time to first token (TTFT),
  tokens/s and GPU utilization.

I think the gap between kernel engineers and production infrastructure engineers is where inference engineering happens,
and I'm building skills on both sides.

## Skills

| Area | Tools |
|---|---|
| Languages | Python, Java |
| ML & LLMs | PyTorch, Transformers, LLM evaluation, agentic frameworks (LangGraph), A2A protocol |
| Learning now | Triton, CUDA, vLLM, SGLang |
| Infrastructure | Kubernetes (CKA), Docker, Terraform, Airflow, FastAPI, microservices |
| Observability | Prometheus, Grafana, OpenTelemetry, Splunk |
| Data | PostgreSQL, MySQL, MS SQL Server, Azure Data Explorer, Redis, Databricks |

## Education

- **Master of Information Systems Management**, Heinz College, Carnegie Mellon University (2017)
- **B.Tech, Information Technology**, SRM University, Chennai (2013)

## Certifications

- Certified Kubernetes Administrator (CKA), The Linux Foundation (2024)
- AI Evals for Engineers & PMs, Maven (Hamel Husain & Shreya Shankar) (2026)
- Cloud DevOps Engineer Nanodegree, Udacity (2020)
- Data Engineering Nanodegree, Udacity (2020)
- Machine Learning Engineer Nanodegree, Udacity (2019)

## Elsewhere

- GitHub: [karthikBalasubramanian](https://github.com/karthikBalasubramanian)
- LinkedIn: [bks4line](https://www.linkedin.com/in/bks4line)
- [RSS feed]({{ '/feed.xml' | relative_url }})
