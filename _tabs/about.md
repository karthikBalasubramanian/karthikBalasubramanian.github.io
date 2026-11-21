---
title: About
icon: fas fa-info-circle
order: 4
description: About Karthik Balasubramanian, a Software Engineer at Adobe moving into LLM inference and GPU kernel engineering.
---

<!-- Avatar: /assets/img/avatar.jpg (approved headshot). -->

I'm **Karthik Balasubramanian**, a Software Engineer at Adobe. I've worked at Adobe since 2017, starting at Marketo
(an Adobe company), with a background in **site reliability, tools and infrastructure**.

My engineering background spans:

- **Production infrastructure & Kubernetes:** Running thousands of scheduled, containerized ETL jobs on Kubernetes and Mesos, building GitOps CI/CD with **Argo CD**, Jenkins, and Helm, working with **Argo Workflows** and progressive delivery via **Argo Rollouts** (Rolling Update, Canary, Blue-Green), and maintaining **Certified Kubernetes Administrator (CKA)** depth.
- **Observability & reliability:** Logging, metrics, tracing, and alerting across Prometheus, Grafana, OpenTelemetry, and Splunk.
- **Agentic systems & evaluation:** Shipping LangGraph agents, agent harnesses and loops, researching isolated sandboxes (e.g., Agent Sandbox) for safe code execution, and evaluating agents using the *analyze → measure → improve* methodology (Hamel Husain & Shreya Shankar).

## What I'm working on now

I'm moving down the stack to where LLM cost and latency are decided: **GPU kernels and inference engines**.

- **AI Kernel Engineering** with Vizuara AI (in progress): Triton and CUDA kernels, the GPU memory hierarchy, and FlashAttention-style fusion.
- **Build a Large Language Model (From Scratch)** by Sebastian Raschka, with all my code in
  [pytorch_tutorials](https://github.com/karthikBalasubramanian/pytorch_tutorials).
- **Technical writing** on kernels, attention, KV cache, continuous batching and speculative decoding, and on serving with
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
| Infrastructure | Kubernetes (CKA), Argo (CD, Workflows, Rollouts), Helm, Docker, Terraform, Airflow, FastAPI, microservices |
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

## Elsewhere & Contact

I'm focused on **ML Inference Engineering** and **GPU Systems / ML Infrastructure** roles where production reliability meets low-level performance. The best way to reach me is on LinkedIn.

- GitHub: [karthikBalasubramanian](https://github.com/karthikBalasubramanian)
- LinkedIn: [bks4line](https://www.linkedin.com/in/bks4line)
- [RSS feed]({{ '/feed.xml' | relative_url }})
