---
layout: page
title: Résumé
permalink: /resume/
description: Résumé of Karthik Balasubramanian, Software Engineer at Adobe (infrastructure, observability, Kubernetes), moving into LLM inference and GPU kernel engineering.
---

<!--
  DRAFT built from verified sources only: Karthik's public résumé PDF
  (github.com/karthikBalasubramanian/resume, an older version), his LinkedIn certifications, and the skills he listed himself.
  Pending: replace with his current résumé and add a PDF download link (e.g. /assets/resume.pdf).
-->

**Karthik Balasubramanian**: Software Engineer, infrastructure → LLM inference & GPU kernels
· [GitHub](https://github.com/karthikBalasubramanian) · [LinkedIn](https://www.linkedin.com/in/bks4line)

## Experience

### Adobe: Software Engineer, Digital Experience
*2019 – present*

- Responsible for the data infrastructure behind the Product & Customer Analytics team. That means terabyte-scale warehouses
  across multiple clouds, and thousands of scheduled ETL jobs running in hundreds of Kubernetes pods and Mesos containers.
- Built logging and alert monitoring for warehouse and ETL workloads, so engineers and data scientists catch data errors early.
- Built GitOps CI/CD with Jenkins, Helm charts and Adobe's internal Kubernetes onboarding platform. It cut ETL deployment
  time into Airflow from 33 minutes to 4 minutes.
- Wrote a custom Airflow operator that cut average warehouse query runtime from 50 minutes to under 9 seconds.
- Architected an internal platform that attributes product and customer usage for Adobe Digital Experience Cloud SaaS products.
- More recently: agentic systems (LangGraph agents, harnesses and agent loops), research into isolated sandboxes,
  and agent evaluation.

### Marketo, an Adobe company: Associate Software Engineer, Tools & Infrastructure
*Oct 2017 – 2019*

- Architected an Apache Airflow workflow that migrated Marketo subscriptions from an on-premises data center to Google Cloud.
- Cut the dump and restore time for a 1 TB database from 6–7 days (logical backup) to about 6.5 hours with XtraBackup,
  and fixed replication and long-running query problems on replica servers.

### SingTel DataSpark Analytics, Singapore: Data Engineering Intern
*Summer 2016*

- Built an interactive Apache Spark job configuration tool from scratch with React and Spring Data.

### Tata Consultancy Services, Chennai: Assistant Systems Engineer
*Jun 2013 – May 2015*

- Delivered a large data-integration project for a US retailer. Received the TCS Technical Excellence Award.

## Currently learning

- AI Kernel Engineering, Vizuara AI (in progress): Triton, CUDA, GPU memory hierarchy, FlashAttention
- *Build a Large Language Model (From Scratch)*, Sebastian Raschka: [code](https://github.com/karthikBalasubramanian/pytorch_tutorials)

## Skills

Python, Java · Kubernetes, Docker, Terraform, Airflow, FastAPI, microservices, A2A protocol · Prometheus, Grafana,
OpenTelemetry, Splunk · PostgreSQL, MySQL, MS SQL Server, Azure Data Explorer, Redis, Databricks · PyTorch, LangGraph

## Education

- Master of Information Systems Management, Heinz College, Carnegie Mellon University (May 2017)
- B.Tech, Information Technology, SRM University, Chennai (Apr 2013)

## Certifications

Certified Kubernetes Administrator (2024) · AI Evals for Engineers & PMs, Maven (2026) ·
Udacity Nanodegrees: Cloud DevOps Engineer (2020), Data Engineering (2020), Machine Learning Engineer (2019)
