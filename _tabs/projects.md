---
title: Projects
icon: fas fa-code
order: 2
description: Open-source projects by Karthik Balasubramanian, from building an LLM from scratch to ML serving on Kubernetes.
---

Code I've written in public, newest first. Every project links to its GitHub repository.

## LLM and inference work

### [pytorch_tutorials](https://github.com/karthikBalasubramanian/pytorch_tutorials): building an LLM from scratch
My working notebook for Sebastian Raschka's *Build a Large Language Model (From Scratch)*, plus PyTorch fundamentals.
So far it covers:

- a simple tokenizer and BPE with `tiktoken`
- a sliding-window GPT dataset, plus token and positional embeddings
- self-attention, then causal attention, then multi-head attention
- a full `GPTModel` with LayerNorm, feed-forward blocks and weight tying, and autoregressive text generation
- CPU vs Apple Silicon (MPS) benchmarks and a Distributed Data Parallel (DDP) training example

Write-ups: [The Flashlight and the Shadow: dot products and attention]({{ '/ml/2026/09/02/utility_of_dot_products_in_attention.html' | relative_url }}).

### [jev-tutorials](https://github.com/karthikBalasubramanian/jev-tutorials): calibrated decision models for support triage
A Node.js, Express and TypeScript app (Vercel AI SDK) that demonstrates TypeSafe AI Jev's non-generative
decision primitives (Choice, Score and Boolean). It runs them in a streaming retail-support ticket pipeline with a policy engine.

### Coming next: GPU kernels lab
Triton and CUDA kernels written while I work through Vizuara AI's AI Kernel Engineering program:
fused softmax, tiled matmul, and a minimal FlashAttention forward pass, each benchmarked against PyTorch.
*(Repository not published yet.)*

## Infrastructure and MLOps

### [project-ml-microservice-kubernetes](https://github.com/karthikBalasubramanian/project-ml-microservice-kubernetes): ML inference API on Kubernetes
Takes a pre-trained scikit-learn model served by Flask and operationalizes it. The steps are linting,
a Docker image, a Kubernetes deployment, structured logging and CircleCI. Built for the Udacity Cloud DevOps Engineer Nanodegree.

### [cloud-devops-project](https://github.com/karthikBalasubramanian/cloud-devops-project)
Project from the Udacity Cloud DevOps Engineer Nanodegree.

## Earlier machine learning projects

- [FER](https://github.com/karthikBalasubramanian/FER): facial emotion recognition with heuristic and transfer learning
  ([write-up]({{ '/ml/2019/05/17/Karthik-_FER_Capstone_Report.html' | relative_url }})).
- [Airbnb](https://github.com/karthikBalasubramanian/Airbnb): price prediction for Airbnb listings.
- [SparkGen](https://github.com/karthikBalasubramanian/SparkGen): a React tool that generates Apache Spark job configurations, built during my DataSpark internship.
- [Altair-Viz](https://github.com/karthikBalasubramanian/Altair-Viz): companion code for my Altair visualization tutorial
  ([write-up]({{ '/ml/2017/01/20/AltairViz.html' | relative_url }})).

## Tools

I've also built a set of [personal finance calculators]({{ '/tools/' | relative_url }}) in React and TypeScript.
