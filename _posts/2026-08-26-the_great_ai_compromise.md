---
layout: post
title: "The Great AI Compromise: How Autonomous Agents Will Reclaim Our Idling GPUs"
date: 2026-08-26
desc: "An exploration of FLOPs, memory bandwidth, MFU across GPT-2 to Claude 3.5 Sonnet, and how long-horizon autonomous agents will unlock maximum GPU throughput."
keywords: "AI, LLM, GPU, FLOPs, Latency, Throughput, MFU, GPT-4, Claude Sonnet, Autonomous Agents, Systems"
categories: [Ml]
tags: [blog, AI, LLM, GPU, Machine Learning, Systems, Agents]
icon: fa-pencil
---

Imagine hiring an Olympic-class speed-reader who can process 1,000 pages every second. You sit down in front of them, hand over a single index card with one sentence, wait for them to read it, and then hand over another card five seconds later.

Your reader is capable of staggering throughput, but they spend 99% of their day twiddling their thumbs waiting for you to pass the next sheet.

That is the exact tension behind serving modern Large Language Models (LLMs). We build monstrous, multi-thousand-dollar GPUs designed to churn through quadrillions of mathematical operations every second, and then deliberately run them at a tiny fraction of their peak capability just so human beings don't get bored waiting for the next word to pop up on their screens.

Let’s dismantle this machinery from first principles: starting with a single arithmetic flick of silicon, moving to how words turn into math, examining MFU benchmarks from GPT-2 to Claude 3.5 Sonnet, and seeing why long-horizon autonomous agents will fundamentally transform AI computing economics.

---

## 1. What on Earth Is a FLOP?

Before we talk about intelligence, we have to talk about addition and multiplication.

Computers don't think in poetry; they crunch decimal numbers (floating-point numbers). A **FLOP** stands for **FLoating-point OPeration**.

* $1.234 \times 5.678$ is one FLOP.
* Adding two fractional numbers together is another FLOP.

When you see **FLOPS** with a capital **S**, it means **FLOPs per Second**—the speed limit of your hardware.

```
• 1 GFLOPS  = 10⁹   FLOPs/sec  (Giga — A decent CPU in the 2000s)
• 1 TFLOPS  = 10¹²  FLOPs/sec  (Tera — Modern laptop / desktop GPU)
• 1 PFLOPS  = 10¹⁵  FLOPs/sec  (Peta — An enterprise AI accelerator like the NVIDIA H100)
```

If a GPU has a flop rate of **1 PFLOPS (FP16/BF16)**, it can perform **one quadrillion** ($1,000,000,000,000,000$) multiplications and additions in the time it takes your heart to beat once.

---

## 2. From FLOPs to Tokens: The Math Behind the Magic

How do raw calculations turn into text?

When an LLM generates a response, it outputs **tokens**—chunks of characters or words (`"banana"` is one token, `" un"` + `"believable"` is two).

A token is the **material**; FLOPs are the **energy** required to shape it.

```
       ┌──────────────┐         Transforms via          ┌─────────────┐
       │ Input Token  │ ──────────────────────────────> │ Next Token  │
       └──────────────┘      Billions of FLOPs          └─────────────┘
                          (Matrix Multiplications)
```

The fundamental rule of thumb for standard Transformer models relates parameters ($N$) to FLOPs:

* **To generate 1 token (Inference):** $\approx 2N \text{ FLOPs}$
* *Why?* Each parameter in the network participates in one multiply-accumulate operation ($a \times b + c = 2\text{ FLOPs}$) during the forward pass.


* **To train on 1 token (Training):** $\approx 6N \text{ FLOPs}$
* *Why?* You run the forward pass ($2N$) plus the backward pass ($4N$, because calculating gradients with respect to both weights and activations takes roughly double the math).

---

## 3. Putting Real Numbers on the Table: 7B vs. 70B

Let's plug in real-world models to see what this scale actually looks like:

| Model Size ($N$) | Inference (FLOPs / Token) | Training (FLOPs / Token) | Training on 2 Trillion Tokens (Total FLOPs) |
| --- | --- | --- | --- |
| **7 Billion (7B)** | $2 \times 7\text{B} = \mathbf{14\text{ Billion}}$ | $6 \times 7\text{B} = \mathbf{42\text{ Billion}}$ | $42 \times 10^9 \times 2 \times 10^{12} = \mathbf{8.4 \times 10^{22} \text{ FLOPs}}$ |
| **70 Billion (70B)** | $2 \times 70\text{B} = \mathbf{140\text{ Billion}}$ | $6 \times 70\text{B} = \mathbf{420\text{ Billion}}$ | $420 \times 10^9 \times 2 \times 10^{12} = \mathbf{8.4 \times 10^{23} \text{ FLOPs}}$ |

If you have a 1-TFLOPS GPU ($10^{12} \text{ FLOP/s}$) and run a 7B model:

$$
\text{Theoretical Token Rate} = \frac{10^{12} \text{ FLOPs/sec}}{14 \times 10^9 \text{ FLOPs/token}} \approx 71.4 \text{ tokens/sec}
$$

On paper, producing 71 tokens per second sounds effortless. But in reality, your GPU rarely hits that number for a single user. Why?

---

## 4. The Memory Wall and MFU (Model FLOPs Utilization)

To understand why GPUs fall short of theoretical peak speeds, we track **MFU (Model FLOPs Utilization)**:

$$
\text{MFU} = \frac{\text{Actual Useful FLOPs Computed / Second}}{\text{Theoretical Peak FLOPs Capacity of Hardware}}
$$

If an H100 GPU can theoretically output ~1,000 TFLOPS, but your workload only extracts 300 TFLOPS worth of useful token generation, your **MFU is 30%**.

---

### The Master Chef and the Small Kitchen Counter

To get an intuitive mental model for MFU without getting bogged down in hardware jargon, imagine hiring a **world-class master chef**:

* **The Chef (GPU Tensor Cores):** Capable of chopping, dicing, and cooking at staggering speeds—making 1,000 knife cuts every second.
* **The Small Kitchen Counter (On-Chip SRAM Cache):** The tiny local workspace (~50–250MB) right on the GPU die—ultra-fast, but only big enough to hold one recipe page at a time.
* **The Pantry (Off-Chip VRAM / HBM):** The massive storage room down the hall (~80GB–140GB) where all 140 Gigabytes of ingredients and recipe volumes (Model Weights) live.

If you ask the chef to make **one single tiny dumpling** (generating 1 token for a single chat user):

1. The chef must walk down the hall to the pantry (VRAM/HBM), load **140 heavy volumes** of recipe books onto a cart, and wheel them back to the small kitchen counter (SRAM).
2. Because the counter is small, the chef can only lay down one page, makes **one quick knife cut** ($\approx 140\text{ billion FLOPs}$) to roll out **one single dumpling**.
3. To make the *next* dumpling, the chef must put all the books back on the cart, walk back down the hall to the pantry, load up the cart again, and wheel it back just to make one more knife cut!

The chef is capable of cutting continuously for hours, but spends 95% of their shift pushing a heavy cart back and forth between VRAM and SRAM. 

**MFU is simply the percentage of time the chef's knife is actually cutting food versus pushing the cart.** In single-user chat, MFU is a tiny **3% – 5%**—meaning 95%+ of your multi-thousand-dollar GPU's processing power is wasted doing nothing while waiting for data to travel from main VRAM to on-chip SRAM.

Now imagine **128 guests** walk into the restaurant at once and order 128 dumplings (Batch Size = 128):

The chef wheels the 140 recipe volumes from VRAM to the small SRAM counter **once**, but now with 128 dumplings lined up, the chef uses that single loaded recipe page to make 128 rapid knife cuts in sequence before needing to touch the cart again. The kitchen comes alive, the knife never stops moving, and **MFU surges to 50% – 60%**.

---

### The Roofline Bottleneck

![The Roofline Model](/static/img/blog/roofline_model.png)
*Figure 1: The Roofline Model depicting Arithmetic Intensity vs Realized Performance (FLOPs/s), highlighting Memory-Bandwidth-Bound vs Compute-Bound operational regimes.*

In hardware terms, this dynamic is formalized by the **Roofline Model**, which maps the performance relationship between ultra-fast **on-chip SRAM** (where active matrix calculations occur) and larger **off-chip VRAM / HBM** (where the 140 GB model parameters reside).

A GPU consists of two primary operational domains:

1. **Compute Engines (Tensor Cores & SRAM):** The hyper-fast calculators executing matrix multiplications.
2. **Off-Chip Memory (VRAM / HBM):** The storage shelves holding model weights and KV caches.

When generating tokens for a 70B parameter model:

* **Batch Size = 1 (Single User Chat):** The GPU has low *arithmetic intensity* (ratio of FLOPs computed per byte of memory transferred from VRAM to SRAM). Fetching 140 GB of weights from VRAM for a single token computation means the workload is completely **memory-bandwidth bound**, dropping MFU to **1% – 5%**.
* **Batch Size = 128 (High Concurrency):** Loading the 140 GB weights from VRAM to SRAM once allows computing 128 parallel token steps. Arithmetic intensity increases by 128x, fully saturating the Tensor Cores. The workload transitions to being **compute-bound**, and MFU surges to **45% – 60%**.

---

### Real-World MFU Benchmarks Across Model Generations

To see how severe this bottleneck has been across AI history, consider the real-world MFU numbers for landmark frontier models during single-user interactive streaming versus high-concurrency batch execution:

| Frontier Model | Active Parameters / Architecture | Single-User Streaming MFU (Batch=1) | Asynchronous / High-Batch MFU (Batch=128+) | Throughput Gain (Batch vs Stream) |
| --- | --- | --- | --- | --- |
| **GPT-2 (1.5B)** | 1.5B Dense | ~1% – 3% | ~25% – 35% | **10x – 12x** |
| **GPT-3 (175B)** | 175B Dense | ~2% – 5% | ~40% – 50% | **10x – 15x** |
| **GPT-4 (MoE)** | ~220B Active / Token | ~4% – 8% | ~45% – 55% | **6x – 10x** |
| **Claude 3.5 Sonnet** | GQA MoE / Dense | ~8% – 12% | ~55% – 65% | **5x – 8x** |

#### Why MFU Evolved Across Generations:
* **GPT-2 (1.5B):** On V100 GPUs, single-token generation was constrained by memory bandwidth. In low-batch interactive mode, MFU hovered around a dismal 1–3%.
* **GPT-3 (175B):** Spanning 8x A100 80GB GPUs via Tensor Parallelism, fetching 350GB of FP16 parameters across NVLink from VRAM to SRAM for just 1 token produced massive idle time. Moving from Batch=1 to Batch=128 boosted effective throughput by **over 10x**, elevating MFU from ~3% to nearly 50%.
* **GPT-4 (Mixture of Experts):** By routing each token to a subset of expert feed-forward networks (~220B active out of ~1.8T total parameters), GPT-4 reduced compute per token while requiring distributed VRAM. High-concurrency batching combined with Prefill/Decode disaggregation allows serving clusters to hit ~50% MFU.
* **Claude 3.5 Sonnet & Modern Frontier Models:** Utilizing **Grouped-Query Attention (GQA)**, **FlashAttention-3**, and **FP8 precision**, modern architectures shrink the KV cache VRAM footprint by 8x. This allows higher batch sizes per GPU, pushing interactive streaming MFU up to ~10%, while asynchronous batching achieves **55%–65% MFU** on NVIDIA H100 clusters.

---

## 5. Latency vs. Throughput: The Core Trade-Off

This brings us to the central fork in the road:

* **Latency (Speed):** *"How quickly does a single user receive each token?"* (Measured in milliseconds per token).
* **Throughput (Capacity):** *"How many total tokens can the entire cluster generate across all users per second?"* (Measured in tokens/sec/GPU).

```
                        THE SERVING SPECTRUM

       LOW LATENCY                              HIGH THROUGHPUT
     (Human-Centric)                           (Machine-Centric)
  ◄─────────────────────────────────────────────────────────────►
  • Batch Size: 1 to 4                      • Batch Size: 64 to 256+
  • Focus: User Experience (Chat, IDE)      • Focus: Cost & Efficiency (Eval, Batch)
  • Compute Utilization (MFU): ~2-8%        • Compute Utilization (MFU): ~45-60%
  • Cost per Token: High                    • Cost per Token: Low
```

---

## 6. Why Today's AI World Deliberately Sacrifices Throughput

If batching hundreds of requests together produces the cheapest, most efficient tokens, why isn't every AI deployment doing it?

**Because humans are in the loop.**

When you chat with an AI assistant or use an autocomplete extension in your code editor:

* If the system waits to batch your request with 100 other people, your **Time to First Token (TTFT)** spikes from 200 milliseconds to 4 seconds.
* If each step has to compute 128 parallel streams, your **Inter-Token Latency (ITL)** slows to a crawl—outputting text slower than human reading speed (~20–30 words per minute).

Nobody wants a conversational chatbot that pauses for five seconds between thoughts, even if that pause cuts the provider's cloud computing bill in half.

### The Modern Engineering Workarounds

To bridge this gap without bankrupting data centers, the infrastructure ecosystem builds clever hybrid techniques:

1. **Continuous / Iteration-Level Batching:** Instead of waiting for a static batch of 64 requests to finish together, new user requests enter and exit on every individual token step.
2. **Speculative Decoding:** A tiny, ultra-fast 1B draft model guesses the next 5 tokens rapidly (memory-light), and the giant 70B model checks all 5 guesses in a single, compute-dense forward pass.
3. **Prefill-Decode Disaggregation:** The compute-dense prompt intake (Prefill) is routed to one dedicated set of GPUs running at high MFU, while the memory-bound generation (Decode) is routed to low-latency streaming GPUs.

---

## 7. The Horizon Shift: Long-Horizon Agents & The Age of Maximum Throughput

While today's AI economy is dominated by human-facing chat interfaces, we are on the precipice of a fundamental paradigm shift: **Long-Horizon Autonomous Agents**.

Imagine an agent tasked with refactoring a massive legacy codebase, executing formal mathematical proofs, simulating drug discovery pipelines, or running complex multi-step research iterations. 

```
                                  HUMAN VS AGENT SERVING DEMAND

   HUMAN INTERACTIVE CHAT                             LONG-HORIZON AUTONOMOUS AGENTS
   (Synchronous / Latency-Bound)                       (Asynchronous / Throughput-Bound)
   ┌───────────────────────────┐                       ┌───────────────────────────┐
   │ • Real-time user waiting  │                       │ • User asleep / away      │
   │ • Single-user stream      │                       │ • Deep multi-step reasoning│
   │ • Low Batch (MFU: ~5-10%) │                       │ • High Batch (MFU: 60%+)  │
   │ • Cost: $$$ / million tok │                       │ • Cost: $ / million tok   │
   └───────────────────────────┘                       └───────────────────────────┘
```

When autonomous agents work on complex tasks, **real-time streaming latency becomes irrelevant**:

* **Humans in the Sleep Cycle:** While you sleep, an autonomous agent can spend 8 hours silently generating millions of tokens—running tests, fixing bugs, and verifying results. It does not matter whether an individual token streams in 15ms or 150ms.
* **Pure Batch Processing:** Data centers can stack agent reasoning workloads into massive asynchronous queues (Batch Size = 128, 256, or 512+).
* **Unlocking Peak MFU:** Operating in batch mode allows GPU clusters to hit **60%+ MFU continuously**. 
* **10x Cost Reduction:** Moving the needle on throughput efficiency means the exact same H100 cluster can process **5x to 10x more reasoning tokens per dollar**.

When we remove the human from the real-time interaction loop, we eliminate the low-MFU memory bottleneck. Compute hardware finally runs at its theoretical limits, transforming AI reasoning from an expensive luxury into an abundant, non-stop utility.

---

## The Takeaway

When you prompt an LLM today and watch words stream across your screen, you are witnessing an intentional economic trade-off: data centers sacrifice raw hardware efficiency—letting 90% of GPU compute capacity sit idle—just to satisfy human real-time perception.

But as AI shifts from interactive chatbots to autonomous long-horizon agents working non-stop in the background, latency constraints fall away. In an agent-driven world, GPUs will no longer sit idle waiting for index cards—they will run at full saturation, day and night.

---

## References & Further Reading

1. **The Roofline Model**: Williams, S., Waterman, A., & Patterson, D. (2009). *Roofline: an insightful visual performance model for multicore architectures*. Communications of the ACM, 52(4), 65-76. [ACM Digital Library](https://dl.acm.org/doi/10.1145/1498765.1498785)
2. **Model FLOPs Utilization (MFU) & Scaling Laws**: 
   - Kaplan, J., et al. (2020). *Scaling Laws for Neural Language Models*. OpenAI. [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
   - Chowdhery, A., et al. (2022). *PaLM: Scaling Language Modeling with Pathways*. Google Research. [arXiv:2204.02311](https://arxiv.org/abs/2204.02311)
3. **LLM Serving Architecture & Continuous Batching**: Yu, G. I., et al. (2022). *Orca: A Distributed Serving System for Transformer-Based Generative Models*. USENIX Symposium on Operating Systems Design and Implementation (OSDI '22). [USENIX Link](https://www.usenix.org/conference/osdi22/presentation/yu)
4. **Speculative Decoding**: Leviathan, Y., Kalman, M., & Matias, Y. (2023). *Fast Inference from Transformers via Speculative Decoding*. International Conference on Machine Learning (ICML 2023). [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
5. **Prefill-Decode Disaggregation**: Patel, P., et al. (2024). *Splitwise: Efficient Generative LLM Serving Using Prefill-Decode Disaggregation*. International Symposium on Computer Architecture (ISCA 2024). [arXiv:2311.18677](https://arxiv.org/abs/2311.18677)
6. **Grouped-Query Attention (GQA)**: Ainslie, J., et al. (2023). *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*. EMNLP 2023. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
7. **FlashAttention Optimizations**: Dao, T., et al. (2022, 2023, 2024). *FlashAttention-1, 2, and 3: Fast and Memory-Efficient Exact Attention*. NeurIPS / ICML. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
