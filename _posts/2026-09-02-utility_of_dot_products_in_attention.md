---
layout: post
title: "The Flashlight and the Shadow: How Dot Products and Normalization Power AI Attention"
date: 2026-09-02
desc: "From flashlight shadows to multi-dimensional Transformer attention: an intuitive, first-principles deep dive into dot products, matrix operations, and why self-attention works the way it does."
keywords: "AI, LLM, Dot Product, Attention Mechanism, Linear Algebra, PyTorch, Transformers, Deep Learning, Backpropagation"
categories: [Ml]
tags: [blog, AI, LLM, Math, Attention, Transformers, Machine Learning]
icon: fa-pencil
---

If you peel back the layers of a modern Large Language Model—past the billions of parameters, the multi-head mechanisms, and the massive GPU clusters—you arrive at a shockingly simple mathematical operation performed trillions of times a second:

$$\mathbf{u} \cdot \mathbf{v} = u_1 v_1 + u_2 v_2 + \dots + u_n v_n$$

That’s it. Just multiplying corresponding numbers together and adding them up.

Yet, this elementary arithmetic trick is the fundamental engine that allows LLMs to understand context, route information, and decide which words in a sentence relate to each other.

How does adding simple products together produce artificial intelligence? Why does it magically measure how "aligned" two concepts are in space? And when an LLM processes a word, why doesn't that word just pay 100% of its attention to itself?

Let's break it down step-by-step from first principles.

---

### Part 1: What Actually Is a Dot Product?

At its core, a dot product takes two vectors (think of them as arrows pointing in space) and outputs a single number. 

Geometrically, the dot product of two vectors $\mathbf{u}$ and $\mathbf{v}$ is defined as:

$$\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\| \|\mathbf{v}\| \cos\theta$$

Where:
* $\|\mathbf{u}\|$ is the length (magnitude) of vector $\mathbf{u}$.
* $\|\mathbf{v}\|$ is the length (magnitude) of vector $\mathbf{v}$.
* $\theta$ is the angle between the two arrows.

#### The Flashlight and Shadow Intuition

Imagine vector $\mathbf{u}$ is lying flat on the floor, and vector $\mathbf{v}$ is floating at an angle above it. 

Now, imagine shining a flashlight straight down from above $\mathbf{v}$ onto $\mathbf{u}$. Vector $\mathbf{v}$ casts a shadow on vector $\mathbf{u}$. 

```
          v (arrow floating up)
         /|
        / |
       /  |  Flashlight from above
      /   |
     /    v
    +----------> u (floor vector)
    |-------|
     Shadow length: ||v|| cos(θ)
```

* The length of that shadow is $\|\mathbf{v}\| \cos\theta$.
* The dot product simply multiplies the length of the floor vector ($\|\mathbf{u}\|$) by the length of the shadow cast upon it ($\|\mathbf{v}\| \cos\theta$).

$$\text{Dot Product} = \text{(Length of Floor Vector)} \times \text{(Length of Shadow Cast On It)}$$

* **Same Direction ($\theta = 0^\circ$)**: The shadow is at its maximum length ($\cos 0^\circ = 1$). The dot product is simply $\|\mathbf{u}\| \|\mathbf{v}\|$.
* **Perpendicular ($\theta = 90^\circ$)**: The flashlight is directly above the arrow, casting zero shadow ($\cos 90^\circ = 0$). The dot product is zero. The two vectors share no directional alignment.
* **Opposite Direction ($\theta = 180^\circ$)**: The shadow falls backwards ($\cos 180^\circ = -1$), producing a negative dot product.

#### The Role of Normalization (Cosine Similarity)

Notice that the raw dot product depends on two factors: **direction** (the angle $\theta$) and **magnitude** (how long the arrows are). 

If an arrow is exceptionally long, its dot product will be huge even if it points in a slightly different direction. In deep learning, we often want to isolate purely *where the arrow is pointing*, regardless of its length.

We achieve this by **normalizing** vectors to unit length ($\|\mathbf{u}\| = 1$ and $\|\mathbf{v}\| = 1$). When vectors are normalized, lengths drop out of the equation entirely:

$$\text{Normalized Dot Product} = (1) \times (1) \times \cos\theta = \cos\theta$$

This is **Cosine Similarity**. A value of $+1$ means complete alignment, $0$ means complete independence (orthogonality), and $-1$ means exact opposition.

---

### Part 2: The Magic Trick — How Code Computes Angles Without Trigonometry

> **A Quick Clarification**: In Part 1, we learned that **direction (the angle $\theta$)** is precisely what we want to measure. But when you look at PyTorch code, you never see `math.cos(theta)` or degree calculations. Why don't we need trigonometric functions in software to compute geometric angles?

When you write `matrix_a.dot(matrix_b)` or `torch.matmul(A, B)` in Python, you don't calculate any angles explicitly. You don't call `math.acos()`, nor do you draw triangles. 

Instead, the computer calculates simple component arithmetic:

$$\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{n} u_i v_i = u_1 v_1 + u_2 v_2 + \dots + u_n v_n$$

Why does multiplying coordinates along perpendicular axes and adding them up **automatically** compute $\|\mathbf{u}\| \|\mathbf{v}\| \cos\theta$?


Think of any 2D vector as a step along the $X$-axis plus a step along the $Y$-axis:
$$\mathbf{u} = u_x \hat{i} + u_y \hat{j}$$
$$\mathbf{v} = v_x \hat{i} + v_y \hat{j}$$

Where $\hat{i}$ and $\hat{j}$ are unit arrows along perpendicular coordinate axes. 

Expanding $(\mathbf{u} \cdot \mathbf{v})$ algebraically:

$$\mathbf{u} \cdot \mathbf{v} = (u_x \hat{i} + u_y \hat{j}) \cdot (v_x \hat{i} + v_y \hat{j})$$
$$= u_x v_x (\hat{i} \cdot \hat{i}) + u_x v_y (\hat{i} \cdot \hat{j}) + u_y v_x (\hat{j} \cdot \hat{i}) + u_y v_y (\hat{j} \cdot \hat{j})$$

Because our coordinate axes are perpendicular to each other:
* $\hat{i} \cdot \hat{j} = 0$ (perpendicular axes cast zero shadow on each other!)
* $\hat{i} \cdot \hat{i} = 1$ (an axis cast on itself is 100% aligned!)

The cross-terms vanish completely, leaving:

$$\mathbf{u} \cdot \mathbf{v} = u_x v_x + u_y v_y$$

> 💡 **Key Takeaway**: By representing vectors in a system of perpendicular axes, hardware computes geometric angles and shadows automatically using fast, simple component-wise arithmetic.

---

### Part 3: How Dot Products Power Attention Mechanisms

In Large Language Models, words aren't just strings of text. They are represented as high-dimensional vectors (e.g., 4,096 dimensions). Each dimension can be thought of as a feature axis representing semantic qualities—like tense, sentiment, subject-verb relations, or context.

When two token vectors have a high dot product, it means their feature numbers match along the same dimensions. They are "pointing" in a similar semantic direction.

#### The Spotlight Search: "Your Journey Starts Here"

Let's walk through how this works in a Transformer's Attention mechanism.

Consider the sequence: **"Your Journey Starts Here"**

Every word in this sentence gets transformed into three distinct roles:
1. **Query ($Q$)**: *"What am I looking for in this sentence?"*
2. **Key ($K$)**: *"What information do I contain?"*
3. **Value ($V$)**: *"What actual content do I pass along if selected?"*

Let's focus on the word **"Journey"** acting as our **Query ($Q_{\text{Journey}}$)**.

```
Sequence:  [ "Your",  "Journey",  "Starts",  "Here" ]
               |          |           |         |
Keys (K):    K_Your   K_Journey   K_Starts   K_Here
                          ^
                          | (Query: Q_Journey)
```

The word **"Journey"** is a noun describing a process or path. Its Query vector $Q_{\text{Journey}}$ asks: *"Where is the action or verb that tells the reader what this journey is doing?"*

To answer this, the Transformer computes the dot product of $Q_{\text{Journey}}$ against the **Key** vectors of every token in the sentence:

1. **$Q_{\text{Journey}} \cdot K_{\text{Your}}$**: Possessive modifier. Mild alignment ($1.2$).
2. **$Q_{\text{Journey}} \cdot K_{\text{Journey}}$**: Noun meeting itself. Moderate alignment ($2.5$).
3. **$Q_{\text{Journey}} \cdot K_{\text{Starts}}$**: Action verb describing movement! **High directional alignment in feature space ($4.8$)!**
4. **$Q_{\text{Journey}} \cdot K_{\text{Here}}$**: Spatial adverb. Low alignment ($0.3$).

$$\text{Raw Dot Products (Logits)} = [ 1.2, \; 2.5, \; 4.8, \; 0.3 ]$$

Notice how **$K_{\text{Starts}}$** scored the highest dot product ($4.8$) because its key vector pointed strongly in the direction requested by $Q_{\text{Journey}}$.

Next, we scale and apply **Softmax** to convert these raw dot products into percentage probabilities:

$$\text{Attention Weights} = \text{Softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) = [ 5\%, \; 15\%, \; 75\%, \; 5\% ]$$

The model pays **75% of its attention** to **"Starts"**! 

Using this weighted percentage, it blends the Value vectors ($V$) of all tokens together, creating an enriched contextual embedding for "Journey" that explicitly encodes *that the journey is starting*.

#### Why Scaled Dot Products Instead of Pure Unit Normalization?

You might notice a subtle distinction here: *In Part 1, we said normalizing vectors to unit length isolates pure direction ($\cos\theta$). Why does standard Transformer attention use matrix multiplication on unnormalized vectors ($Q K^T$) rather than pure unit-normalized cosine similarity?*

This is a deliberate design trade-off in deep learning:

1. **Magnitude Represents Confidence**: In a neural network, the length of a Query or Key vector isn't useless noise—it represents **feature magnitude or confidence**. A token with a larger vector norm can signal *"I am an essential keyword in this sequence!"* Pure unit normalization would erase this confidence signal.
2. **The High-Dimension Variance Problem**: However, in high dimensions (e.g., $d_k = 128$), unnormalized dot products grow large in variance ($O(d_k)$). Extremely large dot products push Softmax into regions with near-zero gradients (saturation).
3. **The Compromise ($\frac{1}{\sqrt{d_k}}$)**: By dividing $Q K^T$ by $\sqrt{d_k}$, Transformers get the best of both worlds: they control magnitude explosion so Softmax doesn't saturate, while still allowing $Q$ and $K$ to use relative magnitude as a signal of feature importance!

---


### Part 4: The Self-Attention Mystery: Why Doesn't a Word Just Attend 100% to Itself?

Looking at the numbers above, a natural question arises:

> *"In our example, $Q_{\text{Journey}} \cdot K_{\text{Journey}}$ only got 15% attention, while $Q_{\text{Journey}} \cdot K_{\text{Starts}}$ got 75%. Wouldn't a vector dotted with itself have an angle of $\theta = 0^\circ$ and $\cos(0^\circ) = 1$? Why doesn't 'Journey' pay almost 100% of its attention to 'Journey'?"*

This is one of the most common misconceptions in deep learning. There are **three fundamental reasons** why tokens don't just attend to themselves:

#### 1. Queries and Keys are Different Vector Spaces ($Q \neq K$)

A token $X$ does **not** perform a dot product with itself ($X \cdot X$). 

Instead, input token $X$ is transformed by two distinct learned projection matrices:
$$Q = X \cdot W_Q \quad \text{and} \quad K = X \cdot W_K$$

* $W_Q$ projects the word into a **Search Space** ("What am I looking for?").
* $W_K$ projects the word into a **Target Space** ("What do I offer to others?").

Because $W_Q$ and $W_K$ are completely different matrices learned during training, **$Q_{\text{Journey}}$ and $K_{\text{Journey}}$ are entirely different vectors!** $Q_{\text{Journey}} \cdot K_{\text{Journey}}$ is a Query dotted with a Key, not a vector dotted with itself. There is no guarantee that $\theta = 0^\circ$.

#### 2. The Goal of Language is Context, Not Identity

If $W_Q$ and $W_K$ were trained such that $Q_{\text{word}} \cdot K_{\text{word}}$ was always the highest value, every token in the network would only listen to itself. 

The phrase **"bank of the river"** and **"bank for a deposit"** would process the word "bank" identically. The model would fail to capture any contextual meaning. The loss function explicitly trains $W_Q$ and $W_K$ to reach out and pull in verbs, adjectives, and subjects from neighboring positions.

#### 3. Softmax Amplifies the Top Candidate

Even if $Q_{\text{Journey}} \cdot K_{\text{Journey}}$ produces a positive dot product ($2.5$), Softmax converts scores into a competitive probability distribution summing to $1.0$.

Because Softmax uses exponentiation ($\exp(x)$), a score of $4.8$ vs $2.5$ creates a massive gap: $\exp(4.8) \approx 121.5$ while $\exp(2.5) \approx 12.2$. The stronger match for "Starts" exponentially crushes "Journey's" self-score.

---

### Part 5: From Static Arithmetic to AI: The Need for Random Weights, Backpropagation, and Derivatives

This leads to the ultimate question: *If attention is just dot products between Queries and Keys, how does the model learn what to search for in the first place?*

Why can't we just take raw word embeddings $X$ and compute $X \cdot X^T$? Why do we need projection matrices ($W_Q, W_K, W_V$), and why must they undergo training?

```
 Raw Input Vectors (X)
         │
         ▼  (Multiply by Random Weights)
 Projection Spaces: Q = X·W_Q,  K = X·W_K
         │
         ▼  (Compute Dot Products & Softmax)
 Attention Scores: Softmax(Q·Kᵀ / √d)
         │
         ▼  (Forward Pass Output vs Ground Truth)
 Calculate Error (Loss)
         │
         ▼  (Backpropagation Chain Rule)
 Derivatives: ∂Loss / ∂W_Q,  ∂Loss / ∂W_K
         │
         ▼  (Gradient Descent Update)
 Updated Weights: W ← W - η (∂Loss / ∂W)
```

#### 1. Raw Word Vectors Are Static and Blind

If you compute dot products directly on input embeddings ($X \cdot X^T$), the similarity scores are **fixed forever**. The word "Journey" would have the exact same dot product with "Starts" regardless of sentence structure or context. 

Raw word vectors cannot adapt. They don't know *what question* is being asked in a specific sentence, nor do they know *which semantic features* matter for the task at hand.

#### 2. Random Weights Break Symmetry

To allow the model to learn, we introduce linear transformation matrices ($W_Q, W_K, W_V$). At the start of training (Iteration 0), these matrices are filled with **random numbers**.

Because they are random, the initial dot products are completely meaningless:
* "Journey" might pay 80% of its attention to a random period or comma.
* The model’s predictions are garbage, resulting in a high **Loss** (error).

However, these random weights serve a vital purpose: **they break symmetry**, providing the network with adjustable high-dimensional projection dials to begin exploring feature space.

#### 3. Derivatives Are the Tuning Dials of Attention

How do random weights transform into precision-crafted linguistic search filters? Through **Backpropagation** and **Derivatives**.

When the model makes a prediction error, we compute the gradient of the Loss with respect to every weight parameter:

$$\frac{\partial \text{Loss}}{\partial W_Q}, \quad \frac{\partial \text{Loss}}{\partial W_K}, \quad \frac{\partial \text{Loss}}{\partial W_V}$$

What does a derivative like $\frac{\partial \text{Loss}}{\partial W_Q}$ actually tell us in plain English?

It is a precise mathematical feedback signal that says:
> *"If you tilt the Query matrix $W_Q$ by a tiny fraction of a degree in direction $\Delta$, the Query vector $Q_{\text{Journey}}$ will rotate slightly closer to $K_{\text{Starts}}$. That tiny rotation will boost their dot product, assign more attention weight to the correct word, and lower the model's total error!"*

#### 4. Gradient Descent Sculpting the Space

Using Gradient Descent, we update the weights at every training step:

$$W_Q \leftarrow W_Q - \eta \frac{\partial \text{Loss}}{\partial W_Q}$$

Over millions of training steps across trillions of tokens:
1. Derivatives continuously tilt and rotate $W_Q$ and $W_K$.
2. High-dimensional vector space bends, aligns, and organizes.
3. Random matrices transform into sharp, specialized feature detectors.

Without backpropagation pushing weights along the path of steepest derivative descent, dot products would just be blind arithmetic on random vectors. **Derivatives are the force that sculpts random dot products into meaningful semantic intelligence.**

---

### Summary: The Intuitive Hierarchy

To bring it all together:

1. **Dot Product ($\mathbf{u} \cdot \mathbf{v}$)**: Measures how much two vectors shadow each other. It answers *"How much do these two arrows point in the same direction?"*
2. **Normalization (Cosine Similarity)**: Strips away vector length so we measure pure directional alignment without being fooled by big numbers.
3. **Element-wise Multiplication ($\sum u_i v_i$)**: A computational trick where perpendicular coordinate axes allow us to compute geometric angles using fast, simple arithmetic.
4. **Attention Mechanisms ($Q K^T$)**: Uses dot products as similarity engines to let Query vectors scan Key vectors across a sequence.
5. **Self-Attention Projections ($W_Q, W_K$)**: Separates "what a word is looking for" from "what a word offers," allowing tokens to attend to contextual neighbors rather than just repeating themselves.
6. **Backpropagation & Derivatives ($\frac{\partial \text{Loss}}{\partial W}$)**: The feedback loop that continuously rotates and tunes random weight matrices ($W_Q, W_K$) so that dot products align precisely with the right contextual words.

The next time you see an LLM write a coherent essay, remember: under the hood, it’s just billions of small flashlights casting shadows on high-dimensional vectors, guided by derivatives to find where the arrows should align.

---

### References & Further Reading

1. **Vaswani et al. (2017)** — *"Attention Is All You Need"* ([arXiv:1706.03762](https://arxiv.org/abs/1706.03762)). The landmark paper introducing Scaled Dot-Product Attention, Transformer architecture, and Query/Key/Value projections.
2. **Jay Alammar** — *"The Illustrated Transformer"* ([jalammar.github.io/illustrated-transformer](https://jalammar.github.io/illustrated-transformer/)). A visual guide to understanding matrix multiplications and attention mechanisms in Transformers.
3. **3Blue1Brown (Grant Sanderson)** — *"Essence of Linear Algebra"* ([youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)). Exceptional geometric visualizations of dot products, basis transformations, and vector projections.
4. **Strang, Gilbert** — *"Linear Algebra and Learning from Data"* (Wellesley-Cambridge Press). Rigorous foundational mathematical treatment of matrix decompositions and deep learning optimization.
