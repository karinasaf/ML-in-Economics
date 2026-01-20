---
title: "Machine Learning in Economics (D200)"
subtitle: "Syllabus (Lent 2026)"
author: "Dr. Stefan Bucher"
format: 
  pdf: 
    cite-method: natbib
  html: default
bibliography: references.bib
---

**University of Cambridge** **Faculty of Economics**

**Course Code and Title:** Machine Learning in Economics (D200)\
**Term:** Lent Term 2026\
**Lecturer:** Dr. Stefan Bucher \
**Office Hours:** Wed 3.00pm-4.00pm. Sign up [here](https://calendly.com/stefabu/office-hours)

**Lectures:** Sat 11.00am-1.00pm in Meade Room, weeks 1-9  \
**Classes:** (some) Fri 3.00-5.00pm/5.00-7.00pm in Room 7 (Lecture Block), weeks 3, 5, 7, and 9.  \
**Teaching Assistant:** Vahan Geghamyan 

**Course Website:** <https://github.com/MLecon/ML-in-Economics> \
**Assignment Submission:** [Github Classroom](https://classroom.github.com/classrooms/195107486-machine-learning-in-economics-2026)\
**Readings:** [Zotero Group Library](https://www.zotero.org/groups/ml_econ)\
**Recordings of further interest:** [Youtube Playlist](https://youtube.com/playlist?list=PLo8op7DIq2yhVzg8sUVAc36PRDZVOw6GD&si=ZV1YOyqJCbzlFdF9)

# Course Overview

## Course Description

Machine Learning is in the process of transforming economics as well as the business world. This course aims to provide a graduate-level introduction to machine learning equipping students with a solid and rigorous foundation necessary to understand the key techniques of this fast-evolving field and apply them to economic problems. The curriculum bridges theoretical foundations with practical implementations and strives to remain relevant by teaching a conceptual understanding that transcends the implementation details of current state-of-the art methods.

## Specific Topics Covered

The course covers key methods in

-   supervised learning, including regression, classification, and neural networks
-   unsupervised learning, including clustering and dimensionality reduction
-   reinforcement learning, including bandit problems
-   applications to economics.

## Course Aims and Objectives

By the end of this course, students will be equipped with:

-   a foundational understanding of the most relevant ML tools and how they are reshaping economic analysis
-   the ability to work with ML models using popular software environments such as [PyTorch](https://pytorch.org) and [scikit-learn](https://scikit-learn.org), and to adapt them for economic problems
-   critical skills in interpreting and explaining sophisticated ML models in economic contexts

## Lecture Materials

Lecture materials will be posted to the course website. 
The course loosely follows the textbook of @prince2023 which is freely available at <https://udlbook.github.io/udlbook/>. The material may be complemented by chapters from further classic textbooks, including @bishop2006, @hastie2009, @goodfellow2016, @mackay2003, @murphy2022, and @sutton2018. These are not required reading.

All readings are also organized in a [Zotero Group Library](https://www.zotero.org/groups/ml_econ). A [Youtube Playlist](https://youtube.com/playlist?list=PLo8op7DIq2yhVzg8sUVAc36PRDZVOw6GD&si=ZV1YOyqJCbzlFdF9) curates videos of further interest to the course's topics.

## Computing Environment

The lectures and classes feature examples in [Jupyter](https://jupyter.org/) Notebooks for use on [Google Colab](https://colab.google/).

### Package Management

This course uses [`uv`](https://docs.astral.sh/uv/) for Python package management. To get started:

1. Install `uv` (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

## Prerequisites

Linear Algebra, calculus, probability theory and statistics, as well as programming skills (in Python) are required.



# Contents and Schedule

### Introduction and Foundations - Week 1 (24 January 2026)

-   A brief overview of AI, ML, and Deep Learning [@prince2023, Chapter 1]
-   Probability and information theory fundamentals [@prince2023, Appendix C]


## Part 1: Supervised Machine Learning

### Prediction and Linear Regression - Week 2 (31 January 2026)

-   Linear Regression: Minimizing mean-squared error using matrix notation [@prince2023, Chapter 2]
-   Optimization: Gradient descent, stochastic gradient descent, Adam optimizer [@prince2023, Chapter 6]
-   Model Evaluation: Bias-variance tradeoff and overfitting, training/test set and cross-validation, double descent [@prince2023, Chapter 8]
-   sklearn
-   PyTorch
-   Introduction to Research Project  


### Classification and Logistic Regression - Week 3 (7 February 2026)

-   Multinomial Logit and Discrete Choice 
-   Loss functions [@prince2023, Chapter 5]
-   Regularization: Explicit and implicit regularization, dropout, ensemble methods, transfer learning [@prince2023, Chapter 9]

::: {.content-hidden}
#### Topics Not Currently Covered

-   Support Vector Machines (SVM)
-   Decision/classification trees
-   Ensemble Methods: Boosting and bagging, random forests, gradient boosting machines
-   Kernel ridge regression
-   Gaussian Process Models
:::


### Artificial Neural Networks and Deep Learning - Week 4 (14 February 2026)

-   Nonlinear models (e.g., GLM)
-   Shallow neural networks  [@prince2023, Chapter 3]
-   Deep feedforward neural networks (multi-layer perceptrons) [@prince2023, Chapter 4]
-   Backpropagation [@prince2023, Chapter 7]


### Representation Learning and Natural Language Processing (NLP) - Week 5 (21 February 2026)

-   Convolutional neural networks (CNN) [@prince2023, Chapter 10]
-   Transformers: Self-attention mechanism, encoder-decoder architecture [@prince2023, Chapter 12.1-12.6]
-   Large Language Models
-   Post-training: Finetuning and adaptation (e.g., QLoRA)

::: {.content-hidden}
#### Topics Not Currently Covered

-   Sequence and time series modeling: Recurrent neural networks (RNN), Hopfield network, LSTM
-   Word embedding (e.g., Word2Vec)
-   Graph Neural Networks (GNN)
:::


## Part 2: Unsupervised Machine Learning

### Generative AI - Week 6 (28 February 2026)

-   Generative Pre-trained Transformers (GPT) [@prince2023, Chapter 12.7-12.10]
-   Unsupervised Learning [@prince2023, Chapter 14]
-   Variational Autoencoders (VAE) [@prince2023, Chapter 17]
-   Diffusion Models [@prince2023, Chapter 18]

::: {.content-hidden}
#### Topics Not Currently Covered

- Unsupervised Learning: 
  - Gaussian Mixture Models, Expectation Maximization (vs gradient descent)
  - Clustering (K-means)
  - Dimensionality reduction (PCA, ICA)
-   Generative Adversarial Networks (GANs) [@prince2023, Chapter 15]
:::


## Part 3: Reinforcement Learning

### Reinforcement Learning - Week 7 (7 March 2026)

-   Reinforcement Learning: Markov Decision Processes (MDP), policies, value functions [@prince2023, Chapter 19]
-   Bellman equations
-   Q-Learning and Deep Q-Networks
-   Proximal Policy Optimization (PPO)

::: {.content-hidden}
#### Topics Not Currently Covered

-   Hidden Markov Models (HMM)
-   Multi-armed bandit testing
-   Bandit Gradient Algorithm as Stochastic Gradient Descent
-   @silver2016
-   @schrittwieser2020
-   Inverse Reinforcement Learning
:::


## Part 4: ML and Economics

### Synthesis: Information-Theoretic Lens - Week 8 (14 March 2026)

-   Review and synthesis: The information-theoretic lens as a unifying principle [@alemi2024]
-   Unified view of supervised learning, unsupervised learning, and representation learning through KL divergence minimization
-   Reinforcement Learning from Human Feedback (RLHF): Alignment of Large Language Models

### ML and Economics - Week 9 (21 March 2026)

-   Prediction vs. estimation/inference [@athey2019]
-   Applications of ML in economics

::: {.content-hidden}
#### Topics Not Currently Covered

-   Brief remarks on causal inference (separate module)
-   Matrix completion problem: Consumer choice modeling and application to recommender systems
:::



# Classes and Problem Sets 

Classes are meant to discuss problem sets and questions arising from the lectures as well as (towards the end of the term) the research projects. 
Problem sets are to be submitted in groups of 4 students (of varying configuration) on Github Classroom.


# Assessment

Assessment in the course is based entirely on the completion of a small-scale research project, which is assessed via 
a written project report of 3 single-spaced pages (approximately 1500 words) due on 16th March 2026, and 
an oral examination which constitutes a brief (5-7 min.) presentation (slides of presentation to be submitted on same day) and Question and Answer session held on either March 23 or 24.  
All elements are essential and constitute 100% weighting.
<!--(15 minutes per student: 5-7min presentation followed by questions)-->
<!--Schedule 30min per student: 15min for my preparation based on the report, 15min for the oral exam incl. presentation. Total effort: 20hrs over 2-3 days. In 2025 I aimed for 20min per report...-->


# Key Dates
24 Jan 2026
: First Lecture

5 Feb 2026 (12 noon)
: Problem Set 1 due <!-- distributed 23 January 2026-->

19 Feb 2026 (12 noon)
: Problem Set 2 due <!-- distributed 6 February 2026-->

5 March 2026 (12 noon)
: Problem Set 3 due <!-- distributed 20 February 2026-->

16 Mar 2026
: Project report due 

23-25 Mar 2026
: Oral examination

<!--
31 Jan 2026 (lecture): Project announced
20 March 2026 (class): final review
-->



# Policies

Attendance
: Regular attendance at lectures and classes is mandatory.

Plagiarism
: All work submitted must be original. Plagiarism will result in serious academic penalties in line with University policy.

Use of Large Language Models
: Submitted work must not be direct output from Large Language Models such as ChatGPT, and may be checked accordingly.

Late Submissions
: Assignments submitted after the deadline will be penalized (unless an extension is granted in advance) by 10% per 24 hours (additively, i.e. a submission received 49 hours after the deadline will receive 70% of full marks). 

## **Support**

Students are encouraged to use office hours to discuss any academic or personal issues related to the course. Additional support services are available through the university's counseling and academic support centers.

## **Feedback**

Feedback of any kind is most welcome. To suggest improvements (e.g. typos) on the teaching material, please open a Github issue.

# Resources and Reading Materials