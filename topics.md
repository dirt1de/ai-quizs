# Comprehensive LLM Study Guide

## 1. Core Foundations

### Neural Network Fundamentals
- **Feed-forward networks**: Structure, activation functions, backpropagation
- **Gradient descent**: Optimization algorithms (SGD, Adam, AdamW)
- **Regularization techniques**: Dropout, batch normalization, weight decay
- **Loss functions**: Cross-entropy, perplexity for language models
- **Computational graphs**: Automatic differentiation, frameworks (PyTorch, TensorFlow)

### Natural Language Processing Basics
- **Word representations**: One-hot encoding, distributed representations
- **Word embeddings**: Word2Vec, GloVe, FastText
- **Language statistics**: N-grams, co-occurrence matrices
- **Sequential data processing**: RNNs, LSTMs, GRUs
- **Classical NLP approaches**: Part-of-speech tagging, named entity recognition, syntactic parsing

### Transformer Architecture
- **Attention mechanisms**: Scaled dot-product attention
- **Multi-head attention**: Parallel attention heads and their benefits
- **Positional encodings**: Absolute vs. relative position information
- **Layer normalization**: Pre-norm vs. post-norm architectures
- **Residual connections**: Information flow in deep transformers
- **Original transformer paper**: "Attention Is All You Need" (Vaswani et al., 2017)
- **Variants**: Encoder-only (BERT), decoder-only (GPT), encoder-decoder (T5)

### Transfer Learning
- **Pretraining and fine-tuning paradigm**: Two-stage learning process
- **Domain adaptation**: Cross-domain transfer challenges
- **Zero-shot, one-shot, and few-shot learning**: Learning with limited examples
- **Task-specific adaptations**: Adding classifier heads

## 2. LLM-Specific Topics

### Tokenization Methods
- **Subword tokenization**: BPE (Byte-Pair Encoding), WordPiece, SentencePiece
- **Tokenizer compression**: Byte-level BPE, regex-based tokenizers
- **Tokenization challenges**: Multilingual support, code representation
- **Vocabulary size tradeoffs**: Memory usage vs. semantic granularity

### Pretraining Objectives
- **Masked language modeling (MLM)**: BERT-style bidirectional training
- **Causal language modeling (CLM)**: GPT-style autoregressive training
- **Span prediction**: T5-style corrupted spans
- **Contrastive objectives**: SimCSE, CLIP (for multimodal models)
- **Hybrid approaches**: UL2, PaLM pretraining methodology

### Scaling Laws
- **Emergent abilities**: Capabilities that appear only at scale
- **Chinchilla scaling laws**: Compute-optimal training
- **Parameter scaling**: Effects on performance across tasks
- **Data scaling**: Dataset size requirements
- **Training compute budget allocation**: Model size vs. training tokens tradeoff

### Context Windows and Attention
- **Attention complexity**: Quadratic scaling challenge
- **Extended context methods**: 
  - Sparse attention patterns (Longformer, BigBird)
  - Recurrent memory mechanisms (Transformer-XL, Memorizing Transformers)
  - KV-caching for efficient inference
  - Position interpolation for extending trained models
- **Attention visualizations and interpretability**

## 3. Advanced Topics

### Alignment Techniques
- **Reinforcement Learning from Human Feedback (RLHF)**:
  - Reward modeling from human preferences
  - Proximal Policy Optimization (PPO) for LLM training
  - RLHF training pipeline: SFT → Reward Model → PPO
- **Constitutional AI (CAI)**: Self-critique and self-improvement
- **Direct Preference Optimization (DPO)**: Reward-free alignment
- **Rejection sampling approaches**: Best-of-n sampling from SFT models
- **Red-teaming**: Adversarial testing to find failure modes

### Prompt Engineering and In-Context Learning
- **Zero-shot prompting**: Direct instruction techniques
- **Few-shot prompting**: In-context examples and demonstrations
- **Chain-of-thought**: Step-by-step reasoning elicitation
- **Retrieval-augmented generation (RAG)**: Combining external knowledge
- **System prompts**: Setting constraints and personas
- **Structured outputs**: Controlling format with prompt design

### Model Evaluation
- **Standard benchmarks**: 
  - MMLU (academic and professional knowledge)
  - HELM (holistic evaluation)
  - Big-Bench Hard (challenging reasoning tasks)
  - GSM8K, MATH (mathematical reasoning)
- **Evaluation harnesses**: LM-Eval, EleutherAI evaluations
- **Human evaluation**: Methods, bias, and best practices
- **Evals-driven development**: Using evaluations to guide research

### Interpretability and Mechanistic Understanding
- **Attribution methods**: Integrated gradients, attention visualization
- **Circuit discovery**: Finding functional subnetworks in models
- **Causal intervention**: Ablation studies, patching, activation engineering
- **Neuron interpretation**: Feature visualization, dataset examples
- **Mechanistic interpretability**: Understanding computation in models
- **Model editing**: Localized changes to model behavior

### Multimodal LLMs
- **Vision-language models**: CLIP, Flamingo, GPT-4V architecture approaches
- **Cross-modal attention**: Connecting different modality embeddings
- **Multimodal tokenization**: Representing images in the vocabulary
- **Training strategies**: Separate vs. joint pretraining
- **Evaluation**: Multimodal benchmarks and capabilities

## 4. Practical Implementation

### Parameter-Efficient Fine-Tuning
- **Low-Rank Adaptation (LoRA)**: Mathematics and implementation
- **Adapters**: Bottleneck adapters, parallel adapters
- **Prompt tuning**: Soft prompts, prefix tuning, P-tuning
- **QLoRA**: Combining quantization with LoRA for memory efficiency
- **Layer freezing strategies**: Which layers to tune and why

### Inference Optimization
- **Quantization approaches**:
  - Post-training quantization (PTQ)
  - Quantization-aware training (QAT) 
  - Mixed precision (INT8, INT4, NF4)
- **Knowledge distillation**: Student-teacher approaches
- **Pruning**: Structured vs. unstructured sparsity
- **Speculative decoding**: Draft models for acceleration
- **Caching strategies**: KV-cache management

### Deployment Considerations
- **Inference hardware**: GPUs, TPUs, dedicated accelerators
- **Memory optimization**: Activation checkpointing, offloading
- **Batching strategies**: Dynamic batching, continuous batching
- **Model serving architecture**: Load balancing, replication
- **Latency-throughput tradeoffs**: Optimizing for different use cases
- **Distributed inference**: Tensor, pipeline, and model parallelism

### Building Applications with LLM APIs
- **Prompt design patterns**: Best practices for production
- **Function calling**: Structured API interactions
- **Tool use**: Integrating LLMs with external tools and APIs
- **Orchestration frameworks**: LangChain, semantic kernels
- **Evaluation in application context**: Task-specific metrics
- **Hybrid systems**: LLMs + retrieval + rule-based components

## 5. Ethical and Safety Considerations

### Bias, Fairness and Representation
- **Data bias sources**: Training corpus biases and amplification
- **Evaluation frameworks**: Bias benchmarks (BOLD, BBQ, WinoBias)
- **Mitigation strategies**: Data curation, fine-tuning, filtering
- **Representation issues**: Language coverage, cultural context
- **Documentation practices**: Model cards, dataset cards, system cards

### Security Vulnerabilities
- **Prompt injection**: Types and prevention strategies
- **Jailbreaking techniques**: Evolution and countermeasures
- **Data poisoning**: Training set attacks
- **Adversarial examples**: Robustness to perturbed inputs
- **Backdoors and trojan attacks**: Detecting and preventing

### Alignment Challenges
- **Alignment tax**: Performance tradeoffs for safety
- **Scalable oversight**: Evaluating capabilities beyond human verifiability
- **Specification gaming**: Reward hacking in language models
- **Deceptive alignment**: Models optimizing for appearing aligned
- **AI safety research landscape**: Technical research directions

### Societal and Labor Market Impacts
- **Automation impacts**: Task displacement vs. job displacement
- **Complementary AI**: Augmentation over replacement
- **Economic effects**: Productivity growth, inequality considerations
- **Content generation markets**: Impacts on creative industries
- **Information ecosystem effects**: Synthetic content, misinformation
