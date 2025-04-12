import { Question } from '../types/quiz';

// Function to randomize answer positions while maintaining the correct answer
const randomizeAnswers = (question: Question): Question => {
  const { options, correctAnswer } = question;
  const correctOption = options[correctAnswer];
  
  // Create a new array of indices excluding the correct answer
  const otherIndices = options
    .map((_, index) => index)
    .filter(index => index !== correctAnswer);
  
  // Shuffle the other indices
  for (let i = otherIndices.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [otherIndices[i], otherIndices[j]] = [otherIndices[j], otherIndices[i]];
  }
  
  // Create new options array with correct answer in random position
  const newOptions = new Array(options.length);
  const newCorrectAnswer = Math.floor(Math.random() * options.length);
  
  // Place correct answer in random position
  newOptions[newCorrectAnswer] = correctOption;
  
  // Fill remaining positions with other options
  let otherIndex = 0;
  for (let i = 0; i < newOptions.length; i++) {
    if (i !== newCorrectAnswer) {
      newOptions[i] = options[otherIndices[otherIndex]];
      otherIndex++;
    }
  }
  
  return {
    ...question,
    options: newOptions,
    correctAnswer: newCorrectAnswer
  };
};

// Apply randomization to each question
export const questions: Question[] = [
  {
    id: 1,
    question: "What is the key difference between batch normalization and layer normalization?",
    options: [
      "Batch normalization normalizes across the batch dimension while layer normalization normalizes across feature dimensions",
      "Batch normalization uses exponential moving averages for inference while layer normalization uses fixed statistics, but both normalize across feature dimensions",
      "Batch normalization requires synchronized batch statistics across devices while layer normalization can be computed independently, but both use running statistics",
      "Batch normalization normalizes across feature dimensions while layer normalization normalizes across the batch dimension, with both using fixed statistics"
    ],
    correctAnswer: 0,
    explanation: "Batch normalization normalizes activations across the batch dimension (N), making it sensitive to batch size, while layer normalization normalizes across feature dimensions (C, H, W), making it more stable across different batch sizes and suitable for recurrent networks.",
    reference: "https://arxiv.org/abs/1607.06450"
  },
  {
    id: 2,
    question: "What is the primary advantage of using residual connections in deep neural networks?",
    options: [
      "They enable direct gradient flow through identity mappings, mitigating the vanishing gradient problem",
      "They reduce memory usage by reusing feature maps while maintaining gradient flow through skip connections",
      "They increase model capacity by adding parallel computation paths while preserving the original signal",
      "They improve feature extraction by combining multiple receptive fields through additive transformations"
    ],
    correctAnswer: 0,
    explanation: "Residual connections create direct paths for gradient flow through identity mappings, allowing networks to learn residual functions and effectively train much deeper architectures by mitigating the vanishing gradient problem.",
    reference: "https://arxiv.org/abs/1512.03385"
  },
  {
    id: 3,
    question: "What is the key innovation in the Swish activation function compared to ReLU?",
    options: [
      "It provides a smooth, non-monotonic function that can better handle negative inputs while maintaining sparsity",
      "It uses exponential operations to increase the effective learning rate while maintaining the benefits of ReLU's linearity",
      "It combines multiple activation functions to create a more complex decision boundary while preserving gradient flow",
      "It automatically adapts its shape based on the input distribution while maintaining the computational efficiency of ReLU"
    ],
    correctAnswer: 0,
    explanation: "Swish (f(x) = x * sigmoid(βx)) is smooth and non-monotonic, allowing it to handle negative inputs better than ReLU while maintaining the benefits of sparsity and gradient flow.",
    reference: "https://arxiv.org/abs/1710.05941"
  },
  {
    id: 4,
    question: "What is the primary purpose of attention mechanisms in neural networks?",
    options: [
      "To dynamically weight the importance of different parts of the input based on context",
      "To reduce computational complexity by focusing on relevant features while maintaining full connectivity",
      "To enable parallel processing of sequential data while preserving temporal dependencies",
      "To increase model capacity through additional parameters while maintaining gradient flow"
    ],
    correctAnswer: 0,
    explanation: "Attention mechanisms allow networks to dynamically focus on different parts of the input by computing importance weights based on the current context, enabling better handling of long-range dependencies and variable-length inputs.",
    reference: "https://arxiv.org/abs/1706.03762"
  },
  {
    id: 5,
    question: "What is the key difference between Adam and AdamW optimizers?",
    options: [
      "AdamW decouples weight decay from the gradient update, preventing it from being scaled by the adaptive learning rate",
      "AdamW uses a different momentum calculation that better handles sparse gradients while maintaining adaptive learning rates",
      "AdamW automatically adjusts the learning rate based on layer depth while preserving the benefits of Adam's momentum",
      "AdamW combines multiple optimization strategies for better generalization while maintaining computational efficiency"
    ],
    correctAnswer: 0,
    explanation: "AdamW decouples weight decay from the gradient update, preventing it from being scaled by the adaptive learning rate, which leads to more effective regularization and better generalization.",
    reference: "https://arxiv.org/abs/1711.05101"
  },
  {
    id: 6,
    question: "What is the primary advantage of using group normalization over batch normalization?",
    options: [
      "It provides stable normalization across different batch sizes and is particularly effective for small batches",
      "It reduces memory usage by processing features in groups while maintaining the benefits of batch statistics",
      "It enables better feature disentanglement by normalizing related features together while preserving spatial information",
      "It automatically determines the optimal group size based on the input distribution while maintaining gradient flow"
    ],
    correctAnswer: 0,
    explanation: "Group normalization divides channels into groups and normalizes within each group, making it independent of batch size and particularly effective for small batches or when batch statistics are unreliable.",
    reference: "https://arxiv.org/abs/1803.08494"
  },
  {
    id: 7,
    question: "What is the key innovation in the Transformer architecture's self-attention mechanism?",
    options: [
      "It enables direct modeling of relationships between all positions in the sequence through parallel attention computation",
      "It uses multiple attention heads to process different aspects of the input simultaneously while maintaining computational efficiency",
      "It automatically determines the optimal attention span for each position while preserving temporal dependencies",
      "It combines convolutional and recurrent operations for better feature extraction while maintaining parallel processing"
    ],
    correctAnswer: 0,
    explanation: "Self-attention in Transformers computes attention weights between all positions in parallel, allowing direct modeling of long-range dependencies without the sequential processing required by RNNs.",
    reference: "https://arxiv.org/abs/1706.03762"
  },
  {
    id: 8,
    question: "What is the primary purpose of dropout in neural networks?",
    options: [
      "To prevent co-adaptation of neurons by randomly deactivating them during training",
      "To reduce model size by pruning unimportant connections while maintaining the network's representational capacity",
      "To speed up training by focusing on the most important features while preserving gradient flow",
      "To enable better gradient flow through the network by creating multiple parallel paths"
    ],
    correctAnswer: 0,
    explanation: "Dropout prevents co-adaptation of neurons by randomly deactivating them during training, forcing the network to learn robust features that don't rely on specific neurons.",
    reference: "https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf"
  },
  {
    id: 9,
    question: "What is the key advantage of using depthwise separable convolutions?",
    options: [
      "They significantly reduce the number of parameters while maintaining representational capacity",
      "They enable better feature extraction by processing each channel independently while preserving spatial relationships",
      "They automatically determine the optimal kernel size for each layer while maintaining computational efficiency",
      "They combine multiple convolution operations for better spatial understanding while reducing memory usage"
    ],
    correctAnswer: 0,
    explanation: "Depthwise separable convolutions split the standard convolution into a depthwise convolution followed by a pointwise convolution, dramatically reducing parameters while maintaining similar representational capacity.",
    reference: "https://arxiv.org/abs/1610.02357"
  },
  {
    id: 10,
    question: "What is the primary purpose of gradient clipping in training deep neural networks?",
    options: [
      "To prevent exploding gradients by limiting the maximum norm of the gradients",
      "To speed up convergence by focusing on the most important parameter updates while maintaining stability",
      "To reduce memory usage during backpropagation while preserving gradient information",
      "To enable better parallelization of gradient computation while maintaining numerical stability"
    ],
    correctAnswer: 0,
    explanation: "Gradient clipping prevents exploding gradients by limiting the maximum norm of the gradients, ensuring stable training in deep networks and preventing numerical instability.",
    reference: "https://arxiv.org/abs/1211.5063"
  },
  {
    id: 11,
    question: "What is the key innovation in the Mish activation function?",
    options: [
      "It combines the benefits of smoothness and self-gating while maintaining gradient flow",
      "It automatically adapts its shape based on the input distribution while preserving computational efficiency",
      "It uses multiple activation functions in parallel for better feature extraction while maintaining sparsity",
      "It reduces computational complexity while maintaining performance through optimized mathematical operations"
    ],
    correctAnswer: 0,
    explanation: "Mish (f(x) = x * tanh(softplus(x))) combines the benefits of smoothness and self-gating, providing better gradient flow and performance than ReLU in many cases.",
    reference: "https://arxiv.org/abs/1908.08681"
  },
  {
    id: 12,
    question: "What is the primary advantage of using weight normalization?",
    options: [
      "It decouples the length and direction of weight vectors, enabling faster convergence",
      "It reduces the number of parameters by normalizing weight matrices while maintaining model capacity",
      "It automatically determines the optimal learning rate for each layer while preserving gradient flow",
      "It enables better parallelization of weight updates while maintaining numerical stability"
    ],
    correctAnswer: 0,
    explanation: "Weight normalization decouples the length and direction of weight vectors, allowing for more stable training and faster convergence compared to standard normalization techniques.",
    reference: "https://arxiv.org/abs/1602.07868"
  },
  {
    id: 13,
    question: "What is the key innovation in the Transformer's positional encoding?",
    options: [
      "It uses sinusoidal functions of different frequencies to encode position information without adding parameters",
      "It automatically learns position embeddings through backpropagation while maintaining computational efficiency",
      "It combines multiple encoding schemes for better position representation while preserving parallel processing",
      "It uses convolutional operations to capture local position information while maintaining global context"
    ],
    correctAnswer: 0,
    explanation: "The Transformer's positional encoding uses sinusoidal functions of different frequencies to encode position information, allowing the model to generalize to sequences of different lengths without adding learnable parameters.",
    reference: "https://arxiv.org/abs/1706.03762"
  },
  {
    id: 14,
    question: "What is the primary purpose of label smoothing in classification tasks?",
    options: [
      "To prevent overconfident predictions by softening the target distribution",
      "To reduce the impact of noisy labels in the training data while maintaining model confidence",
      "To enable better handling of imbalanced datasets while preserving class relationships",
      "To speed up convergence by providing smoother gradients while maintaining prediction accuracy"
    ],
    correctAnswer: 0,
    explanation: "Label smoothing prevents overconfident predictions by replacing hard 0/1 labels with smoothed values, encouraging the model to be less certain about its predictions and improving generalization.",
    reference: "https://arxiv.org/abs/1512.00567"
  },
  {
    id: 15,
    question: "What is the key advantage of using layer-wise learning rate adaptation?",
    options: [
      "It allows different layers to learn at different rates based on their depth and importance",
      "It automatically determines the optimal batch size for each layer while maintaining gradient flow",
      "It reduces memory usage by processing layers sequentially while preserving model capacity",
      "It enables better parallelization of layer updates while maintaining numerical stability"
    ],
    correctAnswer: 0,
    explanation: "Layer-wise learning rate adaptation allows deeper layers to learn at different rates, typically with lower learning rates for earlier layers and higher rates for later layers, improving training stability.",
    reference: "https://arxiv.org/abs/1706.02677"
  },
  {
    id: 16,
    question: "What is the primary purpose of knowledge distillation?",
    options: [
      "To transfer knowledge from a large, complex model to a smaller, more efficient one",
      "To combine multiple models into a single, more powerful model while preserving their individual strengths",
      "To reduce overfitting by training on multiple datasets simultaneously while maintaining model performance",
      "To enable better parallelization of model training while preserving the original model's capabilities"
    ],
    correctAnswer: 0,
    explanation: "Knowledge distillation transfers knowledge from a large, complex model (teacher) to a smaller, more efficient one (student) by training the student to mimic the teacher's output distribution.",
    reference: "https://arxiv.org/abs/1503.02531"
  },
  {
    id: 17,
    question: "What is the key innovation in the EfficientNet architecture?",
    options: [
      "It uses compound scaling to uniformly scale network depth, width, and resolution",
      "It automatically determines the optimal network architecture through neural architecture search while maintaining efficiency",
      "It combines multiple efficient operations for better feature extraction while preserving computational resources",
      "It uses dynamic computation to adapt to different input sizes while maintaining model performance"
    ],
    correctAnswer: 0,
    explanation: "EfficientNet uses compound scaling to uniformly scale network depth, width, and resolution in a principled way, achieving better performance than conventional scaling methods.",
    reference: "https://arxiv.org/abs/1905.11946"
  },
  {
    id: 18,
    question: "What is the primary advantage of using mixed precision training?",
    options: [
      "It reduces memory usage and speeds up computation while maintaining model accuracy",
      "It automatically determines the optimal precision for each layer while preserving numerical stability",
      "It enables better parallelization of model training while maintaining gradient flow",
      "It reduces the number of parameters by using lower precision weights while preserving model capacity"
    ],
    correctAnswer: 0,
    explanation: "Mixed precision training uses FP16 for most operations while keeping critical computations in FP32, reducing memory usage and speeding up computation while maintaining model accuracy.",
    reference: "https://arxiv.org/abs/1710.03740"
  },
  {
    id: 19,
    question: "What is the key innovation in the Vision Transformer (ViT) architecture?",
    options: [
      "It applies the Transformer architecture directly to image patches without using convolutions",
      "It combines convolutional and attention operations for better feature extraction while maintaining computational efficiency",
      "It automatically determines the optimal patch size for each image while preserving spatial relationships",
      "It uses multiple attention mechanisms for different spatial scales while maintaining parallel processing"
    ],
    correctAnswer: 0,
    explanation: "ViT applies the Transformer architecture directly to sequences of image patches, demonstrating that convolutions are not necessary for strong performance on vision tasks.",
    reference: "https://arxiv.org/abs/2010.11929"
  },
  {
    id: 20,
    question: "What is the primary advantage of using quantization-aware training (QAT) over post-training quantization?",
    options: [
      "It preserves model accuracy better by simulating quantization during training",
      "It reduces the model size more effectively while maintaining computational efficiency",
      "It requires less computational resources during inference while preserving model performance",
      "It eliminates the need for calibration data while maintaining quantization accuracy"
    ],
    correctAnswer: 0,
    explanation: "Quantization-aware training (QAT) simulates the effects of quantization during the training process, allowing the model to learn to compensate for the precision loss. This typically results in better accuracy compared to post-training quantization, which applies quantization after the model is already trained.",
    reference: "https://arxiv.org/abs/1712.05877"
  },
  {
    id: 21,
    question: "What is the key innovation of the LoRA (Low-Rank Adaptation) method for fine-tuning large language models?",
    options: [
      "It freezes the pre-trained weights and adds trainable low-rank matrices",
      "It reduces the model size by pruning unimportant weights while maintaining model performance",
      "It quantizes the model weights to lower precision while preserving the original model's capabilities",
      "It splits the model across multiple GPUs while maintaining training efficiency"
    ],
    correctAnswer: 0,
    explanation: "LoRA's key innovation is freezing the pre-trained weights and adding trainable low-rank matrices that are much smaller than the original weight matrices. This allows for efficient fine-tuning with minimal memory overhead while maintaining most of the model's performance.",
    reference: "https://arxiv.org/abs/2106.09685"
  },
  {
    id: 22,
    question: "What is the primary purpose of using gradient checkpointing in training large neural networks?",
    options: [
      "To reduce memory usage by recomputing activations during backpropagation",
      "To accelerate the forward pass computation while maintaining numerical stability",
      "To improve model convergence by optimizing memory usage during training",
      "To enable distributed training while preserving gradient information"
    ],
    correctAnswer: 0,
    explanation: "Gradient checkpointing is a memory optimization technique that trades computation for memory. Instead of storing all intermediate activations during the forward pass, it stores only certain checkpoints and recomputes the intermediate values during backpropagation. This significantly reduces memory usage at the cost of additional computation.",
    reference: "https://arxiv.org/abs/1604.06174"
  },
  {
    id: 23,
    question: "What is the main advantage of using mixed precision training with FP16 and FP32?",
    options: [
      "It reduces memory usage while maintaining numerical stability",
      "It increases model accuracy by using higher precision for critical operations",
      "It eliminates the need for gradient clipping while maintaining training stability",
      "It enables training without batch normalization while preserving model performance"
    ],
    correctAnswer: 0,
    explanation: "Mixed precision training combines FP16 and FP32 to achieve both memory efficiency and numerical stability. FP16 reduces memory usage and speeds up computation, while FP32 maintains precision for critical operations like weight updates.",
    reference: "https://arxiv.org/abs/1710.03740"
  },
  {
    id: 24,
    question: "What is the key innovation of the Zero Redundancy Optimizer (ZeRO) in distributed training?",
    options: [
      "It eliminates memory redundancy by partitioning optimizer states across GPUs",
      "It reduces communication overhead between GPUs while maintaining training efficiency",
      "It compresses gradients before transmission while preserving gradient information",
      "It asynchronously updates model parameters while maintaining training stability"
    ],
    correctAnswer: 0,
    explanation: "ZeRO's key innovation is partitioning the optimizer states (parameters, gradients, and optimizer states) across GPUs to eliminate memory redundancy. This allows training of models that are much larger than the GPU memory capacity.",
    reference: "https://arxiv.org/abs/1910.02054"
  },
  {
    id: 25,
    question: "What is the primary purpose of using tensor parallelism in model deployment?",
    options: [
      "To split large weight matrices across multiple devices for inference",
      "To reduce the model's memory footprint while maintaining computational efficiency",
      "To accelerate the forward pass computation while preserving model accuracy",
      "To enable dynamic batch sizing while maintaining inference speed"
    ],
    correctAnswer: 0,
    explanation: "Tensor parallelism splits large weight matrices across multiple devices, allowing for inference of models that are too large to fit on a single device. This is particularly useful for large language models.",
    reference: "https://arxiv.org/abs/1909.08053"
  },
  {
    id: 26,
    question: "What is the key advantage of using speculative decoding for large language model inference?",
    options: [
      "It predicts multiple tokens ahead and verifies them in parallel",
      "It reduces the model's memory requirements while maintaining inference quality",
      "It improves the quality of generated text by using multiple verification steps",
      "It enables training with smaller batch sizes while maintaining model performance"
    ],
    correctAnswer: 0,
    explanation: "Speculative decoding uses a smaller, faster model to predict multiple tokens ahead, which are then verified in parallel by the larger model. This can significantly speed up inference while maintaining the same output quality.",
    reference: "https://arxiv.org/abs/2211.17192"
  },
  {
    id: 27,
    question: "What is the primary benefit of using continuous batching in model serving?",
    options: [
      "It maximizes GPU utilization by dynamically processing requests",
      "It reduces the model's memory footprint while maintaining inference speed",
      "It improves inference accuracy by processing similar requests together",
      "It enables training with larger batch sizes while maintaining efficiency"
    ],
    correctAnswer: 0,
    explanation: "Continuous batching dynamically combines multiple inference requests into a single batch, maximizing GPU utilization and throughput. It's particularly effective when requests arrive at different times.",
    reference: "https://arxiv.org/abs/2302.06112"
  },
  {
    id: 28,
    question: "What is the key innovation of the QLoRA method for fine-tuning large language models?",
    options: [
      "It combines quantization with LoRA for memory-efficient fine-tuning",
      "It reduces communication overhead in distributed training while maintaining model performance",
      "It eliminates the need for gradient checkpointing while preserving training stability",
      "It enables training without optimizer states while maintaining model accuracy"
    ],
    correctAnswer: 0,
    explanation: "QLoRA combines 4-bit quantization with LoRA to achieve extremely memory-efficient fine-tuning of large language models. It quantizes the pre-trained weights to 4 bits and uses LoRA for the fine-tuning updates.",
    reference: "https://arxiv.org/abs/2305.14314"
  },
  {
    id: 29,
    question: "What is the primary purpose of using activation checkpointing in training large neural networks?",
    options: [
      "To trade computation for memory by recomputing activations during backpropagation",
      "To accelerate the forward pass computation while maintaining numerical stability",
      "To improve model convergence by optimizing memory usage during training",
      "To enable distributed training while preserving gradient information"
    ],
    correctAnswer: 0,
    explanation: "Activation checkpointing is a memory optimization technique that stores only certain intermediate activations during the forward pass and recomputes the others during backpropagation. This significantly reduces memory usage at the cost of additional computation.",
    reference: "https://arxiv.org/abs/1604.06174"
  },
  {
    id: 30,
    question: "What is the main advantage of using structured pruning over unstructured pruning?",
    options: [
      "It produces models that are more efficient on hardware accelerators",
      "It achieves higher compression ratios while maintaining model performance",
      "It preserves model accuracy better by maintaining structural relationships",
      "It requires less fine-tuning after pruning while maintaining computational efficiency"
    ],
    correctAnswer: 0,
    explanation: "Structured pruning removes entire structures (like channels or filters) from the model, resulting in a regular architecture that can be efficiently executed on hardware accelerators.",
    reference: "https://arxiv.org/abs/1810.07322"
  },
  {
    id: 31,
    question: "What is the key innovation of the Chinchilla scaling laws for language models?",
    options: [
      "They optimize the compute budget allocation between model size and training tokens",
      "They reduce the memory requirements for training while maintaining model performance",
      "They eliminate the need for distributed training by optimizing model architecture",
      "They improve the quality of pre-training data by optimizing the training process"
    ],
    correctAnswer: 0,
    explanation: "The Chinchilla scaling laws provide an optimal allocation of compute budget between model size and the number of training tokens. They show that for a given compute budget, models should be smaller and trained on more data than previously thought.",
    reference: "https://arxiv.org/abs/2203.15556"
  },
  {
    id: 32,
    question: "What is the primary purpose of using adapter layers in fine-tuning large language models?",
    options: [
      "To add small, trainable layers between transformer layers while freezing the main model",
      "To reduce the model's memory footprint during inference while maintaining performance",
      "To accelerate the forward pass computation while preserving model accuracy",
      "To enable training with larger batch sizes while maintaining efficiency"
    ],
    correctAnswer: 0,
    explanation: "Adapter layers are small, trainable layers inserted between the transformer layers of a pre-trained model. The main model weights are frozen, and only the adapter layers are updated during fine-tuning.",
    reference: "https://arxiv.org/abs/1902.00751"
  },
  {
    id: 33,
    question: "What is the key advantage of using weight sharing in transformer models?",
    options: [
      "It reduces the number of parameters while maintaining model capacity",
      "It accelerates the forward pass computation while preserving model accuracy",
      "It improves model convergence by sharing gradient information across layers",
      "It enables training with smaller batch sizes while maintaining efficiency"
    ],
    correctAnswer: 0,
    explanation: "Weight sharing in transformer models (like in ALBERT) reduces the number of parameters by sharing weights across layers while maintaining similar model capacity.",
    reference: "https://arxiv.org/abs/1909.11942"
  },
  {
    id: 34,
    question: "What is the primary purpose of using knowledge distillation in model optimization?",
    options: [
      "To transfer knowledge from a large, complex model to a smaller, more efficient one",
      "To reduce the memory requirements for training while maintaining model performance",
      "To accelerate the forward pass computation while preserving model accuracy",
      "To enable training with larger batch sizes while maintaining efficiency"
    ],
    correctAnswer: 0,
    explanation: "Knowledge distillation transfers the knowledge learned by a large, complex model (teacher) to a smaller, more efficient model (student) by training the student to mimic the teacher's outputs.",
    reference: "https://arxiv.org/abs/1503.02531"
  },
  {
    id: 35,
    question: "What is the key innovation of the EfficientNet architecture?",
    options: [
      "It uses compound scaling to balance model depth, width, and resolution",
      "It reduces the number of parameters in convolutional networks while maintaining performance",
      "It eliminates the need for batch normalization while preserving training stability",
      "It enables training without data augmentation while maintaining model accuracy"
    ],
    correctAnswer: 0,
    explanation: "EfficientNet's key innovation is compound scaling, which uniformly scales the network's depth, width, and resolution in a principled way. This leads to better performance than scaling any single dimension.",
    reference: "https://arxiv.org/abs/1905.11946"
  },
  {
    id: 36,
    question: "What is the primary purpose of using progressive resizing in training neural networks?",
    options: [
      "To start training with smaller images and gradually increase their size while maintaining model stability",
      "To reduce the memory requirements for training through adaptive batch size adjustment",
      "To accelerate the forward pass computation by processing images at multiple scales simultaneously",
      "To enable training with larger batch sizes through dynamic resolution scaling"
    ],
    correctAnswer: 0,
    explanation: "Progressive resizing starts training with smaller images and gradually increases their size, which can speed up training and sometimes improve final performance.",
    reference: "https://arxiv.org/abs/1710.09829"
  },
  {
    id: 37,
    question: "What is the key advantage of using weight normalization in neural networks?",
    options: [
      "It decouples the weight vector's length from its direction",
      "It reduces the number of parameters in the model while maintaining capacity",
      "It eliminates the need for batch normalization while preserving training stability",
      "It enables training with larger learning rates while maintaining convergence"
    ],
    correctAnswer: 0,
    explanation: "Weight normalization decouples the weight vector's length from its direction, which can improve training dynamics and sometimes lead to better performance.",
    reference: "https://arxiv.org/abs/1602.07868"
  },
  {
    id: 38,
    question: "What is the primary purpose of using gradient accumulation in training large neural networks?",
    options: [
      "To effectively increase the batch size when memory is limited",
      "To reduce the memory requirements for training while maintaining performance",
      "To accelerate the forward pass computation while preserving accuracy",
      "To improve model convergence by optimizing gradient updates"
    ],
    correctAnswer: 0,
    explanation: "Gradient accumulation performs multiple forward/backward passes with smaller batches and accumulates the gradients before updating the weights. This effectively increases the batch size when memory is limited.",
    reference: "https://arxiv.org/abs/1706.02677"
  },
  {
    id: 39,
    question: "What is the key innovation of the Vision Transformer (ViT) architecture?",
    options: [
      "It applies the transformer architecture directly to sequences of image patches through learned patch embeddings",
      "It reduces the number of parameters in convolutional networks while maintaining performance through efficient attention",
      "It eliminates the need for positional encodings while preserving spatial information through relative position biases",
      "It enables training without data augmentation while maintaining model accuracy through architectural innovations"
    ],
    correctAnswer: 0,
    explanation: "ViT's key innovation is applying the transformer architecture directly to sequences of image patches, treating them like tokens in a language model.",
    reference: "https://arxiv.org/abs/2010.11929"
  },
  {
    id: 40,
    question: "What is the primary advantage of using model parallelism in distributed training?",
    options: [
      "It enables training models that are too large to fit on a single device through strategic model partitioning",
      "It reduces communication overhead between devices while maintaining efficiency through optimized data flow",
      "It accelerates the forward pass computation while preserving accuracy through parallel layer processing",
      "It enables better parallelization of data loading while maintaining throughput through distributed data handling"
    ],
    correctAnswer: 0,
    explanation: "Model parallelism splits the model across multiple devices, allowing training of models that are too large to fit on a single device.",
    reference: "https://arxiv.org/abs/1909.08053"
  },
  {
    id: 41,
    question: "What is the primary advantage of using dynamic batching in model serving?",
    options: [
      "It maximizes throughput by combining requests of different sizes into optimal batches while maintaining latency requirements",
      "It reduces the model's memory footprint by dynamically adjusting the batch size based on available resources",
      "It improves inference accuracy by processing similar requests together through adaptive batching strategies",
      "It enables training with larger batch sizes by automatically scaling the model's capacity during inference"
    ],
    correctAnswer: 0,
    explanation: "Dynamic batching combines requests of different sizes into optimal batches to maximize throughput while maintaining latency requirements. It's particularly effective when requests have varying computational requirements.",
    reference: "https://arxiv.org/abs/2302.06112"
  },
  {
    id: 42,
    question: "What is the key innovation of the FlashAttention mechanism?",
    options: [
      "It reduces memory reads/writes by computing attention in blocks and storing only the final output",
      "It eliminates the need for positional encodings by using relative position biases in attention computation",
      "It enables training without gradient checkpointing through optimized memory management",
      "It improves model accuracy by using more precise attention calculations with higher numerical precision"
    ],
    correctAnswer: 0,
    explanation: "FlashAttention's key innovation is computing attention in blocks and storing only the final output, significantly reducing memory reads/writes. This makes it much more memory-efficient than standard attention implementations.",
    reference: "https://arxiv.org/abs/2205.14135"
  },
  {
    id: 43,
    question: "What is the primary purpose of using KV-caching in transformer inference?",
    options: [
      "To reuse previously computed key-value pairs for autoregressive generation while maintaining computational efficiency",
      "To reduce the model's memory footprint during training through selective caching of important activations",
      "To accelerate the forward pass computation by parallelizing key-value pair generation",
      "To enable training with larger batch sizes by caching intermediate computations"
    ],
    correctAnswer: 0,
    explanation: "KV-caching stores previously computed key-value pairs during autoregressive generation, avoiding redundant computation and significantly speeding up inference.",
    reference: "https://arxiv.org/abs/2201.11990"
  },
  {
    id: 44,
    question: "What is the key advantage of using sparse attention in transformer models?",
    options: [
      "It reduces the quadratic complexity of attention to linear or sub-quadratic while maintaining model performance",
      "It eliminates the need for positional encodings by using relative position biases in sparse attention patterns",
      "It enables training without gradient checkpointing through optimized memory access patterns",
      "It improves model accuracy by focusing on more relevant tokens through learned sparsity patterns"
    ],
    correctAnswer: 0,
    explanation: "Sparse attention reduces the quadratic complexity of attention to linear or sub-quadratic by computing attention only between selected token pairs.",
    reference: "https://arxiv.org/abs/1904.10509"
  },
  {
    id: 45,
    question: "What is the primary purpose of using model quantization in deployment?",
    options: [
      "To reduce model size and inference latency while maintaining acceptable accuracy through precision reduction",
      "To improve model accuracy by using more precise calculations with higher numerical precision",
      "To eliminate the need for gradient checkpointing through optimized memory usage",
      "To enable training with larger batch sizes by reducing memory requirements"
    ],
    correctAnswer: 0,
    explanation: "Model quantization reduces model size and inference latency by using lower-precision numbers (e.g., 8-bit instead of 32-bit) while maintaining acceptable accuracy.",
    reference: "https://arxiv.org/abs/1712.05877"
  },
  {
    id: 46,
    question: "What is the key innovation of the Mixture of Experts (MoE) architecture?",
    options: [
      "It activates only a subset of experts for each input, enabling efficient scaling of model capacity",
      "It reduces the number of parameters by sharing weights across experts through weight tying",
      "It eliminates the need for gradient checkpointing through expert-level parallelization",
      "It improves model accuracy by combining multiple specialized models through ensemble learning"
    ],
    correctAnswer: 0,
    explanation: "MoE's key innovation is activating only a subset of experts for each input, allowing the model to scale its capacity efficiently while keeping computation constant.",
    reference: "https://arxiv.org/abs/1701.06538"
  },
  {
    id: 47,
    question: "What is the primary advantage of using weight tying in transformer models?",
    options: [
      "It reduces the number of parameters by sharing weights between input and output embeddings while maintaining model capacity",
      "It accelerates the forward pass computation through optimized memory access patterns",
      "It improves model convergence by sharing gradient information across layers",
      "It enables training with smaller batch sizes through memory-efficient weight updates"
    ],
    correctAnswer: 0,
    explanation: "Weight tying shares weights between input and output embeddings, significantly reducing the number of parameters in transformer models.",
    reference: "https://arxiv.org/abs/1608.05859"
  },
  {
    id: 48,
    question: "What is the key innovation of the Reformer architecture?",
    options: [
      "It uses locality-sensitive hashing to reduce the complexity of attention to O(L log L) while maintaining model performance",
      "It eliminates the need for positional encodings through learned position embeddings in the hash space",
      "It enables training without gradient checkpointing through optimized memory access patterns",
      "It improves model accuracy by using more precise attention calculations with locality-sensitive hashing"
    ],
    correctAnswer: 0,
    explanation: "Reformer's key innovation is using locality-sensitive hashing to reduce the complexity of attention from O(L²) to O(L log L), making it much more memory-efficient.",
    reference: "https://arxiv.org/abs/2001.04451"
  },
  {
    id: 49,
    question: "What is the primary purpose of using gradient accumulation in training large language models?",
    options: [
      "To effectively increase the batch size when memory is limited by accumulating gradients across multiple forward/backward passes",
      "To reduce the memory requirements for training through selective gradient storage",
      "To accelerate the forward pass computation by parallelizing gradient accumulation",
      "To improve model convergence by optimizing gradient updates through accumulation"
    ],
    correctAnswer: 0,
    explanation: "Gradient accumulation performs multiple forward/backward passes with smaller batches and accumulates the gradients before updating the weights. This effectively increases the batch size when memory is limited.",
    reference: "https://arxiv.org/abs/1706.02677"
  },
  {
    id: 50,
    question: "What is the key advantage of using mixed precision training with FP16 and FP32?",
    options: [
      "It reduces memory usage while maintaining numerical stability through careful precision management",
      "It increases model accuracy by using higher precision for critical operations with automatic precision selection",
      "It eliminates the need for gradient clipping through precision-aware optimization",
      "It enables training without batch normalization through precision-based regularization"
    ],
    correctAnswer: 0,
    explanation: "Mixed precision training combines FP16 and FP32 to achieve both memory efficiency and numerical stability. FP16 reduces memory usage and speeds up computation, while FP32 maintains precision for critical operations like weight updates.",
    reference: "https://arxiv.org/abs/1710.03740"
  },
  {
    id: 51,
    question: "What is the primary purpose of using progressive resizing in training neural networks?",
    options: [
      "To start training with smaller images and gradually increase their size while maintaining model stability",
      "To reduce the memory requirements for training through adaptive batch size adjustment",
      "To accelerate the forward pass computation by processing images at multiple scales simultaneously",
      "To enable training with larger batch sizes through dynamic resolution scaling"
    ],
    correctAnswer: 0,
    explanation: "Progressive resizing starts training with smaller images and gradually increases their size, which can speed up training and sometimes improve final performance.",
    reference: "https://arxiv.org/abs/1710.09829"
  },
  {
    id: 52,
    question: "What is the key innovation of the EfficientNet architecture?",
    options: [
      "It uses compound scaling to balance model depth, width, and resolution through a principled approach",
      "It reduces the number of parameters in convolutional networks through weight sharing and pruning",
      "It eliminates the need for batch normalization through alternative normalization techniques",
      "It enables training without data augmentation through architectural innovations"
    ],
    correctAnswer: 0,
    explanation: "EfficientNet's key innovation is compound scaling, which uniformly scales the network's depth, width, and resolution in a principled way. This leads to better performance than scaling any single dimension.",
    reference: "https://arxiv.org/abs/1905.11946"
  },
  {
    id: 53,
    question: "What is the primary purpose of using knowledge distillation in model optimization?",
    options: [
      "To transfer knowledge from a large, complex model to a smaller, more efficient one through output distribution matching",
      "To reduce the memory requirements for training through selective parameter freezing",
      "To accelerate the forward pass computation through parallel model execution",
      "To enable training with larger batch sizes through distributed knowledge transfer"
    ],
    correctAnswer: 0,
    explanation: "Knowledge distillation transfers the knowledge learned by a large, complex model (teacher) to a smaller, more efficient model (student) by training the student to mimic the teacher's outputs.",
    reference: "https://arxiv.org/abs/1503.02531"
  },
  {
    id: 54,
    question: "What is the key advantage of using weight sharing in transformer models?",
    options: [
      "It reduces the number of parameters while maintaining model capacity through cross-layer parameter reuse",
      "It accelerates the forward pass computation through optimized memory access patterns",
      "It improves model convergence by sharing gradient information across attention heads",
      "It enables training with smaller batch sizes through memory-efficient weight updates"
    ],
    correctAnswer: 0,
    explanation: "Weight sharing in transformer models (like in ALBERT) reduces the number of parameters by sharing weights across layers while maintaining similar model capacity.",
    reference: "https://arxiv.org/abs/1909.11942"
  },
  {
    id: 55,
    question: "What is the primary purpose of using adapter layers in fine-tuning large language models?",
    options: [
      "To add small, trainable layers between transformer layers while freezing the main model for efficient adaptation",
      "To reduce the model's memory footprint during inference through selective layer activation",
      "To accelerate the forward pass computation through parallel adapter processing",
      "To enable training with larger batch sizes through distributed adapter updates"
    ],
    correctAnswer: 0,
    explanation: "Adapter layers are small, trainable layers inserted between the transformer layers of a pre-trained model. The main model weights are frozen, and only the adapter layers are updated during fine-tuning.",
    reference: "https://arxiv.org/abs/1902.00751"
  },
  {
    id: 56,
    question: "What is the key innovation of the Chinchilla scaling laws for language models?",
    options: [
      "They optimize the compute budget allocation between model size and training tokens through empirical analysis",
      "They reduce the memory requirements for training while maintaining model performance through architectural innovations",
      "They eliminate the need for distributed training by optimizing model architecture through efficient scaling",
      "They improve the quality of pre-training data by optimizing the training process through data selection"
    ],
    correctAnswer: 0,
    explanation: "The Chinchilla scaling laws provide an optimal allocation of compute budget between model size and the number of training tokens. They show that for a given compute budget, models should be smaller and trained on more data than previously thought.",
    reference: "https://arxiv.org/abs/2203.15556"
  },
  {
    id: 57,
    question: "What is the main advantage of using structured pruning over unstructured pruning?",
    options: [
      "It produces models that are more efficient on hardware accelerators through regular sparsity patterns",
      "It achieves higher compression ratios through systematic weight removal while maintaining model performance",
      "It preserves model accuracy better by maintaining structural relationships between pruned components",
      "It requires less fine-tuning after pruning through structured weight updates"
    ],
    correctAnswer: 0,
    explanation: "Structured pruning removes entire structures (like channels or filters) from the model, resulting in a regular architecture that can be efficiently executed on hardware accelerators.",
    reference: "https://arxiv.org/abs/1810.07322"
  },
  {
    id: 58,
    question: "What is the primary purpose of using activation checkpointing in training large neural networks?",
    options: [
      "To trade computation for memory by recomputing activations during backpropagation through strategic checkpoint placement",
      "To accelerate the forward pass computation through parallel activation processing",
      "To improve model convergence by optimizing memory usage during training through adaptive checkpointing",
      "To enable distributed training through efficient activation sharing between devices"
    ],
    correctAnswer: 0,
    explanation: "Activation checkpointing is a memory optimization technique that stores only certain intermediate activations during the forward pass and recomputes the others during backpropagation. This significantly reduces memory usage at the cost of additional computation.",
    reference: "https://arxiv.org/abs/1604.06174"
  },
  {
    id: 59,
    question: "What is the key innovation of the Vision Transformer (ViT) architecture?",
    options: [
      "It applies the transformer architecture directly to sequences of image patches through learned patch embeddings",
      "It reduces the number of parameters in convolutional networks while maintaining performance through efficient attention",
      "It eliminates the need for positional encodings while preserving spatial information through relative position biases",
      "It enables training without data augmentation while maintaining model accuracy through architectural innovations"
    ],
    correctAnswer: 0,
    explanation: "ViT's key innovation is applying the transformer architecture directly to sequences of image patches, treating them like tokens in a language model.",
    reference: "https://arxiv.org/abs/2010.11929"
  },
  {
    id: 60,
    question: "What is the primary purpose of using model parallelism in distributed training?",
    options: [
      "It enables training models that are too large to fit on a single device through strategic model partitioning",
      "It reduces communication overhead between devices while maintaining efficiency through optimized data flow",
      "It accelerates the forward pass computation while preserving accuracy through parallel layer processing",
      "It enables better parallelization of data loading while maintaining throughput through distributed data handling"
    ],
    correctAnswer: 0,
    explanation: "Model parallelism splits the model across multiple devices, allowing training of models that are too large to fit on a single device.",
    reference: "https://arxiv.org/abs/1909.08053"
  },
  {
    id: 61,
    question: "What is the primary advantage of using the GELU activation function over ReLU?",
    options: [
      "It provides a smooth, differentiable approximation of ReLU through Gaussian error function integration",
      "It reduces the number of parameters in the network through adaptive activation thresholds",
      "It eliminates the need for batch normalization through self-normalizing properties",
      "It enables training without gradient clipping through bounded activation ranges"
    ],
    correctAnswer: 0,
    explanation: "GELU (Gaussian Error Linear Unit) provides a smooth, differentiable approximation of ReLU, which can help with gradient flow and model optimization.",
    reference: "https://arxiv.org/abs/1606.08415"
  },
  {
    id: 62,
    question: "What is the key innovation of the Swin Transformer architecture?",
    options: [
      "It uses shifted windows to enable cross-window connections while maintaining linear complexity through hierarchical feature processing",
      "It eliminates the need for positional encodings through learned spatial relationships in window partitions",
      "It reduces the number of parameters by sharing weights across windows through attention-based parameter sharing",
      "It enables training without gradient checkpointing through efficient memory management in window operations"
    ],
    correctAnswer: 0,
    explanation: "The Swin Transformer's key innovation is using shifted windows to enable cross-window connections while maintaining linear complexity with respect to image size.",
    reference: "https://arxiv.org/abs/2103.14030"
  },
  {
    id: 63,
    question: "What is the primary purpose of using the Mish activation function?",
    options: [
      "It combines the benefits of smoothness and self-gating while maintaining gradient flow through continuous differentiability",
      "It reduces the number of parameters in the network through adaptive activation thresholds",
      "It eliminates the need for batch normalization through self-normalizing properties",
      "It enables training without gradient clipping through bounded activation ranges"
    ],
    correctAnswer: 0,
    explanation: "Mish combines the benefits of smoothness and self-gating while maintaining good gradient flow, often outperforming ReLU in deep networks.",
    reference: "https://arxiv.org/abs/1908.08681"
  },
  {
    id: 64,
    question: "What is the key advantage of using the Focal Loss in object detection?",
    options: [
      "It addresses class imbalance by down-weighting easy examples and focusing on hard ones through adaptive loss scaling",
      "It reduces the number of parameters in the detection head through efficient feature reuse",
      "It eliminates the need for non-maximum suppression through confidence-based filtering",
      "It enables training without data augmentation through loss-based regularization"
    ],
    correctAnswer: 0,
    explanation: "Focal Loss addresses class imbalance by down-weighting easy examples and focusing on hard ones, which is particularly useful in object detection where background examples vastly outnumber objects.",
    reference: "https://arxiv.org/abs/1708.02002"
  },
  {
    id: 65,
    question: "What is the primary purpose of using the Label Smoothing technique?",
    options: [
      "To prevent overconfident predictions by softening the target distribution through uniform noise injection",
      "To reduce the number of parameters in the model through label-based regularization",
      "It eliminates the need for batch normalization through label-based normalization",
      "It enables training without gradient clipping through smoothed gradient updates"
    ],
    correctAnswer: 0,
    explanation: "Label Smoothing prevents overconfident predictions by softening the target distribution, which can improve model generalization.",
    reference: "https://arxiv.org/abs/1512.00567"
  },
  {
    id: 66,
    question: "What is the key innovation of the DETR (DEtection TRansformer) architecture?",
    options: [
      "It eliminates the need for hand-designed components like anchor boxes and NMS through end-to-end object detection",
      "It reduces the number of parameters in the detection head through efficient transformer-based feature extraction",
      "It enables training without data augmentation through architectural innovations",
      "It eliminates the need for positional encodings through learned spatial relationships"
    ],
    correctAnswer: 0,
    explanation: "DETR's key innovation is eliminating the need for hand-designed components like anchor boxes and NMS by using a transformer architecture to directly predict object detections.",
    reference: "https://arxiv.org/abs/2005.12872"
  },
  {
    id: 67,
    question: "What is the primary advantage of using the Contrastive Learning framework?",
    options: [
      "It learns representations by contrasting positive and negative pairs through similarity-based optimization",
      "It reduces the number of parameters in the model through efficient feature reuse",
      "It eliminates the need for data augmentation through contrastive regularization",
      "It enables training without batch normalization through contrastive normalization"
    ],
    correctAnswer: 0,
    explanation: "Contrastive Learning learns representations by contrasting positive and negative pairs, which can lead to better feature learning with unlabeled data.",
    reference: "https://arxiv.org/abs/2002.05709"
  },
  {
    id: 68,
    question: "What is the key innovation of the Vision Transformer (ViT) architecture?",
    options: [
      "It applies the transformer architecture directly to sequences of image patches through learned patch embeddings",
      "It reduces the number of parameters in convolutional networks while maintaining performance through efficient attention",
      "It eliminates the need for positional encodings while preserving spatial information through relative position biases",
      "It enables training without data augmentation while maintaining model accuracy through architectural innovations"
    ],
    correctAnswer: 0,
    explanation: "ViT's key innovation is applying the transformer architecture directly to sequences of image patches, treating them like tokens in a language model.",
    reference: "https://arxiv.org/abs/2010.11929"
  },
  {
    id: 69,
    question: "What is the primary purpose of using the Mixup data augmentation technique?",
    options: [
      "To create new training examples by linearly interpolating between pairs of examples through convex combinations",
      "To reduce the number of parameters in the model through mixup-based regularization",
      "It eliminates the need for batch normalization through mixup-based normalization",
      "It enables training without gradient clipping through smoothed gradient updates"
    ],
    correctAnswer: 0,
    explanation: "Mixup creates new training examples by linearly interpolating between pairs of examples and their labels, which can improve model robustness and generalization.",
    reference: "https://arxiv.org/abs/1710.09412"
  },
  {
    id: 70,
    question: "What is the key advantage of using the CutMix data augmentation technique?",
    options: [
      "It combines the benefits of Cutout and Mixup by replacing image regions with patches from other images through region-based mixing",
      "It reduces the number of parameters in the model through cutmix-based regularization",
      "It eliminates the need for batch normalization through cutmix-based normalization",
      "It enables training without gradient clipping through region-based gradient updates"
    ],
    correctAnswer: 0,
    explanation: "CutMix combines the benefits of Cutout and Mixup by replacing image regions with patches from other images, which can improve model robustness and localization ability.",
    reference: "https://arxiv.org/abs/1905.04899"
  },
  {
    id: 71,
    question: "What is the primary purpose of using the Stochastic Depth technique?",
    options: [
      "To randomly drop entire layers during training to improve model robustness through adaptive depth adjustment",
      "To reduce the number of parameters in the model through stochastic parameter pruning",
      "It eliminates the need for batch normalization through stochastic normalization",
      "It enables training without gradient clipping through stochastic gradient updates"
    ],
    correctAnswer: 0,
    explanation: "Stochastic Depth randomly drops entire layers during training, which can improve model robustness and training efficiency.",
    reference: "https://arxiv.org/abs/1603.09382"
  },
  {
    id: 72,
    question: "What is the key innovation of the EfficientNet architecture?",
    options: [
      "It uses compound scaling to balance model depth, width, and resolution through a principled approach",
      "It reduces the number of parameters in convolutional networks through weight sharing and pruning",
      "It eliminates the need for batch normalization through alternative normalization techniques",
      "It enables training without data augmentation through architectural innovations"
    ],
    correctAnswer: 0,
    explanation: "EfficientNet's key innovation is compound scaling, which uniformly scales the network's depth, width, and resolution in a principled way. This leads to better performance than scaling any single dimension.",
    reference: "https://arxiv.org/abs/1905.11946"
  },
  {
    id: 73,
    question: "What is the primary advantage of using the Squeeze-and-Excitation (SE) module?",
    options: [
      "It adaptively recalibrates channel-wise feature responses through learned channel-wise attention",
      "It reduces the number of parameters in the model through efficient channel compression",
      "It eliminates the need for batch normalization through channel-wise normalization",
      "It enables training without gradient clipping through channel-wise gradient updates"
    ],
    correctAnswer: 0,
    explanation: "The SE module adaptively recalibrates channel-wise feature responses by explicitly modeling interdependencies between channels.",
    reference: "https://arxiv.org/abs/1709.01507"
  },
  {
    id: 74,
    question: "What is the key innovation of the MobileNet architecture?",
    options: [
      "It uses depthwise separable convolutions to reduce computation and model size through efficient feature extraction",
      "It reduces the number of parameters by sharing weights across layers through cross-layer parameter reuse",
      "It eliminates the need for batch normalization through depthwise normalization",
      "It enables training without data augmentation through architectural innovations"
    ],
    correctAnswer: 0,
    explanation: "MobileNet's key innovation is using depthwise separable convolutions, which significantly reduce computation and model size while maintaining good performance.",
    reference: "https://arxiv.org/abs/1704.04861"
  },
  {
    id: 75,
    question: "What is the primary purpose of using the ShuffleNet architecture?",
    options: [
      "To enable efficient computation using pointwise group convolutions and channel shuffle through optimized feature processing",
      "To reduce the number of parameters by sharing weights across layers through group-wise parameter sharing",
      "It eliminates the need for batch normalization through group-wise normalization",
      "It enables training without data augmentation through architectural innovations"
    ],
    correctAnswer: 0,
    explanation: "ShuffleNet enables efficient computation using pointwise group convolutions and channel shuffle, which significantly reduces computation while maintaining good performance.",
    reference: "https://arxiv.org/abs/1707.01083"
  },
  {
    id: 76,
    question: "What is the key advantage of using the GhostNet architecture?",
    options: [
      "It generates more feature maps from cheap operations to reduce computation through efficient feature generation",
      "It reduces the number of parameters by sharing weights across layers through ghost feature sharing",
      "It eliminates the need for batch normalization through ghost feature normalization",
      "It enables training without data augmentation through architectural innovations"
    ],
    correctAnswer: 0,
    explanation: "GhostNet generates more feature maps from cheap operations, significantly reducing computation while maintaining good performance.",
    reference: "https://arxiv.org/abs/1911.11907"
  },
  {
    id: 77,
    question: "What is the primary purpose of using the Neural Architecture Search (NAS) technique?",
    options: [
      "To automatically discover optimal neural network architectures through efficient search strategies",
      "To reduce the number of parameters in the model through architecture-based pruning",
      "It eliminates the need for batch normalization through architecture-based normalization",
      "It enables training without data augmentation through architecture-based regularization"
    ],
    correctAnswer: 0,
    explanation: "NAS automatically discovers optimal neural network architectures by searching through a space of possible architectures.",
    reference: "https://arxiv.org/abs/1806.10282"
  },
  {
    id: 78,
    question: "What is the key innovation of the EfficientNetV2 architecture?",
    options: [
      "It uses progressive learning and adaptive regularization to improve training efficiency through dynamic scaling",
      "It reduces the number of parameters in convolutional networks through efficient architecture design",
      "It eliminates the need for batch normalization through progressive normalization",
      "It enables training without data augmentation through architectural innovations"
    ],
    correctAnswer: 0,
    explanation: "EfficientNetV2's key innovation is using progressive learning and adaptive regularization to improve training efficiency.",
    reference: "https://arxiv.org/abs/2104.00298"
  },
  {
    id: 79,
    question: "What is the primary advantage of using the RegNet architecture?",
    options: [
      "It provides a simple, effective design space for neural networks through systematic architecture design",
      "It reduces the number of parameters by sharing weights across layers through regularized parameter sharing",
      "It eliminates the need for batch normalization through regularized normalization",
      "It enables training without data augmentation through architectural innovations"
    ],
    correctAnswer: 0,
    explanation: "RegNet provides a simple, effective design space for neural networks by discovering general design principles.",
    reference: "https://arxiv.org/abs/2003.13678"
  },
  {
    id: 80,
    question: "What is the key innovation of the ConvNeXt architecture?",
    options: [
      "It modernizes the ResNet architecture with design choices from transformers through architectural improvements",
      "It reduces the number of parameters by sharing weights across layers through modernized parameter sharing",
      "It eliminates the need for batch normalization through modernized normalization",
      "It enables training without data augmentation through architectural innovations"
    ],
    correctAnswer: 0,
    explanation: "ConvNeXt modernizes the ResNet architecture with design choices from transformers, achieving competitive performance.",
    reference: "https://arxiv.org/abs/2201.03545"
  },
  {
    id: 81,
    question: "What is the primary purpose of using the Swish activation function?",
    options: [
      "It provides a smooth, non-monotonic function that can better handle negative inputs through self-gating properties",
      "It reduces the number of parameters in the network through adaptive activation thresholds",
      "It eliminates the need for batch normalization through self-normalizing properties",
      "It enables training without gradient clipping through bounded activation ranges"
    ],
    correctAnswer: 0,
    explanation: "Swish provides a smooth, non-monotonic function that can better handle negative inputs while maintaining the benefits of ReLU.",
    reference: "https://arxiv.org/abs/1710.05941"
  },
  {
    id: 82,
    question: "What is the key advantage of using the Mish activation function?",
    options: [
      "It combines the benefits of smoothness and self-gating while maintaining gradient flow through continuous differentiability",
      "It reduces the number of parameters in the network through adaptive activation thresholds",
      "It eliminates the need for batch normalization through self-normalizing properties",
      "It enables training without gradient clipping through bounded activation ranges"
    ],
    correctAnswer: 0,
    explanation: "Mish combines the benefits of smoothness and self-gating while maintaining good gradient flow, often outperforming ReLU in deep networks.",
    reference: "https://arxiv.org/abs/1908.08681"
  },
  {
    id: 83,
    question: "What is the primary purpose of using the Focal Loss in object detection?",
    options: [
      "It addresses class imbalance by down-weighting easy examples and focusing on hard ones through adaptive loss scaling",
      "It reduces the number of parameters in the detection head through efficient feature reuse",
      "It eliminates the need for non-maximum suppression through confidence-based filtering",
      "It enables training without data augmentation through loss-based regularization"
    ],
    correctAnswer: 0,
    explanation: "Focal Loss addresses class imbalance by down-weighting easy examples and focusing on hard ones, which is particularly useful in object detection where background examples vastly outnumber objects.",
    reference: "https://arxiv.org/abs/1708.02002"
  },
  {
    id: 84,
    question: "What is the key innovation of the DETR (DEtection TRansformer) architecture?",
    options: [
      "It eliminates the need for hand-designed components like anchor boxes and NMS through end-to-end object detection",
      "It reduces the number of parameters in the detection head through efficient transformer-based feature extraction",
      "It enables training without data augmentation through architectural innovations",
      "It eliminates the need for positional encodings through learned spatial relationships"
    ],
    correctAnswer: 0,
    explanation: "DETR's key innovation is eliminating the need for hand-designed components like anchor boxes and NMS by using a transformer architecture to directly predict object detections.",
    reference: "https://arxiv.org/abs/2005.12872"
  },
  {
    id: 85,
    question: "What is the primary advantage of using the Contrastive Learning framework?",
    options: [
      "It learns representations by contrasting positive and negative pairs through similarity-based optimization",
      "It reduces the number of parameters in the model through efficient feature reuse",
      "It eliminates the need for data augmentation through contrastive regularization",
      "It enables training without batch normalization through contrastive normalization"
    ],
    correctAnswer: 0,
    explanation: "Contrastive Learning learns representations by contrasting positive and negative pairs, which can lead to better feature learning with unlabeled data.",
    reference: "https://arxiv.org/abs/2002.05709"
  },
  {
    id: 86,
    question: "What is the key innovation of the Vision Transformer (ViT) architecture?",
    options: [
      "It applies the transformer architecture directly to sequences of image patches through learned patch embeddings",
      "It reduces the number of parameters in convolutional networks while maintaining performance through efficient attention",
      "It eliminates the need for positional encodings while preserving spatial information through relative position biases",
      "It enables training without data augmentation while maintaining model accuracy through architectural innovations"
    ],
    correctAnswer: 0,
    explanation: "ViT's key innovation is applying the transformer architecture directly to sequences of image patches, treating them like tokens in a language model.",
    reference: "https://arxiv.org/abs/2010.11929"
  },
  {
    id: 87,
    question: "What is the primary purpose of using the Mixup data augmentation technique?",
    options: [
      "To create new training examples by linearly interpolating between pairs of examples through convex combinations",
      "To reduce the number of parameters in the model through mixup-based regularization",
      "It eliminates the need for batch normalization through mixup-based normalization",
      "It enables training without gradient clipping through smoothed gradient updates"
    ],
    correctAnswer: 0,
    explanation: "Mixup creates new training examples by linearly interpolating between pairs of examples and their labels, which can improve model robustness and generalization.",
    reference: "https://arxiv.org/abs/1710.09412"
  },
  {
    id: 88,
    question: "What is the key advantage of using the CutMix data augmentation technique?",
    options: [
      "It combines the benefits of Cutout and Mixup by replacing image regions with patches from other images through region-based mixing",
      "It reduces the number of parameters in the model through cutmix-based regularization",
      "It eliminates the need for batch normalization through cutmix-based normalization",
      "It enables training without gradient clipping through region-based gradient updates"
    ],
    correctAnswer: 0,
    explanation: "CutMix combines the benefits of Cutout and Mixup by replacing image regions with patches from other images, which can improve model robustness and localization ability.",
    reference: "https://arxiv.org/abs/1905.04899"
  },
  {
    id: 89,
    question: "What is the primary purpose of using the Stochastic Depth technique?",
    options: [
      "To randomly drop entire layers during training to improve model robustness through adaptive depth adjustment",
      "To reduce the number of parameters in the model through stochastic parameter pruning",
      "It eliminates the need for batch normalization through stochastic normalization",
      "It enables training without gradient clipping through stochastic gradient updates"
    ],
    correctAnswer: 0,
    explanation: "Stochastic Depth randomly drops entire layers during training, which can improve model robustness and training efficiency.",
    reference: "https://arxiv.org/abs/1603.09382"
  },
  {
    id: 90,
    question: "What is the key innovation of the EfficientNet architecture?",
    options: [
      "It uses compound scaling to balance model depth, width, and resolution through a principled approach",
      "It reduces the number of parameters in convolutional networks through weight sharing and pruning",
      "It eliminates the need for batch normalization through alternative normalization techniques",
      "It enables training without data augmentation through architectural innovations"
    ],
    correctAnswer: 0,
    explanation: "EfficientNet's key innovation is compound scaling, which uniformly scales the network's depth, width, and resolution in a principled way. This leads to better performance than scaling any single dimension.",
    reference: "https://arxiv.org/abs/1905.11946"
  },
  {
    id: 91,
    question: "What is the primary advantage of using the Squeeze-and-Excitation (SE) module?",
    options: [
      "It adaptively recalibrates channel-wise feature responses through learned channel-wise attention",
      "It reduces the number of parameters in the model through efficient channel compression",
      "It eliminates the need for batch normalization through channel-wise normalization",
      "It enables training without gradient clipping through channel-wise gradient updates"
    ],
    correctAnswer: 0,
    explanation: "The SE module adaptively recalibrates channel-wise feature responses by explicitly modeling interdependencies between channels.",
    reference: "https://arxiv.org/abs/1709.01507"
  },
  {
    id: 92,
    question: "What is the key innovation of the MobileNet architecture?",
    options: [
      "It uses depthwise separable convolutions to reduce computation and model size through efficient feature extraction",
      "It reduces the number of parameters by sharing weights across layers through cross-layer parameter reuse",
      "It eliminates the need for batch normalization through depthwise normalization",
      "It enables training without data augmentation through architectural innovations"
    ],
    correctAnswer: 0,
    explanation: "MobileNet's key innovation is using depthwise separable convolutions, which significantly reduce computation and model size while maintaining good performance.",
    reference: "https://arxiv.org/abs/1704.04861"
  },
  {
    id: 93,
    question: "What is the primary purpose of using the ShuffleNet architecture?",
    options: [
      "To enable efficient computation using pointwise group convolutions and channel shuffle through optimized feature processing",
      "To reduce the number of parameters by sharing weights across layers through group-wise parameter sharing",
      "It eliminates the need for batch normalization through group-wise normalization",
      "It enables training without data augmentation through architectural innovations"
    ],
    correctAnswer: 0,
    explanation: "ShuffleNet enables efficient computation using pointwise group convolutions and channel shuffle, which significantly reduces computation while maintaining good performance.",
    reference: "https://arxiv.org/abs/1707.01083"
  },
  {
    id: 94,
    question: "What is the key advantage of using the GhostNet architecture?",
    options: [
      "It generates more feature maps from cheap operations to reduce computation through efficient feature generation",
      "It reduces the number of parameters by sharing weights across layers through ghost feature sharing",
      "It eliminates the need for batch normalization through ghost feature normalization",
      "It enables training without data augmentation through architectural innovations"
    ],
    correctAnswer: 0,
    explanation: "GhostNet generates more feature maps from cheap operations, significantly reducing computation while maintaining good performance.",
    reference: "https://arxiv.org/abs/1911.11907"
  },
  {
    id: 95,
    question: "What is the primary purpose of using the Neural Architecture Search (NAS) technique?",
    options: [
      "To automatically discover optimal neural network architectures through efficient search strategies",
      "To reduce the number of parameters in the model through architecture-based pruning",
      "It eliminates the need for batch normalization through architecture-based normalization",
      "It enables training without data augmentation through architecture-based regularization"
    ],
    correctAnswer: 0,
    explanation: "NAS automatically discovers optimal neural network architectures by searching through a space of possible architectures.",
    reference: "https://arxiv.org/abs/1806.10282"
  },
  {
    id: 96,
    question: "What is the key innovation of the EfficientNetV2 architecture?",
    options: [
      "It uses progressive learning and adaptive regularization to improve training efficiency through dynamic scaling",
      "It reduces the number of parameters in convolutional networks through efficient architecture design",
      "It eliminates the need for batch normalization through progressive normalization",
      "It enables training without data augmentation through architectural innovations"
    ],
    correctAnswer: 0,
    explanation: "EfficientNetV2's key innovation is using progressive learning and adaptive regularization to improve training efficiency.",
    reference: "https://arxiv.org/abs/2104.00298"
  },
  {
    id: 97,
    question: "What is the primary advantage of using the RegNet architecture?",
    options: [
      "It provides a simple, effective design space for neural networks through systematic architecture design",
      "It reduces the number of parameters by sharing weights across layers through regularized parameter sharing",
      "It eliminates the need for batch normalization through regularized normalization",
      "It enables training without data augmentation through architectural innovations"
    ],
    correctAnswer: 0,
    explanation: "RegNet provides a simple, effective design space for neural networks by discovering general design principles.",
    reference: "https://arxiv.org/abs/2003.13678"
  },
  {
    id: 98,
    question: "What is the key innovation of the ConvNeXt architecture?",
    options: [
      "It modernizes the ResNet architecture with design choices from transformers through architectural improvements",
      "It reduces the number of parameters by sharing weights across layers through modernized parameter sharing",
      "It eliminates the need for batch normalization through modernized normalization",
      "It enables training without data augmentation through architectural innovations"
    ],
    correctAnswer: 0,
    explanation: "ConvNeXt modernizes the ResNet architecture with design choices from transformers, achieving competitive performance.",
    reference: "https://arxiv.org/abs/2201.03545"
  },
  {
    id: 99,
    question: "What is the primary purpose of using the Swish activation function?",
    options: [
      "It provides a smooth, non-monotonic function that can better handle negative inputs through self-gating properties",
      "It reduces the number of parameters in the network through adaptive activation thresholds",
      "It eliminates the need for batch normalization through self-normalizing properties",
      "It enables training without gradient clipping through bounded activation ranges"
    ],
    correctAnswer: 0,
    explanation: "Swish provides a smooth, non-monotonic function that can better handle negative inputs while maintaining the benefits of ReLU.",
    reference: "https://arxiv.org/abs/1710.05941"
  },
  {
    id: 100,
    question: "What is the key advantage of using the Mish activation function?",
    options: [
      "It combines the benefits of smoothness and self-gating while maintaining gradient flow through continuous differentiability",
      "It reduces the number of parameters in the network through adaptive activation thresholds",
      "It eliminates the need for batch normalization through self-normalizing properties",
      "It enables training without gradient clipping through bounded activation ranges"
    ],
    correctAnswer: 0,
    explanation: "Mish combines the benefits of smoothness and self-gating while maintaining good gradient flow, often outperforming ReLU in deep networks.",
    reference: "https://arxiv.org/abs/1908.08681"
  }
].map(randomizeAnswers); 