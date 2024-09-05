# Conditional score guidance using data_class$n$
$\nabla_x log \hat{p}(y|x,t;\theta)$

## Toy setting 1: noisy 
```python
self.data_class1 = torch.stack([torch.from_numpy(np.random.choice(np.arange(self.alphabet_size), size=args.toy_seq_len, replace=True)) for _ in range(num_seq)])

self.data_class2 = torch.stack([torch.from_numpy(np.random.choice(np.arange(self.alphabet_size), size=args.toy_seq_len, replace=True)) for _ in range(num_seq)])

```
The reason `data_class1` and `data_class2` are generated using the same procedure in the provided code is that, without any additional modifications or constraints, both datasets are being drawn from the same uniform distribution over the available classes. This could be done for simplicity in a toy experiment, where the goal is to test or demonstrate certain aspects of the classifier-guided generative modeling procedure rather than working with real, distinct classes.

### Possible Reasons for `data_class1` and `data_class2` Being the Same:

1. **Toy Example for Testing**:
   - The setup might be intentionally simple to test the implementation of the algorithm or to debug the code. Using the same distribution for both classes allows you to focus on the mechanics of the classifier guidance without the additional complexity of handling different class distributions.

2. **Uniformity in Class Distribution**:
   - In some scenarios, you might want to start with uniformly distributed data for both classes and then introduce differences later (e.g., through transformations, noise, or other conditioning variables). This can serve as a baseline before adding complexity.

3. **Simplified Scenario**:
   - The experiment might be designed to simulate a scenario where classes are initially indistinguishable, and the model’s task is to learn subtle differences or to verify that the model doesn't rely on initial biases in the data.

4. **Placeholder for More Complex Data**:
   - The current setup might be a placeholder, where `data_class1` and `data_class2` are generated identically for now, but in a real experiment, you would replace them with datasets that reflect actual differences between classes.

### How to Introduce Differences Between `data_class1` and `data_class2`:

If the intent is to simulate two distinct classes, you would typically introduce some differences in how `data_class1` and `data_class2` are generated. Here are a few ways to do that:

1. **Different Ranges for Classes**:
   - You could sample `data_class1` and `data_class2` from different ranges or subsets of the `alphabet_size`:
     ```python
     self.data_class1 = torch.stack([torch.from_numpy(np.random.choice(np.arange(self.alphabet_size // 2), size=args.toy_seq_len, replace=True)) for _ in range(args.toy_num_seq)])
     self.data_class2 = torch.stack([torch.from_numpy(np.random.choice(np.arange(self.alphabet_size // 2, self.alphabet_size), size=args.toy_seq_len, replace=True)) for _ in range(args.toy_num_seq)])
     ```

2. **Different Probabilities for Classes**:
   - Assign different probabilities to each class for sampling:
     ```python
     probs_class1 = np.random.dirichlet(np.ones(self.alphabet_size // 2))
     probs_class2 = np.random.dirichlet(np.ones(self.alphabet_size // 2))

     self.data_class1 = torch.stack([torch.from_numpy(np.random.choice(np.arange(self.alphabet_size // 2), size=args.toy_seq_len, replace=True, p=probs_class1)) for _ in range(args.toy_num_seq)])
     self.data_class2 = torch.stack([torch.from_numpy(np.random.choice(np.arange(self.alphabet_size // 2, self.alphabet_size), size=args.toy_seq_len, replace=True, p=probs_class2)) for _ in range(args.toy_num_seq)])
     ```

3. **Introduce Biases**:
   - You could introduce a bias in the generation process, such as always favoring certain numbers or patterns for one class and not the other.

4. **Noisy or Transformed Data**:
   - Apply a transformation or add noise to one of the datasets:
     ```python
     noise = np.random.normal(0, 0.1, size=(args.toy_seq_len,))
     self.data_class2 = torch.stack([torch.from_numpy(np.random.choice(np.arange(self.alphabet_size), size=args.toy_seq_len, replace=True) + noise) for _ in range(args.toy_num_seq)])
     ```

### Summary:

- **Identical Data**: The current setup where `data_class1` and `data_class2` are the same may be a simplified or initial test case in a toy experiment.
- **Differentiation**: To simulate distinct classes, you would typically adjust the generation process to create meaningful differences between `data_class1` and `data_class2`.
- **Experimentation**: This allows you to explore how the classifier guidance handles or learns from these differences, validating the robustness and functionality of your model.

## Toy setting 2: FM /noisy

Certainly! To create a larger ensemble for a tensor of size \(6 \times 6\) with elements 0 or 1, and divided into two classes, we can generate variations of the basic patterns (all 0s for one class and all 1s for the other). These variations can include random noise and partially flipped elements to create a richer dataset that still represents the original class structure.

### Objective

- **Class 1 (`data_class1`)**: Tensors where the majority of elements are `0`.
- **Class 2 (`data_class2`)**: Tensors where the majority of elements are `1`.

### Steps to Create the Ensemble

1. **Clean Data**: Start with clean, representative tensors for each class:
   - `data_class1`: All elements are `0`.
   - `data_class2`: All elements are `1`.

2. **Introduce Noisy Variants**: Generate noisy versions of each class where some elements are randomly flipped.

3. **Mixed Variants**: Generate tensors that are mixtures of 0s and 1s but biased towards the primary class (e.g., 75% 0s for `data_class1` and 75% 1s for `data_class2`).

### Implementation

```python
import torch

num_samples = 100  # Number of samples per variant
size = 6  # Size of the tensor (6x6)

# 1. Clean data
data_class1_clean = torch.zeros((num_samples, size, size))
data_class2_clean = torch.ones((num_samples, size, size))

# 2. Noisy variants
noise_level = 0.1  # Probability of flipping an element
data_class1_noisy = torch.bernoulli(torch.full((num_samples, size, size), noise_level))
data_class2_noisy = 1 - data_class1_noisy  # Flipping the elements for class 2

# 3. Mixed variants
mix_ratio = 0.75  # 75% of the elements are from the primary class
data_class1_mixed = torch.bernoulli(torch.full((num_samples, size, size), mix_ratio))
data_class2_mixed = 1 - data_class1_mixed  # Flipping the elements for class 2

# Combine all variants into larger ensembles
data_class1 = torch.cat([data_class1_clean, data_class1_noisy, data_class1_mixed], dim=0)
data_class2 = torch.cat([data_class2_clean, data_class2_noisy, data_class2_mixed], dim=0)

print("data_class1 shape:", data_class1.shape)  # Should be (300, 6, 6)
print("data_class2 shape:", data_class2.shape)  # Should be (300, 6, 6)
```

### Explanation:

1. **Clean Data**:
   - `data_class1_clean`: Tensors with all elements set to `0`.
   - `data_class2_clean`: Tensors with all elements set to `1`.

2. **Noisy Variants**:
   - `data_class1_noisy`: Tensors where each element has a `10%` chance of being `1` (i.e., 90% chance of being `0`).
   - `data_class2_noisy`: Flipped version of `data_class1_noisy` (90% chance of being `1` and 10% chance of being `0`).

3. **Noisy outliers**:
```python
   - random_class = torch.stack([torch.from_numpy(np.random.choice(np.arange(self.alphabet_size), size=args.toy_seq_len, replace=True)) for _ in range(num_seq*2)]).to(device=device, dtype=torch.int64)
```

### Combined Dataset

By concatenating these variants, you create a dataset where each class has 300 examples (100 clean, 100 noisy, and 100 mixed).
```python
        self.data_class1 = torch.cat([data_class1_clean, data_class2_clean, data_class1_noisy, data_class2_noisy], dim=0)
        self.data_class2 = torch.cat([random_class,], dim=0)
```

### Adjustments

- **Noise Level**: The noise level (`noise_level`) and mix ratio (`mix_ratio`) can be adjusted depending on how much variability you want in your dataset.
- **Sample Size**: The number of samples (`num_samples`) per variant can also be adjusted.

This approach creates a more robust ensemble for each class, which can help in training models that generalize better, especially in tasks involving noisy data or classification under uncertainty.

## Toy setting 3: All Spin down / noisy

### Combined Dataset
```python
        self.data_class1 = torch.cat([data_class1_clean, data_class1_noisy], dim=0)
        self.data_class2 = torch.cat([random_class, data_class2_clean, data_class2_noisy], dim=0)
```