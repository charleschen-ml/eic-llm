## Step 4 – Implementation Details

### ✅ What is the task accuracy achieved after applying various quantization bit-width configurations to the SQuAD dataset?

I evaluated Exact Match (EM) and F1 scores after fine-tuning GPT-2 on the SQuAD dataset under multiple quantization setups:

| Configuration                            | Training Loss (Final) | EM Score | F1 Score |
|-----------------------------------------|------------------------|----------|----------|
| Full Precision (no quantization)        | 3.11                   | 41.00    | 53.66    |
| All Layers @ 8-bit (Config 3)           | 5.95                   | 0.10     | 0.68     |
| Mixed 4/8-bit (even/odd layers)         | 6.57                   | 0.00     | 0.20     |
| Only `transformer.h.0` @ 8-bit          | 3.61                   | 22.00    | 29.71    |
| Only `transformer.h.0` @ 16-bit         | 3.90                   | 18.00    | 28.51    |
| Only `transformer.h.6` @ 8-bit          | 3.14                   | 36.00    | 46.83    |
| Only `transformer.h.6` @ 16-bit         | 3.13                   | 37.00    | 48.83    |
| Only `transformer.h.11` @ 8-bit         | 3.15                   | 40.00    | 54.69    |

**Observations:**
- Full quantization across all layers severely degrades performance.
- Selective quantization of later layers (e.g., `transformer.h.11`) retains accuracy close to full-precision.
- Lower-bit quantization (e.g., 4-bit) is highly lossy unless targeted.

---

### ✅ How did you determine the optimal quantization bit-width configurations? Have you gleaned any insights from your observations that could guide future work to further enhance performance?

To determine optimal configurations, I explored bit-width and layer sensitivity through:

1. Uniform 8-bit quantization across all layers → poor performance.
2. Selective per-layer quantization → varied sensitivity observed.
3. Trials excluding `lm_head` → improved stability and output.
4. Comparison of early vs. late layers → later layers tolerate quantization better.

**Key Insights:**
- Sensitivity to quantization varies by layer position.
- Later layers (e.g., `h.11`) are more robust to quantization than earlier ones.
- Excluding `lm_head` from quantization avoids generation degradation.
- 4-bit quantization requires advanced handling (e.g., smooth quant, better init).

These results suggest a promising direction in using **learned or greedy per-layer bit allocation** instead of uniform quantization.

---

### ✅ A motivation behind switchable quantization is to support diverse layer-wise quantization configurations simultaneously, accommodating different resource allocation needs. Could you suggest additional training objectives that could more effectively facilitate the mechanism for switching quantization bit-widths?

To better enable dynamic, switchable quantization, the following training objectives could be introduced:

#### 1. Multi-bit Consistency Loss
Encourage consistent behavior across multiple bit-widths by computing outputs at two bit-widths and minimizing their divergence:
```python
L = CE(output_8bit, labels) + λ * KL(output_8bit || output_4bit)
```

#### 2. Bit-Width Dropout
Randomly sample bit-widths per layer during training, similar to dropout/stochastic depth:
```python
bitwidth = random.choice([4, 8, 16])
```

#### 3. Bit-Aware Regularization
Penalize higher-bit usage to encourage efficient representations:
```python
L_total = L_task + α * sum(bitwidths)
```

#### 4. Knowledge Distillation
Use outputs from the full-precision model as soft targets to guide training under sampled quantized settings.

---

These techniques aim to make the model robust across varying quantization configurations, aligning well with the motivation behind switchable quantization.
