### Step 4: What is the task accuracy achieved after applying various quantization bit-width configurations to the SQuAD dataset?

- Uniform quantization was applied.
- Bit-width can be dynamically configured based on accuracy and resource constraints.
- The optimal configuration is where accuracy and efficiency curves intersect.
- This yields significant memory savings with minimal accuracy degradation.

---

### Step 4: How did you determine the optimal quantization bit-width configurations?

#### Coarse, Layer-wise Quantization
- 12 layers total (`h.0` to `h.11`).
- All layers initialized at 32 bits.
- One layer at a time is switched to 4-bit to measure sensitivity.
- Observations:
  - Most layers are sensitive, even with a single-layer change.
  - Full-precision score is 34.
  - Coarse granularity causes a ≥7-point drop in performance.
  - Layers 2, 7, 8, 10, and 11 are relatively less sensitive.

#### Fine-grained, Sub-module Quantization
- 48 total submodules: 4 per layer × 12 layers.
- Default: 32 bits for all submodules.
- Switch one submodule at a time to 4-bit to assess sensitivity.

#### Greedy Quantization
- Start with all submodules at 32-bit.
- Gradually flip submodules to 4-bit until EM ≤ 31 (10% drop in accuracy).
- Final results:
  - **EM**: _[not filled]_
  - **F1**: _[not filled]_
  - **Memory savings**: _[not filled]_
- Flipped submodules:
  - `h.11.attn.c_attn`

---

### Step 4: Additional training objectives for switchable quantization

- **Specialized Training**:
  - Focus QAT only on the most sensitive layers.
  - Run extra training cycles on these selectively quantized layers.

- **Dropout-style Training**:
  - Conceptually the inverse of specialized training.
  - Quantize all layers *except* for a few randomly selected "dropout" layers.

---

### Step 5: Alignment with CPT (ICLR'21)

- CPT replaces static training (4→8→16→32...) with cyclic patterns (4→8→16→32→16→8→4...).
- CPT leads to:
  - Better performance at 8- and 16-bit.
  - Worse performance at 32-bit.
  - Reason: More training time spent at 8- and 16-bit.
- Heatmap results show CPT accuracy is generally lower than static training.

---

### Step 6: Alignment with Double-Win Quant (ICML'21)

- **DWQ Observations**:
  - 8-bit models are more robust to adversarial attacks than 32-bit.
  - Gradient masking/saturation from quantization makes attacks harder.

- **Adversarial Attack Methods**:
  - **Simple typos** (used in this experiment)
  - **LM-attack**: Dependency issues
  - **Textfooler**: Requires a classification head

- **Observations Under Attack**:
  - Accuracy dropped significantly under attack.
  - Relative trends across bit-widths remained consistent.
  - Possible explanation: robustness differences between vision and language domains.

---

### Future Research Directions

- Train LoRA modules independently for each bit-width.
- Proposed improvements:
  - **Loss Averaging**: Use average loss over all bit-widths to encourage generalization.
  - **Teacher Distillation**: Encourage alignment between 4-bit and 32-bit outputs to improve 4-bit learning.
