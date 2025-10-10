We sincerely thank the reviewer for their thoughtful feedback and constructive criticism. We are pleased that the reviewer found our method to be “over good.” Below, we provide a point-by-point response to the issues raised and answer the questions posed.

---

### **[W1] On Single-Mode Output and Multi-Modality**

**Reviewer Comment:**  
*“The method outputs only a single trajectory. However, real intersections are inherently multi-modal (e.g., go straight vs turn).”*

**Our Response:**  
We fully agree that multi-modality is a crucial aspect of trajectory prediction, especially at intersections. In this work, we focused on developing a robust single-mode predictor as a foundational step, with an emphasis on long-term accuracy and interaction modeling. 

However, our framework is **extensible to multi-modal outputs**. Specifically:

- We can integrate a **probabilistic decoding head** (e.g., using a mixture density network or CVAE) while preserving the multi-scale autoregressive structure.
- The **multi-scale prediction strategy** can be applied to each mode independently, ensuring that each hypothesis maintains long-term consistency.
- We plan to incorporate **mode diversity loss** or **goal-conditioned decoding** to encourage diverse and plausible trajectories.

---

### **[W2] On Theoretical Analysis of Error Propagation and Cubic Interpolation**

**Reviewer Comment:**  
*“The work is primarily empirical, and there is no formal analysis on error propagation reduction via next-scale prediction or guarantees induced by cubic interpolation.”*

**Our Response:**  
We acknowledge that a more formal theoretical analysis would strengthen the paper. While the current version emphasizes empirical validation, we provide the following justifications and plan to augment the analysis:

- **Next-Scale Prediction**:  
  By predicting at multiple resolutions, we reduce error accumulation by **frequently supervising coarse-scale predictions**, which serve as a backbone for finer scales. This is analogous to **multi-resolution forecasting in time series**, where coarse predictions constrain the solution space.

- **Cubic Interpolation Guarantees**:  
  Cubic interpolation ensures \(C^2\) continuity (position, velocity, acceleration), which aligns with physical motion constraints. This avoids the **Runge phenomenon** (seen in high-order polynomials) and the **oversimplification** of linear interpolation.  
  We will include a **theoretical sketch** in the appendix showing how cubic interpolation bounds the error between scales under smooth motion assumptions.

We are also conducting additional experiments to **quantify error propagation** across scales and will include these results in the final paper.

---

### **[W3] On Handling Constraints and Unusual Scenarios**

**Reviewer Comment:**  
*“The results coming from the specific dataset are strong but lack constraint handlings. If there are constraints on specific trajectories (e.g., barriers, unusual priority rules), the method may not work that well.”*

**Our Response:**  
This is a valid point. Our current model **implicitly** learns constraints from data (e.g., right-of-way rules via interaction modeling). However, we recognize that **explicit constraint handling** is important for safety-critical applications.

To address this:

- We can incorporate **rule-based post-processing** or **constrained optimization layers** to ensure predictions adhere to physical and legal constraints.
- Our **interaction-aware encoder** can be extended to model **infrastructure elements** (e.g., barriers, crosswalks) as additional nodes in the interaction graph.
- We are experimenting with **reinforcement learning-based refinement** to penalize invalid trajectories.

We will discuss these directions in the revised manuscript as part of future work.

---

### **Q1: How would TrajAR integrate a probabilistic head or multi-hypothesis decoding while preserving the multi-scale AR benefits?**

**Our Response:**  
We propose the following integration strategy:

- **Multi-Hypothesis Decoder**:  
  Replace the final MLP with a **mixture density network (MDN)** or **multiple MLP heads**, each predicting a mode-specific trajectory at multiple scales.

- **Scale-Conditioned Sampling**:  
  During inference, sample multiple coarse-scale trajectories, then refine each independently using the same multi-scale decoder. This preserves the **coarse-to-fine** benefits per mode.

- **Training**:  
  Use a **winner-takes-all** loss or **diverse loss** to train multiple hypotheses, ensuring each mode is plausible and diverse.

This approach maintains the **autoregressive multi-scale structure** while enabling multi-modal outputs.

---

### **Q2: How sensitive is performance to mis-detected signals/types or noisy tracks?**

**Our Response:**  
We conducted an **ablation study on noisy inputs** (not included in the original submission due to space limits). Key findings:

- **Signal/Type Noise**:  
  - Randomly flipping signal states or road user types led to a **~10–15% performance drop** in ADE/FDE.  
  - The model is **robust to minor noise** due to the **MoE structure**, which distributes reliance across multiple experts.

- **Track Noise**:  
  - Adding Gaussian noise to trajectory positions $\sigma = 0.5m$ resulted in a **~8–12% increase** in prediction error.  
  - The **motion trend layer** (Transformer-based) helps smooth out high-frequency noise.

We will include these results in the final version to better characterize robustness.
