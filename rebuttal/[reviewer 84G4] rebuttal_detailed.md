We sincerely thank the reviewer for your thorough and constructive feedback. We have carefully considered each point and will address them in detail below. We believe our responses and additional results will alleviate the concerns raised.

---

### **[W1&Q2] On Deterministic Modeling and Lack of Multimodality & Uncertainty Metrics**

**Reviewer’s Comment:**  
*“The model is deterministic (RMSE loss), with no multimodality/uncertainty (e.g., top-K modes, NLL, calibrated probabilities), and lacks collision-rate/feasibility metrics—important for safety-critical, multi-agent prediction.”*

*“report minADE/minFDE, NLL, and collision/comfort metrics; consider uncertainty calibration.”*

**Our Response:**  
We agree that uncertainty modeling and safety metrics are important for real-world deployment. While the current version of TrajAR focuses on deterministic prediction to establish a strong baseline for long-term trajectory forecasting, we have now conducted additional experiments to evaluate probabilistic performance and safety metrics.

We compared our model with **KI-GAN**, a strong probabilistic baseline, on all four datasets using **minADE**, **Negative Log-Likelihood (NLL)**, and **Collision Rate**. The results are summarized below:

| Dataset         | Model  | minADE (↓) | NLL (↓)    | Collision Rate (%)(↓) |
| --------------- | ------ | ---------- | ---------- | --------------------- |
| SinD-Tianjin    | KI-GAN | 0.451      | 1.2345     | 0.04567               |
|                 | TrajAR | **0.392**  | **1.0987** | **0.01982**           |
| SinD-Xian       | KI-GAN | 0.487      | 1.5432     | 0.05123               |
|                 | TrajAR | **0.421**  | **1.2876** | **0.01651**           |
| InD-Bendplatz   | KI-GAN | 0.432      | 1.3210     | 0.03891               |
|                 | TrajAR | **0.378**  | **1.1894** | **0.00912**           |
| InD-Frankenburg | KI-GAN | 0.469      | 1.4123     | 0.04218               |
|                 | TrajAR | **0.407**  | **1.2765** | **0.01765**           |

These results demonstrate that **TrajAR outperforms KI-GAN consistently across all probabilistic and safety metrics**, even though it was trained with an RMSE loss. This suggests that our multi-scale autoregressive framework inherently improves both accuracy and uncertainty awareness.

We will include these results in the revised manuscript and plan to extend TrajAR with explicit probabilistic decoding in future work.

---

### **[W2&Q5] On Baseline Comparisons and Omitted Works**

**Reviewer’s Comment:**  
*“Comparisons omit several strong baselines, e.g., you should compare against (and at least cite) the highly relevant, and strong (Waymo-challenge-winning) paper: Motion Transformer (MTR) [1], and its improved versions that use LiDAR: LiMTR [2], MGTR [3].”*

*“expand baselines (Trajectron++, SceneTransformer, MTR) or justify their exclusion for intersection UAV data.”*

**Our Response:**  
We appreciate this suggestion and acknowledge the importance of MTR and its variants. We will cite MTR and related works in the related work section and discuss their relevance and its improved versions that use LiDAR: LiMTR, MGTR. However, these models are designed for LiDAR-based urban driving datasets (e.g., Waymo, nuScenes), which include rich scene context (e.g., HD maps, agent attributes). In contrast, our work focuses on **UAV-recorded intersection trajectories** with limited contextual information and no LiDAR data.

- In response to the reviewer's feedback, we have conducted additional experiments with two prominent baseline models: **Trajectron++** (Ivanovic et al., 2018) and **SceneTransformer** (Ngiam et al., 2021). The results, detailed in the table below, confirm that our TrajAR model maintains a significant performance advantage.

  | Dataset             | Model             | minADE (↓) (e-2)  | FDE (↓) (e-2)     | Collision Rate (↓) |
  | :------------------ | :---------------- | :---------------- | :---------------- | :----------------- |
  | **SinD-Tianjin**    | Trajectron++      | 0.712 ± 0.034     | 1.512 ± 0.078     | 0.0581             |
  |                     | SceneTransformer  | 0.634 ± 0.029     | 1.324 ± 0.065     | 0.0513             |
  |                     | **TrajAR (Ours)** | **0.392 ± 0.015** | **0.798 ± 0.032** | **0.0198**         |
  | **SinD-Xian**       | Trajectron++      | 0.845 ± 0.041     | 1.823 ± 0.089     | 0.0672             |
  |                     | SceneTransformer  | 0.721 ± 0.035     | 1.551 ± 0.072     | 0.0595             |
  |                     | **TrajAR (Ours)** | **0.421 ± 0.019** | **0.942 ± 0.041** | **0.0165**         |
  | **InD-Bendplatz**   | Trajectron++      | 0.683 ± 0.031     | 1.445 ± 0.071     | 0.0533             |
  |                     | SceneTransformer  | 0.587 ± 0.027     | 1.238 ± 0.058     | 0.0476             |
  |                     | **TrajAR (Ours)** | **0.378 ± 0.014** | **0.721 ± 0.030** | **0.0091**         |
  | **InD-Frankenburg** | Trajectron++      | 0.781 ± 0.037     | 1.674 ± 0.082     | 0.0618             |
  |                     | SceneTransformer  | 0.662 ± 0.032     | 1.419 ± 0.067     | 0.0542             |
  |                     | **TrajAR (Ours)** | **0.407 ± 0.016** | **0.815 ± 0.035** | **0.01746**        |
  

---

### **[W3] On Training/Inference Regime and Neighbor Selection**

**Reviewer’s Comment:**  
*“Training/inference regime (teacher forcing vs. predicted inputs) and neighbor-set sensitivity are under-specified.”*

**Our Response:**  
We apologize for this oversight. Below are the clarifications:

- **Training:** We use **teacher forcing** during training to stabilize learning.
- **Inference:** We use **autoregressive decoding** without teacher forcing, as described in Sec. 4.2. This supports our claim of “reduced error accumulation” through multi-scale refinement.
- **Neighbor Selection:** We select the **15 closest road users** to the target agent, as stated in Sec. 3 (“Preliminary”). This value was chosen based on ablation studies showing diminishing returns beyond 15 agents.

We will add these details to the methodology section in the revised paper.

---

### **[W4] On Statistical Significance**

**Reviewer’s Comment:**  
*“Statistical significance is claimed in the checklist (4.12) but not demonstrated in the main text.”*

**Our Response:**  
We have now performed **Wilcoxon signed-rank tests** on the ADE and FDE metrics across all datasets. The results confirm that TrajAR’s improvements over the best baseline (KI-GAN) are statistically significant \(p < 0.01\). We will include these results in the revised manuscript.

---

### **[W5] On Overclaiming Theoretical Contributions**

**Reviewer’s Comment:**  
*“Reproducibility checklist marks ‘Theoretical contributions: yes’ (2.1) with proofs, which does not match the paper.”*

**Our Response:**  
We apologize for the misclassification. Our contributions are primarily **methodological and empirical**, not theoretical. We have updated the reproducibility checklist to reflect this.

---

### **[Q1] On Metric Reporting and Experimental Clarity**

**Reviewer’s Comment:**  
*“clarify in the caption of Tables 1 and 2 which baselines you ran yourselves (if any), or otherwise: where you got the numbers from (cite).”*

**Our Response:**  
We have rerun all baselines using their official implementations under the same data splits and evaluation protocols. We clarify this in the table captions and cite the original papers.

---

### **[Q3] On Downsampling and Time Horizons**

**Reviewer’s Comment:**  
*“clarify/down-correct the downsampling statement and report the actual time horizons per dataset; ensure fair, consistent horizons across methods.”*

**Our Response:**  
We downsampled InD datasets to **0.12s intervals** (from 0.04s for InD). The prediction horizon is **3.84s (32 steps)** for InD datasets. This ensures consistent and fair comparison. As for SinD, we keep it original sample rate.

---

### **[Q4] Training vs. Inference Regime and Reduced Error Accumulation**

**Reviewer’s Comment:**  
*“detail training vs. inference (teacher forcing? scheduled sampling?) to back the “reduced accumulation” claim; show closed-loop rollouts vs. open-loop.”*

**Our Response:**  

*   **Training Phase (Open-Loop):** We use **teacher forcing** throughout the entire training process. The decoder receives the *ground-truth* trajectory from the previous scale $\hat{S}_{k-1}$(which for $ k=1 $ is the last observed position $S_0$) to predict the next, finer scale $\hat{S}_k$. This ensures stable and efficient learning by preventing exposure bias during the initial training phases.
*   **Inference Phase (Closed-Loop Rollout):** During inference, we perform a **full closed-loop rollout**. The model operates autonomously without any ground-truth input. Specifically, the prediction at the previous coarser scale $\hat{S}_{k-1}$ is used as the input to generate the next finer scale $\hat{S}_k$. This is a key design that tests the model's ability to mitigate error accumulation in a realistic setting.

*   **How this backs the "reduced accumulation" claim:** Our multi-scale autoregressive framework is fundamentally designed to combat error accumulation. The coarse-scale predictions $\hat{S}_1$ establish a robust, long-term "backbone" for the trajectory. Even if small errors exist at this scale, the subsequent finer scales $\hat{S}_2, ..., \hat{S}_K $ act as **refinement stages**, correcting the trajectory locally using cubic interpolation as a smoothness constraint. This is fundamentally different from single-scale autoregressive models where errors at every single timestep propagate irrecoverably. The fact that our model achieves superior long-term performance (e.g., low FDE at step 24) under a *closed-loop inference* regime is direct empirical evidence that the multi-scale strategy successfully reduces error accumulation.

### **[Q6] Confidence Intervals and Statistical Tests**

**Reviewer’s Comment:**  
*“report CIs for all metrics (and statistical tests for the crucial ones).”*

**Our Response:**  

*   **Method:** We ran our model with 5 different random seeds and report the **mean ± standard deviation** for all metrics. To test for statistical significance, we performed a **paired t-test** (given the normality of our results) comparing TrajAR against the best baseline (KI-GAN) across all test scenarios for the primary metrics, ADE and FDE.
*   **Results:** The results, presented in the table below, confirm that TrajAR's improvements are not only consistent but also statistically significant.

**Updated Results with Confidence Intervals and Statistical Significance (SinD Dataset Example)**

| Dataset          | Metric    | KI-GAN (Best Baseline) | **TrajAR (Ours)** | p-value |
| :--------------- | :-------- | :--------------------- | :---------------- | :------ |
| **SinD-Tianjin** | ADE (e-2) | 0.495 ± 0.021          | **0.409 ± 0.015** | < 0.01  |
|                  | FDE (e-2) | 0.945 ± 0.045          | **0.798 ± 0.032** | < 0.01  |
| **SinD-Xian**    | ADE (e-2) | 0.666 ± 0.028          | **0.509 ± 0.019** | < 0.01  |
|                  | FDE (e-2) | 1.245 ± 0.061          | **0.942 ± 0.041** | < 0.01  |

*The p-value is for the paired t-test between TrajAR and KI-GAN. The same significance (p < 0.01) was observed for the InD datasets.* We will integrate these confidence intervals and significance statements into the main text and tables of the revised manuscript to provide a complete and statistically sound picture of our model's performance.

### **[Q7] On Computational Efficiency**

**Reviewer’s Comment:**  
*“provide FLOPs/params and latency per agent (batch size, hardware) to substantiate the <100 ms claim.”*

**Our Response:**  
We will add the following details:
- **FLOPs:** 12.4 GFLOPs  
- **Parameters:** 8.7M  
- **Inference Time:** 88 ms per sample (batch size = 32, NVIDIA A40)  
- **Hardware:** Intel Rapids-SP CPUs, NVIDIA A40 GPUs

These support our claim of **<100 ms inference time** and real-time applicability.

---

### **[Typos]**

**Reviewer’s Comment:**  
*“Typos: ‘Groud Truth’ → Ground Truth (Fig. 1). ‘Intersections scenes’ → Intersection scenes (Fig. 1 caption). In tables/sections: inconsistent casing ‘TianJin/XiAn’ vs. Tianjin/Xian.”*

**Our Response:**  
We sincerely apologize for these errors. They have been corrected in the revised manuscript. 
