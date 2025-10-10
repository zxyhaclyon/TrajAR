We sincerely appreciate the reviewer's thoughtful and constructive feedback. We are especially grateful for the reviewer's openness to raising the score if our responses adequately address the concerns. Below, we provide detailed responses to each point.

---

### **[W1] On the Scope of Experiments and Evaluations**

**Reviewer Comment:**
> "The paper is very ambitious in the beginning but the experiments are kind of narrow. It mentions road users, complex road intersection scenarios. But no evaluations on types of road users. Even the visualizations are simple scenarios (fig 5) not complex scenarios."

**Our Response:**
We sincerely thank the reviewer for this critical and constructive feedback. We agree that a comprehensive evaluation is crucial for a paper aiming at complex intersection scenarios. In our initial submission, we focused on establishing overall state-of-the-art performance. Following your suggestion, we have now significantly expanded our experimental analysis to provide a much deeper and more nuanced validation of TrajAR's capabilities, directly addressing the points raised.

**1. Detailed Per-Category Performance Analysis:**
We have conducted a new analysis breaking down the performance by road user type. The results, as shown in the new Table 4 in our revised manuscript (and summarized below for the SinD-TianJin dataset), confirm that TrajAR performs consistently and robustly across all categories—**vehicles, bicycles, and pedestrians**. This demonstrates that our model effectively handles the diverse dynamics and behaviors of different road users, a core challenge in intersection prediction.

**2. Robustness Evaluation in Peak vs. Off-Peak Traffic:**
To directly answer the concern about "complex scenarios," we went a step further. We evaluated our model's performance not just on average, but under varying levels of traffic complexity. We split the test data into **Off-Peak** (lower density, simpler interactions) and **Peak** (high density, complex interactions) periods. We compared TrajAR against one of the strongest baselines, **KI-GAN**, across all road user types.

The results are revealing and are presented in the new supplementary table below:

##### **Supplementary Table: Per-Category Performance on SinD-TianJin under Peak vs. Off-Peak Conditions (ADE, ×10⁻²)**

| **Road User Type** | **Model**  | **Off-Peak** | **Peak**  | **Performance Drop (Peak vs. Off-Peak)** |
| ------------------ | ---------- | ------------ | --------- | ---------------------------------------- |
| **Vehicles**       | **TrajAR** | **0.295**    | **0.320** | **+8.5%**                                |
|                    | KI-GAN     | 0.310        | 0.395     | +27.4%                                   |
| **Bicycles**       | **TrajAR** | **0.610**    | **0.680** | **+11.5%**                               |
|                    | KI-GAN     | 0.650        | 0.850     | +30.8%                                   |
| **Pedestrians**    | **TrajAR** | **0.405**    | **0.435** | **+7.4%**                                |
|                    | KI-GAN     | 0.420        | 0.540     | +28.6%                                   |

This analysis yields two critical findings that strongly support our claims:
*   **All models experience performance degradation during peak hours**, which is expected as interactions become more complex and dense.
*   **Crucially, TrajAR's performance degradation is significantly smaller than KI-GAN's**. While KI-GAN performs competitively in off-peak conditions, its accuracy drops markedly (over 27% on average) when scene complexity increases. In contrast, TrajAR maintains high accuracy with a much smaller performance drop (under 10% on average).

This demonstrates that **TrajAR's true advantage lies in its robustness and superior performance in genuinely complex, high-interaction scenarios**, which is the primary focus of our work.

We believe these substantial additions—**per-category breakdown, a rigorous complexity-based evaluation**—thoroughly address the reviewer's concern and robustly validate TrajAR's performance across the diverse and complex scenarios found in real-world urban intersections.

---

### **[W2] Clarity of Methodology and Notations**

> *"Methodology especially multi-scale decoder is not easy to understand. Too many notations and annotations. Some notations are not even used in paragraph such as latent embeddings epsilon in Equ 8 even though it is used in figure2."*

**Our Response:**

We sincerely apologize for the lack of clarity and the notational inconsistencies. We will thoroughly revise the manuscript to streamline the notations and provide a more intuitive explanation.

**1.Intuitive Explanation of the Multi-Scale Decoder:**

The core innovation is to reformulate autoregressive prediction from a **temporal** domain (next-step) to a **spatial-resolution** domain (next-scale). This design directly combats error accumulation in long-term forecasting.

*   **Analogy:** Imagine sketching a trajectory. First, you draw the rough path ($S_1$). Then, you add detail, making curves smoother and positions more precise ($S_2$, $S_3$). You avoid drawing point-by-point from start to finish, which compounds errors.

*   **Formal Workflow:**
    1.  **Input:** The encoder's output $O_{his}$ and the last observed position $S_0 = \tau_{t_0}$.
    2.  **Coarse Prediction ($k=1$):** The decoder predicts $\hat{S}_1$, a low-resolution trajectory. This establishes the long-term path backbone.
    3.  **Refinement ($k>1$):** To predict the next finer scale $\hat{S}_k$:
        a.  We apply **cubic interpolation** on the previous scale $\hat{S}_{k-1}$ to generate a preliminary, higher-resolution sequence $\bar{\mathcal{T}}_k$.
        b.  This sequence is mapped to latent embeddings $\mathcal{E}_k = \langle e_{t_0 + \frac{1 \cdot t_f}{2^{k-1}}}, \ldots \rangle$.
        c.  The **TrajAR Transformer Decoder** takes $\mathcal{E}_k$ and all previous coarser scales as input, refining the interpolated points into the final prediction $\hat{S}_k$.
    4.  **Output:** The final output is the finest scale $\hat{S}_K$, which has the full target resolution (e.g., 32 points).

**2.Clarification on Latent Embeddings $\mathcal{E}_k$:**

The reviewer is correct. The latent embeddings $\mathcal{E}_k$ are a crucial input to the TrajAR Transformer decoder in Fig. 2(b). They represent the *interpolated trajectory* in a latent feature space, enabling non-linear refinement. We will amend the text to state explicitly:

 "*The sequence of latent embeddings $\mathcal{E}_k$ serves as the input to the TrajAR Transformer decoder (Fig. 2b), which refines these points to produce the prediction $\hat{S}_k$.*"

---

#### **[W3] Connection between Cubic Interpolation and Transformer**

> *"The connection between cubic interpolation and transformer encoder is not clear. Better explain more."*

**Our Response:**

Thank you for this question. The cubic interpolation is a **key component within the decoder**, acting as a **structural and kinematic prior** for the Transformer.

The connection is central to the decoder's multi-scale process:

1.  **Role of Cubic Interpolation:** After predicting a coarse trajectory $\hat{S}_{k-1}$, we need a preliminary guess for the next scale. Cubic interpolation provides this by fitting a smooth curve (cubic spline) through the coarse points. This ensures the preliminary path is **kinematically feasible**—maintaining smoothness in position, velocity (1st derivative), and acceleration/curvature (2nd derivative).

2.  **Bridge to the Transformer:** The interpolated sequence $\bar{\mathcal{T}}_k$ is projected into latent embeddings $\mathcal{E}_k$. The Transformer decoder uses these embeddings as its input.

3.  **Synergy:**
    *   **Without interpolation,** the Transformer must generate the finer-scale path from scratch, a difficult task prone to error propagation.
    *   **With cubic interpolation,** the Transformer is provided a "good draft" that respects basic motion constraints. Its role becomes **refinement**: making localized adjustments based on learned interaction patterns (e.g., yielding), correcting the draft where necessary.

In essence, cubic interpolation handles **low-level, physics-based smoothness**, while the Transformer handles **high-level, interaction-based reasoning**. This division of labor is key to our model's accuracy and stability.

---

### **[W4] Consistency of Notation in Figure 2**

> *"Are the tau hat 32 down blow the same as tau hat 32 on the top in figure 2?"*

**Our Response:**

Yes, they represent the **exact same entity**. The $\hat{\tau}_{32}$ at the top of Fig. 2(b) is the final point in the predicted sequence for the finest scale $\hat{S}_K$. The $\hat{\tau}_{32}$ at the bottom is a highlighted representation of this final output.

We acknowledge this can be confusing. We will revise the figure to unify the notation, for example, by removing the duplicate at the bottom and adding a clearer annotation to the output sequence.

---

### **[W5] Generalization Across Datasets and Hyperparameters**

> *"Are model trained and tested on those 4 datasets respectively? The parameters of K etc are different for those datasets in Implementation settings. I doubt if the model can generalize well because it seems the model highly depends on the settings."*

**Our Response:**

This is a crucial point regarding generalization. Let us clarify the setup and argue why the results demonstrate strong generalization.

**1. Training and Hyperparameter Tuning:**

Yes, the model was **independently trained and tested on each dataset**. Hyperparameters ($K_r$, $l_d$, $N_a$, $N_l$) were tuned per dataset using a validation set. This is standard practice because datasets have inherent differences:
*   $N_a$, $N_l$ are **dataset-specific properties** (number of road user types and signal states). It is logical to adjust these.
*   Tuning $K_r$ and $l_d$ allows the model to adapt its capacity to each intersection's complexity.

The **core architecture** of TrajAR—the multi-scale decoder, Motion MoE, interaction-aware encoder—**remains unchanged**. The tuned hyperparameters allow this robust architecture to specialize its capacity effectively for each environment. 

Furthermore, TrajAR outperforms other recent models (e.g., KI-GAN, FJMP) that also underwent per-dataset tuning. This head-to-head comparison on multiple benchmarks strongly supports that TrajAR's architectural advances lead to superior and more generalizable performance.

---

We hope this detailed response has adequately addressed all your concerns. We are committed to incorporating these clarifications to significantly improve the manuscript's readability and rigor. Thank you once again for your valuable time and constructive criticism.

