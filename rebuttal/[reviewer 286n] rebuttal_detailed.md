We sincerely thank the reviewer for your valuable feedback. We have carefully considered the comments and will address the concerns regarding the **clarity of motivation** and **explicit summary of novelty** in our revised manuscript. Below, we provide a detailed response to each point.

---

### **1. On the Motivation of the Paper**    

**Reviewer Comment:**  
*“The motivation of the paper is not sufficiently clear. At present, the manuscript does not explicitly explain which concrete limitations in current trajectory prediction methods remain unresolved and why they are important.”*

**Our Response:**  
We appreciate this comment and acknowledge that the motivation can be made more explicit. In the revised manuscript, we will clearly articulate the **two major limitations of existing methods** that our work aims to address:

- **Limitation 1: Inadequate Modeling of Implicit Interactions in Complex Intersection Scenarios**  
  Most existing methods (e.g., CS-LSTM, PiP, WSiP) are designed for highway or simple interaction scenarios, where interactions are often homogeneous and limited. In contrast, urban intersections involve **diverse road users** (vehicles, bicycles, pedestrians) and **complex implicit interactions** (e.g., yielding, right-of-way rules). Existing methods fail to **dynamically model these implicit interactions** in a way that accounts for both motion trends and traffic rules. For example, KI-GAN and FJMP incorporate traffic signals but do not fully capture the **dynamic and multi-type interactions** that occur in real-world intersections.

- **Limitation 2: Error Accumulation and Overall Bias in Long-Term Prediction**  
  Existing autoregressive methods suffer from **error accumulation** over long horizons, while non-autoregressive methods often produce **kinematically unrealistic trajectories** due to the lack of step-wise correction. This is especially critical in safety-sensitive intersection scenarios where long-term accuracy is essential.

Our proposed **TrajAR** framework directly addresses these limitations through:
- A **dynamic interaction-aware encoder** that models multi-type road users and implicit interactions using a mixture of experts (MoE) and graph attention networks.
- A **multi-scale autoregressive decoder** that refines trajectories from coarse to fine scales, mitigating error accumulation and ensuring kinematic plausibility.

These points will be explicitly stated in the **Introduction** and **Methodology** sections in the revised version.

---

### **2. On the Novelty and Contributions**

**Reviewer Comment:**  
*“The novelty of the work is not clearly summarized. The authors are encouraged to clearly and explicitly highlight the key contributions and innovations of the paper, and to distinguish them from prior studies.”*

**Our Response:**  
We thank the reviewer for this suggestion. We will add a **clear summary of contributions** at the end of the Introduction in the revised manuscript. The key innovations of our work are:

- **Novel Multi-Scale Autoregressive Decoding**:  
  We reformulate autoregressive trajectory prediction from **next-step** to **next-scale**, enabling coarse-to-fine trajectory generation. This approach reduces error accumulation and improves long-term prediction accuracy, which is a departure from both classical autoregressive and non-autoregressive methods.
- **Dynamic Interaction-Aware Encoding**:  
  We introduce a **motion MoE module** and a **GAT-based interaction layer** to jointly model road user types, motion trends, traffic signals, and implicit interactions. This allows TrajAR to better handle complex multi-agent scenarios at intersections.
- **Strong Empirical Performance**:  
  Extensive experiments on four real-world intersection datasets show that TrajAR outperforms state-of-the-art methods by up to **32.0% in FDE** and **29.8% in APDE**, demonstrating its robustness and generalization ability.

These contributions are **distinct from prior work** such as KI-GAN (signal-aware but limited interaction modeling) and C2F-TP (diffusion-based but not multi-scale autoregressive). Our approach is the first to combine **multi-scale autoregressive decoding** with **dynamic interaction perception** for intersection trajectory prediction.

Of course. Here is the refined and comprehensive response to the first reviewer's comment, now incorporating the new, realistic comparative analysis between TrajAR and KI-GAN under different traffic conditions.

---

### **3. Additional Revisions**

In addition to the above, we will:
- Strengthen the **comparison with related work** in Section II to better highlight the gaps our method fills.
- Include a **summary of limitations of existing methods** and how TrajAR addresses them.
- Improve the **clarity and positioning** of the contribution statements.

---

We believe these revisions will significantly improve the clarity, motivation, and novelty presentation of the paper. We thank the reviewer again for their constructive comments, which have helped us strengthen the manuscript.
