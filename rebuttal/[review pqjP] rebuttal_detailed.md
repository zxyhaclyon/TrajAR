We sincerely thank the reviewer for your insightful comments and constructive feedback. We have carefully considered each point and provide our responses below.

---

### **[W1] On the Lack of a Clear Contribution Summary in the Introduction**

**Reviewer Comment:**  
*“The introduction lacks a clear and explicit summary of contributions, which makes it harder for readers to quickly grasp the novelty and main achievements. What is the research gap of existing techniques? How is the proposed method superior to the existing methods?”*

**Our Response:**  
We appreciate this valuable suggestion. To better highlight our contributions, we have now added an explicit summary of contributions at the end of the Introduction section. The main contributions of our work are as follows:

- **A novel multi-scale autoregressive trajectory prediction framework (TrajAR)** that reframes long-term prediction as a coarse-to-fine sequence generation task, mitigating error accumulation and improving long-horizon accuracy.
- **A dynamic interaction encoder** that integrates motion trends, road user types, traffic signals, and implicit interactions via a Motion Mixture-of-Experts (MoE) and Graph Attention Networks (GAT), enabling nuanced reasoning in complex intersection scenarios.
- **A multi-scale decoding strategy** that progressively refines trajectory predictions from coarse to fine resolutions, with cubic interpolation ensuring smooth and kinematically plausible trajectories.
- **Extensive experiments on four real-world urban intersection datasets** demonstrating state-of-the-art performance, with improvements of up to 32.0% in FDE and 29.8% in APDE over existing methods.

**Research Gap and Superiority:**  
Existing methods fall short in two key aspects:  
- **Interaction Modeling:** Prior works often use static graphs or simple pooling mechanisms, which fail to capture the dynamic and implicit interactions among diverse road users (e.g., vehicles yielding to pedestrians or turning vehicles).  
- **Long-Term Prediction:** Autoregressive models suffer from error accumulation, while non-autoregressive models introduce global bias and violate kinematic constraints.

**TrajAR** addresses these gaps by:  
- Dynamically modeling interactions via MoE and GAT, allowing the model to adapt to varying road user behaviors and traffic rules.  
- Using a multi-scale autoregressive decoder that reduces long-term error propagation through coarse-scale supervision and fine-scale refinement.

---

### **[W2] On the Distinction of Interaction Modeling (MoE + GAT)**

**Reviewer Comment:**  
*“The interaction modeling component (MoE combined with GAT) is not sufficiently distinguished from prior works, making the claimed innovation less convincing.”*

**Our Response:**  
We thank the reviewer for raising this point. Our interaction modeling approach is indeed a key innovation, and we clarify its novelty below:

- **Motion MoE with Specialized Experts:** Unlike standard MoE used in NLP or vision, our Motion MoE incorporates **indicator experts** (for discrete features like road user type and traffic signals) and **routed experts** (for continuous motion trends). This allows the model to jointly reason about semantic context and motion dynamics, which is novel in trajectory prediction.
- **GAT with Dynamic Edge Weights:** We augment GAT with a learnable Gaussian kernel to weight interactions based on spatial proximity, enabling the model to focus on relevant road users adaptively. This is a departure from static social pooling or fixed graph structures used in prior works (e.g., GRIP++, FJMP).
- **Integration with Transformer-Based Situation Awareness:** The outputs of MoE and GAT are fused via a Transformer encoder to capture temporal dependencies and global context, which is not present in prior GNN- or RNN-based interaction models.

In summary, our interaction module is **not merely a combination of existing components**, but a carefully designed **hybrid architecture** that captures both explicit and implicit interactions in a unified and interpretable manner.

---

### **[W3] On the Risk of Error Propagation in Coarse-to-Fine Prediction**

**Reviewer Comment:**  
*“The coarse-to-fine prediction strategy may inherit a potential risk that an early mistake could propagate through the refinement process. The paper would benefit from a discussion or analysis of such cases.”*

**Our Response:**  
We agree that error propagation is a valid concern in multi-stage prediction. However, our design incorporates several mechanisms to mitigate this risk:

- **Coarse-Scale Supervision:** The more critical (coarse-grained) nodes will undergo more supervised backpropagation to ensure the accurate prediction of coarse-grained nodes. This minimizes early error propagation, ensuring that the long-term trajectory backbone is accurate and stable.
- **Cubic Interpolation Constraints:** The use of cubic interpolation between scales ensures smooth transitions and preserves kinematic feasibility, reducing the impact of local errors.
- **Multi-Scale Training Loss:** The model is trained to minimize errors at **all scales simultaneously**, which regularizes the coarse-scale outputs and prevents early mistakes from amplifying.

**Empirical Evidence:**  
In our ablation study (Table 3), removing the multi-scale decoder led to significant performance degradation, especially in long-term metrics (e.g., FDE increased by over 100% in some cases). This indicates that our coarse-to-fine approach **reduces** rather than amplifies error accumulation compared to standard autoregressive or non-autoregressive baselines.