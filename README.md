## Bidirectional Learning of Facial Action Units and Expressions via Structured Semantic Mapping across Heterogeneous Datasets

<img width="740" height="314" alt="image" src="https://github.com/user-attachments/assets/c50dc554-9010-4f09-b2f0-832198ce9cd6" />

### TL;DR


SSM presents a Structured Semantic Mapping (SSM) framework for bidirectional learning between Facial Action Units (AUs) and Facial Expressions (FEs) under heterogeneous datasets. Unlike prior one-way transfer (AU → FE), SSM enables mutual enhancement (AU ↔ FE) without requiring joint annotations, addressing inconsistencies in annotation granularity and data domains.  
🚧 This paper is currently under review. Code will be released upon acceptance.
￼

### 🔑 Key Ideas
- **Bidirectional Learning across Tasks**  
  Establishes reciprocal knowledge transfer between fine-grained AUs and coarse-grained expressions.
- **Textual Semantic Prototypes (TSP)**  
  Builds structured semantic anchors from textual descriptions with learnable prompts.
- **Dynamic Prior Mapping (DPM)**  
  Learns a bidirectional, data-driven association matrix guided by FACS priors for cross-task alignment.
- **Heterogeneous Joint Learning**  
  Enables training across datasets with different annotation formats (frame-level vs. clip-level).


### 🚀 Highlights
	•	First systematic study of AU ↔ FE bidirectional learning under heterogeneous supervision
	•	Achieves state-of-the-art performance on multiple AU and DFER benchmarks
	•	Demonstrates that expression semantics can improve AU detection, not just the reverse
	•	Strong cross-dataset generalization and zero-shot transfer ability  ￼


### 📊 Benchmarks for Experiments
	•	AU datasets: BP4D, DISFA
	•	DFER datasets: DFEW, FERV39K, MAFW

SSM consistently outperforms single-task and baseline models across diverse dataset combinations.  ￼

<img width="746" height="297" alt="image" src="https://github.com/user-attachments/assets/08695279-9768-45fb-bbb6-c6e6470892f2" />
<img width="742" height="315" alt="image" src="https://github.com/user-attachments/assets/5c03d81b-300d-4109-9872-4ad54455da42" />
<img width="740" height="453" alt="image" src="https://github.com/user-attachments/assets/7069e745-3202-456c-bf23-89e9a6581019" />




### 📎 Citation
```
@article{li2026bidirectional,
  title={Bidirectional Learning of Facial Action Units and Expressions via Structured Semantic Mapping across Heterogeneous Datasets},
  author={Li, Jia and Zhang, Yu and Chen, Yin and Hu, Zhenzhen and Li, Yong and Hong, Richang and Shan, Shiguang and Wang, Meng},
  journal={arXiv preprint arXiv:2604.10541},
  year={2026}
}
```


