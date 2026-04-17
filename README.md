## Bidirectional Learning of Facial Action Units and Expressions via Structured Semantic Mapping across Heterogeneous Datasets

<img width="561" height="238" alt="image" src="https://github.com/user-attachments/assets/84ff5187-1533-4353-b582-99ee7759d158" />



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

<img width="451" height="182" alt="image" src="https://github.com/user-attachments/assets/968302f1-106d-4e3c-aa6b-bd67693f5895" />

<img width="659" height="269" alt="image" src="https://github.com/user-attachments/assets/0818da66-6fa2-44e4-8b19-51f1e2d67e65" />

<img width="551" height="336" alt="image" src="https://github.com/user-attachments/assets/623fdc58-b3a7-41de-9740-9ba57c5b0a17" />



### 📎 Citation
```
@article{li2026bidirectional,
  title={Bidirectional Learning of Facial Action Units and Expressions via Structured Semantic Mapping across Heterogeneous Datasets},
  author={Li, Jia and Zhang, Yu and Chen, Yin and Hu, Zhenzhen and Li, Yong and Hong, Richang and Shan, Shiguang and Wang, Meng},
  journal={arXiv preprint arXiv:2604.10541},
  year={2026}
}
```


