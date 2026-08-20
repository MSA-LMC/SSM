# Bidirectional Learning of Facial Action Units and Expressions via Structured Semantic Mapping across Heterogeneous Datasets

[Link to IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/11653010), [Link to Arxiv](https://arxiv.org/abs/2604.10541)

<img width="561" height="238" alt="SSM framework overview" src="https://github.com/user-attachments/assets/84ff5187-1533-4353-b582-99ee7759d158" />

## 📌 TL;DR

SSM presents a Structured Semantic Mapping (SSM) framework for bidirectional learning between Facial Action Units (AUs) and Facial Expressions (FEs) under heterogeneous datasets. Unlike prior one-way transfer (AU → FE), SSM enables mutual enhancement (AU ↔ FE) without requiring joint annotations, addressing inconsistencies in annotation granularity and data domains.

🚧 2026.07.31 This code has been released and will continue to be updated and improved...

🎉 2026.08.04 This paper has been accepted by IEEE Transactions on Affective Computing (TAFFC).


## 🔑 Key Ideas

- **Bidirectional Learning across Tasks**
  Establishes reciprocal knowledge transfer between fine-grained AUs and coarse-grained expressions.
- **Textual Semantic Prototypes (TSP)**
  Builds structured semantic anchors from textual descriptions with learnable prompts.
- **Dynamic Prior Mapping (DPM)**
  Learns a bidirectional, data-driven association matrix guided by FACS priors for cross-task alignment.
- **Heterogeneous Joint Learning**
  Enables training across datasets with different annotation formats (frame-level vs. clip-level).

## 🌟 Highlights

- First systematic study of AU ↔ FE bidirectional learning under heterogeneous supervision
- Achieves state-of-the-art performance on multiple AU and DFER benchmarks
- Demonstrates that expression semantics can improve AU detection, not just the reverse
- Strong cross-dataset generalization and zero-shot transfer ability

## 📊 Benchmarks

- AU datasets: BP4D, DISFA
- DFER datasets: DFEW, FERV39K, MAFW

SSM consistently outperforms single-task and baseline models across diverse dataset combinations.

<img width="451" height="182" alt="SSM benchmark summary" src="https://github.com/user-attachments/assets/968302f1-106d-4e3c-aa6b-bd67693f5895" />

<img width="659" height="269" alt="SSM comparison results" src="https://github.com/user-attachments/assets/0818da66-6fa2-44e4-8b19-51f1e2d67e65" />

<img width="551" height="336" alt="SSM ablation results" src="https://github.com/user-attachments/assets/623fdc58-b3a7-41de-9740-9ba57c5b0a17" />

## 🛠️ Installation

```bash
conda env create -f environment.yml
conda activate ssm
```

## 📂 Data

Prepare the datasets following [docs/DATA.md](docs/DATA.md). The experiment split files are included in `splits/`.

## 🏋️ Training

The six BP4D/DISFA and DFEW/FERV39K/MAFW combinations are configured in `configs/`.

```bash
python train.py --config configs/bp4d_dfew.json
```

Each run jointly trains the expression and AU branches. Use `--emotion-fold` and `--au-fold` for one fold pair, or `--all-folds` for all configured pairs. The provided configurations use a three-GPU DataParallel setup.

## ⚡ Checkpoints

Representative fine-tuned weights are provided through [Google Drive](https://drive.google.com/drive/folders/16r6gvPjMV0anKe9pGWYk6yI4QrH5EIOf?usp=drive_link), and additional weights will be uploaded continuously. Each released checkpoint contains the complete joint model and evaluates both tasks.

## 🧪 Evaluation

```bash
python evaluate.py \
  --config configs/bp4d_dfew.json \
  --checkpoint /path/to/best.pth \
  --emotion-fold 5 \
  --au-fold 1
```

One evaluation command reports both expression (`uar`, `war`) and AU (`f1`, `auc`) metrics. The configuration and fold pair must match the checkpoint.

## 📎 Citation

```bibtex
@ARTICLE{11653010,
  author={Li, Jia and Zhang, Yu and Chen, Yin and Hu, Zhenzhen and Li, Yong and Hong, Richang and Shan, Shiguang and Wang, Meng},
  journal={IEEE Transactions on Affective Computing}, 
  title={Bidirectional Learning of Facial Action Units and Expressions Via Structured Semantic Mapping Across Heterogeneous Datasets}, 
  year={2026},
  volume={},
  number={},
  pages={1-16},
  doi={10.1109/TAFFC.2026.3722867}}

@ARTICLE{11207542,
  author={Chen, Yin and Li, Jia and Zhang, Yu and Hu, Zhenzhen and Shan, Shiguang and Wang, Meng and Hong, Richang},
  journal={IEEE Transactions on Affective Computing}, 
  title={Static for Dynamic: Towards a Deeper Understanding of Dynamic Facial Expressions Using Static Expression Data}, 
  year={2026},
  volume={17},
  number={1},
  pages={438-451},
}

@ARTICLE{10663980,
  author={Chen, Yin and Li, Jia and Shan, Shiguang and Wang, Meng and Hong, Richang},
  journal={IEEE Transactions on Affective Computing}, 
  title={From Static to Dynamic: Adapting Landmark-Aware Image Models for Facial Expression Recognition in Videos}, 
  year={2025},
  volume={16},
  number={2},
  pages={624-638}}
```


## 📬 Contact

If you have any questions, please contact yuz@mail.hfut.edu.cn.

## 🙏 Acknowledgments

This repository builds on [DFER-CLIP](https://github.com/zengqunzhao/DFER-CLIP), [OpenAI CLIP](https://github.com/openai/CLIP), and [CoOp](https://github.com/KaiyangZhou/CoOp). We sincerely thank them for their open-source contributions.
