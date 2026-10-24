# MLAfication: Bringing Multi-Head Latent Attention to Pretrained Models

**MLAfication** brings together our research on converting pretrained language and vision-language models to Multi-Head Latent Attention (MLA). The projects explore contribution-aware partial RoPE, low-rank KV compression, multimodal adaptation, and conversion of QKNorm-based dense and MoE models.

This is the continuation of **MHA2MLA**, now organized as one repository for three papers. Each project includes its own implementation, dependencies, and reproduction guide.

## 📦 Projects

| Project | Venue / status | Focus | Paper | Code |
| --- | --- | --- | --- | --- |
| **MHA2MLA** | **ACL 2025** | Convert pretrained MHA/GQA LLMs to MLA using partial RoPE and joint KV SVD. | [Proceedings](https://aclanthology.org/2025.acl-long.1597/) · [arXiv](https://arxiv.org/abs/2502.14837) | [MHA2MLA](MHA2MLA/) |
| **MHA2MLA-VLM** | **AAAI 2026** | Modality-adaptive partial RoPE and modality-decoupled SVD for LLaVA-1.5, LLaVA-NeXT, and Qwen2.5-VL. | [Proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/40319) · [arXiv](https://arxiv.org/abs/2601.11464) | [MHA2MLA-VLM](MHA2MLA-VLM/) |
| **MLAfication** | **Preprint (under review)** | Contribution-aware partial RoPE, activation-aware SVD initialization, and recovery training for standard MHA/GQA and QKNorm-based dense/MoE models. | [arXiv](https://arxiv.org/abs/2610.17202) | [MLAfication](MLAfication/) |

## 📰 News

- **[2026.10.24]** Added the [MLAfication preprint](https://arxiv.org/abs/2610.17202) and its citation.
- **[2026.10.13]** Consolidated the three projects into **MLAfication**, continuing the original MHA2MLA repository.
- **[2026.03.14]** MHA2MLA-VLM published in the **AAAI 2026** proceedings.
- **[2026.01.22]** Released MHA2MLA-VLM checkpoints derived from Qwen2.5-VL-7B.
- **[2025.07]** MHA2MLA published at **ACL 2025**.
- **[2025.06.13]** Released the refactored MHA2MLA implementation with Qwen support.

Earlier project updates are preserved in the individual READMEs.

## 🚀 Getting started

```bash
git clone https://github.com/JT-Ushio/MLAfication.git
cd MLAfication
```

Choose the project for the paper or workflow you want to reproduce:

```bash
# ACL 2025: original language-model conversion
cd MHA2MLA

# AAAI 2026: vision-language model conversion
# cd MHA2MLA-VLM

# MLAfication: standard MHA/GQA and QKNorm-based dense/MoE models
# cd MLAfication
```

Create a separate Python environment for each project and follow its README. Run project commands from that project's directory: configuration, data, rank-tensor, and output paths are relative to it. The ACL/AAAI implementations target an older stack than the MLAfication implementation; install the requirements for the selected project.

For MLAfication, start with [standard MHA/GQA conversion](MLAfication/mha_gqa/) or [QKNorm conversion](MLAfication/qknorm/). The [migration guide](docs/MIGRATION.md) maps the former repository layouts to their new locations.

## 🤗 Checkpoints

| Project | Released models |
| --- | --- |
| MHA2MLA | [MHA2MLA collection](https://huggingface.co/collections/fnlp/mha2mla-67c51287dfc6cd46127e1b92) — Llama-2 and SmolLM |
| MHA2MLA-VLM | [Qwen2.5-VL MLA checkpoints](https://huggingface.co/cnxup/models) |
| MLAfication | [MLAfication collection](https://huggingface.co/collections/Gnaudzzx/mlafication-6a91b14d4fb0e7dec2e1fc10) |

Model weights and datasets are downloaded separately. Existing small rank-selection tensors in the published projects are retained.

## 📚 Citation

Please cite the paper corresponding to the code you use. All entries are also available in [CITATIONS.bib](CITATIONS.bib).

```bibtex
@inproceedings{ji-etal-2025-towards,
  title = {Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs},
  author = {Ji, Tao and Guo, Bin and Wu, Yuanbin and Guo, Qipeng and Shen, Lixing and Chen, Zhan and Qiu, Xipeng and Zhang, Qi and Gui, Tao},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year = {2025},
  pages = {33313--33328},
  publisher = {Association for Computational Linguistics},
  doi = {10.18653/v1/2025.acl-long.1597},
  url = {https://aclanthology.org/2025.acl-long.1597/}
}

@inproceedings{fan2026mha2mlavlm,
  title = {MHA2MLA-VLM: Enabling DeepSeek's Economical Multi-Head Latent Attention Across Vision-Language Models},
  author = {Fan, Xiaoran and Sun, Zhichao and Ji, Tao and Shen, Lixing and Gui, Tao},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume = {40},
  number = {36},
  pages = {30638--30646},
  year = {2026},
  doi = {10.1609/aaai.v40i36.40319},
  url = {https://ojs.aaai.org/index.php/AAAI/article/view/40319}
}
```

**MLAfication manuscript:** *MLAfication: Efficiently Enabling High-Fidelity MLA for Modern LLMs* — Zile Wang, Zhide Xie, and Tao Ji. [arXiv:2610.17202](https://arxiv.org/abs/2610.17202). Under review.

```bibtex
@misc{wang2026mlafication,
  title = {MLAfication: Efficiently Enabling High-Fidelity MLA for Modern LLMs},
  author = {Wang, Zile and Xie, Zhide and Ji, Tao},
  year = {2026},
  eprint = {2610.17202},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2610.17202}
}
```

## License and provenance

Original code is licensed under [Apache-2.0](LICENSE), except where otherwise noted. Bundled and adapted code retains its original license and attribution; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the project-specific notices.

The source repositories and exact imported commits are recorded in [docs/SOURCES.json](docs/SOURCES.json). Paper manuscripts, figures, model weights, and datasets remain subject to their respective terms.

## ⭐ Star history

[![Star History Chart](https://api.star-history.com/chart?repos=JT-Ushio/MLAfication&type=Date)](https://www.star-history.com/#JT-Ushio/MLAfication&Date)
