# A Small Language Model Based Approach using In-Context Learning for Natural Language to SQL

![Made with Python](https://img.shields.io/badge/Made%20with-Python-brightgreen) ![Maintained](https://img.shields.io/badge/Maintained%3F-yes-yellow) ![Open Source](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-red) ![Made with Pytorch](https://img.shields.io/badge/Made%20with-Transformers-orange)

## Abstract

**Abstract**—The transformation of natural language queries into SQL commands (NL-to-SQL) is a critical task for democratizing data access. While large language models (LLMs) have shown promising results, their deployment is often hindered by high computational costs and resource demands. This paper explores the potential of high-performance small language models (SLMs) as efficient alternatives. We propose a novel approach tailored for SLMs to address the NL-to-SQL task. Particularly, our approach leverages in-context learning (ICL) augmented by semantic retrieval to dynamically select relevant few-shot examples without the need for fine-tuning. Experimental results on the complex CORDIS scientific dataset demonstrate that the proposed method achieves a competitive execution accuracy of 41% with 3-shot learning. This performance surpasses several baselines, proving that SLMs combined with prompt engineering can effectively handle complex database schema logic on a single consumer-grade GPU. We share our experimental setup of this work on Github repository.
## Paper

(Link paper will be deployed)

## Database

The PostgreSQL databases for database used for this benchmark can be found at the following links: [CORDIS](https://drive.usercontent.google.com/download?id=1-YXF9mPN7GmK4ApBz2aZjAoJkdpPE6JF&export=download&authuser=0)

After downloading, extract (if needed) and place all files into data

## Citation

If you would like to cite this paper, please use the following reference:

```bibtex
@inproceedings{trang2025slmaproach,
  title={A Small Language Model Based Approach using In-Context Learning for Natural Language to SQL},
  author={Nguyen, Kieu-Trang and Le, Quang-Hung},
  year={2025}
}
