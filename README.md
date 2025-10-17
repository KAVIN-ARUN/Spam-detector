The TinyBERT POC demonstrates SMS Spam Detection functionality.
Overview

The notebook serves as a Proof of Concept (POC) to show how TinyBERT works for SMS spam detection.
The model achieves three main objectives through this demonstration.

The UCI SMS Spam dataset undergoes fine-tuning operations using TinyBERT as the model.

The research compares TinyBERT embeddings to baseline models which include Random Forest and ANN for their performance evaluation.

The model receives evaluation through classification metrics and ROC-AUC performance assessment.


For experiment logging, set your *W&B API key*:

 bash
import wandb
wandb.login(key="YOUR_WANDB_API_KEY")


Citation

If you use this POC in research, please cite:

Dataset: Almeida, T.A., Gómez Hidalgo, J.M., & Yamakami, A. (2011). Contributions to the Study of SMS Spam Filtering. DocEng '11. DOI: 10.1145/2034691.2034742
.

TinyBERT: Jiao, X. et al. (2020). TinyBERT: Distilling BERT for Natural Language Understanding. Findings of EMNLP 2020. DOI: 10.18653/v1/2020.findings-emnlp.372
