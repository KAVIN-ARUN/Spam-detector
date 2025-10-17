#  Multimodal SMS OCR Spam Detection

## Overview

The system uses a hybrid SMS spam detection method which includes explainable AI (XAI) functionality.
The system integrates two spam filtering methods which include TF-IDF with Random Forest and TinyBERT fine-tuning for NLP classification.\
The system uses dynamic quantization (INT8) and ONNX export to achieve efficient deployment.
The system provides SHAP explainability through bar/force/token plots and similarity metrics.



------------------------------------------------------------------------



## Requirements

Install dependencies before running:

 bash
pip install torch transformers scikit-learn shap onnx onnxruntime matplotlib pandas


Use only  Google Colab:

 bash
!pip install "gradio>=4.44" "transformers>=4.43" "torch>=2.3" "shap>=0.45"              "pytesseract>=0.3.10" "Pillow>=9.5" "sqlalchemy>=2.0" "psycopg2-binary>=2.9"              "langdetect>=1.0.9" "pydantic>=2.8" "fastapi>=0.111" "uvicorn[standard]>=0.30"              "easyocr>=1.7.1" "opencv-python-headless>=4.10.0.84"
!apt-get -y install tesseract-ocr tesseract-ocr-eng >/dev/null


------------------------------------------------------------------------

## Dataset

-   📚 *UCI SMS Spam Collection* (5,574 messages; 747 spam).\
-   Notebook performs *80/20 stratified train/test split*.

------------------------------------------------------------------------

## Running the Notebook

1.  Clone or upload Multimodal_SMS_OCR_Spam_Detection.ipynb to Colab.\
2.  Install the dependencies (see above).\
3.  Run all cells step-by-step:
    -   Train baseline (TF-IDF + RF).\
    -   Fine-tune TinyBERT.\
    -   Evaluate on test set.\
    -   Apply quantization and export ONNX.\
    -   Generate SHAP explanations.

------------------------------------------------------------------------

## Outputs

-   *Classification metrics* (accuracy, precision, recall, F1,
    ROC-AUC).\
-   *ROC Curve plots*.\
-   *Quantized model*: tinybert_spam_int8.pt.\
-   *ONNX export*: tinybert_spam_float.onnx.\
-   *SHAP bar chart*: shap_summary_tokens.png.\
-   *SHAP-Sim score* between float and quantized models.

------------------------------------------------------------------------

## Weights & Biases (Optional)

For experiment logging, set your *W&B API key*:

 bash
import wandb
wandb.login(key="YOUR_WANDB_API_KEY")


------------------------------------------------------------------------

## Citation

UCI SMS Spam Collection Dataset :🔗 https://archive.ics.uci.edu/ml/datasets/sms+spam+collection

Almeida, T.A., Gómez Hidalgo, J.M., & Yamakami, A. (2011). Contributions to the Study of SMS Spam Filtering: New Collection and Results. Proceedings of the 11th ACM Symposium on Document Engineering (DocEng '11), 259–262. https://doi.org/10.1145/2034691.2034742

TinyBERT (HuggingFace)

Jiao, X., Yin, Y., Shang, L., Jiang, X., Chen, X., Li, L., Wang, F., & Liu, Q. (2020). TinyBERT: Distilling BERT for Natural Language Understanding. Findings of the Association for Computational Linguistics: EMNLP 2020, 4163–4174. https://doi.org/10.18653/v1/2020.findings-emnlp.372