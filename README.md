# LAB2: Fine-Tuning LLM

This repository contains the work for **LAB2: Fine-Tuning LLM**, focusing on fine-tuning large language models (LLMs) and evaluating their performance. All related work is available in the included notebooks, and you can access the Hugging Face interface for testing the fine-tuned model at the following link:  
[Hugging Face Space](https://huggingface.co/spaces/PierreJousselin/LAB_LLM)
[Gradio in Google Colab](https://colab.research.google.com/drive/1OdHS-It5L6eRE0fIRWW6IcGvjgmIX2X9?usp=sharing)

## Evaluation Metrics

We use **BERTScore** to evaluate the semantic similarity between generated text and reference text. For tasks like translation, metrics such as **BLEU score** are more suitable, as they measure sequence overlap between reference and generated outputs. The choice of metric should align with the task, as each metric emphasizes different aspects of model performance.

## Fine-Tuning Methodologies

**Model-centric approaches** aim to improve the model itself. For example, we think that we can tune hyperparameters like the learning rate, batch size, and optimizer settings to achieve better convergence during fine-tuning. Also we can  experiment with different architectures or models if we want to do some more special task. We used Lora adapter which consist in freezing most parts of the model and train only a subset of parameters. Therefore there is some experimentation to do on which subset to train in order to have the best result.

**Data-centric approaches** focus on improving the quality and relevance of the dataset used for training. For instance, we wanted to generate Python code, so we  fine-tuned the model on a dataset containing annotated Python scripts. Another example could be if we wanted to train or improve translation accuracy, then we would fine-tune on bilingual corpora like the WMT dataset. There are also some tweeking like data augmentation methods, such as paraphrasing sentences or adding domain-specific examples that can also enhance the model's ability to generalize. Also we needed to be careful and to clean the data as the model is trained using a certain template that we need to keep in order to be coherent in our approach.

## Models
As the inference was made on cpu, we had to restrict ourself to small model. Even with only 2.5Gb we can have inference that last 300s which is very long. We really saw in practice that GPUs are very very usefull and almost mandatory when using large language model.
