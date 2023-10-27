# Question Generation with KET5


A Question Generation KET5 fine-tuned with [`SQUAD`](https://huggingface.co/datasets/squad) and [`Korquad 1.0`](https://huggingface.co/datasets/squad_kor_v1)

base-model: [KETI-AIR/ke-t5-base](https://huggingface.co/KETI-AIR/ke-t5-base)

## Process

1. Fine-tune `KET5` with `SQUAD` dataset
2. load checkpoint from the output of 1.
3. Fine-tune using `Korquad 1.0`

