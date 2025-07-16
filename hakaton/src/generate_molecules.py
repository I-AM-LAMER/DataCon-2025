import os
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from reward_function import score_smiles

MODEL = "seyonec/REINVENT-2.0-RNN-PubChemFingerprint-v2"
OUTPUT = "results/generated.smi"
N_SAMPLES = 1000
BATCH_SIZE = 64

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL)

with open(OUTPUT, "w") as fout:
    for _ in range(N_SAMPLES // BATCH_SIZE):
        inputs = tokenizer([""]*BATCH_SIZE, return_tensors="pt", padding=True)
        outs = model.generate(**inputs, max_length=128)
        smiles_batch = tokenizer.batch_decode(outs, skip_special_tokens=True)
        for smi in smiles_batch:
            score = score_smiles(smi)
            fout.write(f"{smi}\t{score:.4f}\n")
print("Generated and scored molecules:", OUTPUT)
