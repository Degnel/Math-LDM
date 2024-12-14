import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from model import TransformerLM

mode = "ldm"

if mode == "ldm":
    model = TransformerLM(max_steps=100)
elif mode == "llm":
    model = TransformerLM()

model.load_checkpoint(mode)
# res = model.predict("1+2=")
res = model.predict("7/8.13=0.9")
print(res)