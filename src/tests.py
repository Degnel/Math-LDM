import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from model import TransformerLM

model = TransformerLM()
model.load_checkpoint()
# res = model.predict("1+2=")
res = model.predict("7/8.13=0.9")
print(res)
