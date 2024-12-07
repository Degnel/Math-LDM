import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from model import TransformerLM

model = TransformerLM()
model.load_checkpoint()
res = model.predict('1+2=')
print(res)