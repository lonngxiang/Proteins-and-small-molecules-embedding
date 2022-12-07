
from transformers import BertTokenizer, AutoTokenizer,BertModel
tokenizer_ = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")

model_ = BertModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM")


outputs_ = model_(**tokenizer_("Clc1c(Cl)c(O)ccc1O", return_tensors='pt'))

## molecule embedding

outputs_.pooler_output[0] ##向量，384维度
## or
outputs_mol.last_hidden_state[:,0,:][0].detach().numpy() 
