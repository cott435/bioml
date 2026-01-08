from data import get_tdc_ppi, SequenceEpitopeTokenizer
import esm
import torch

data = get_tdc_ppi('HuRI')

esm2, esm2_alphabet = esm.pretrained.esm2_t6_8M_UR50D()
tokenizer = SequenceEpitopeTokenizer(esm2_alphabet)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


