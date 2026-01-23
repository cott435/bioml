from data.parse import *
from pathlib import Path
from proteins.data.datasets import MultiSequenceDS, SingleSequenceDS, ESMCMultiDS, ESMCSingleDS

base_data_dir = Path.cwd() / 'data' / 'data_files'


multi = ESMCMultiDS('HuRI', '')




single = get_tdc_epitope(file_dir=base_data_dir)
ppi_ds = SingleSequenceDS('IEDB_Jespersen', df=single, save_dir=base_data_dir)

ppi = get_tdc_ppi(file_dir=base_data_dir)
multi = MultiSequenceDS('HuRI', df=ppi, save_dir=base_data_dir)

aab = get_tdc_antibody_aff(file_dir=base_data_dir)
epp = get_tdc_epitope_binding(file_dir=base_data_dir)




