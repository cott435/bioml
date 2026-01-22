data_dir = "./data_files"


def get_tdc_epitope(name, file_dir=data_dir):
    from tdc.single_pred import Epitope
    mapping = {'Antigen_ID': 'ID', 'Antigen': 'Sequence'}
    data = Epitope(name=name,  path=file_dir).get_data().rename(columns=mapping)
    data['ID'] = data['ID'].str.replace(" ", "")
    return data

def get_tdc_ppi(name='HuRI', split=False, neg_frac=None, file_dir=data_dir):
    from tdc.multi_pred import PPI
    data = PPI(name=name, path=file_dir)
    data = data.neg_sample(frac=neg_frac) if neg_frac else data
    return data.get_split() if split else data.get_data()

def get_tdc_protein_pep(name='brown_mdm2_ace2_12ca5', split=False, file_dir=data_dir):
    from tdc.multi_pred import ProteinPeptide
    data = ProteinPeptide(name=name, path=file_dir)
    return data.get_split() if split else data.get_data()

def get_tdc_epitope_binding(name='weber', split=False, file_dir=data_dir):
    from tdc.multi_pred import TCREpitopeBinding
    data = TCREpitopeBinding(name=name, path=file_dir)
    return data.get_split() if split else data.get_data()


