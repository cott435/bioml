data_dir = "./data_files"


def get_tdc_epitope(name, file_dir=data_dir):
    from tdc.single_pred import Epitope
    mapping = {'Antigen_ID': 'ID', 'Antigen': 'Sequence'}
    data = Epitope(name=name,  path=file_dir).get_data().rename(columns=mapping)
    data['ID'] = data['ID'].str.replace(" ", "")
    return data

def get_tdc_ppi(name, split=False, neg_frac=1, file_dir=data_dir):
    from tdc.multi_pred import PPI
    data = PPI(name=name, path=file_dir).neg_sample(frac=neg_frac)
    return data.get_split() if split else data.get_data()


