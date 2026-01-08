from tdc.multi_pred import PPI, PeptideMHC, ProteinPeptide, AntibodyAff

file_dir="./raw_data_files"

ppi = PPI(name="HuRI", path=file_dir).neg_sample(frac=1).get_data()

peptide_mhc = PeptideMHC(name="MHC1_IEDB-IMGT_Nielsen", path=file_dir).get_data()
protein_pep = ProteinPeptide(name="brown_mdm2_ace2_12ca5", path=file_dir).get_data()
antibodyaff = AntibodyAff(name="Protein_SAbDab", path=file_dir).get_data()



