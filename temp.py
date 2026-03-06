from testing.analyze import ProteinEmbeddingAnalyzer
from pathlib import Path
from data import ESMCSingleDS

data_name = 'IEDB_Jespersen'
model_name = 'esmc_300m'
base_data_dir = Path.cwd() / 'data' / 'data_files'

dataset=ESMCSingleDS(data_name, model_name, save_dir=base_data_dir)

h5_path = 'path/to/your/embeddings.h5'
ids = ['prot1', 'prot2', 'prot3']  # Your list of IDs
labels = {'prot1': 1, 'prot2': 0, 'prot3': 1}  # Your labels dict
lengths = {'prot1': 100, 'prot2': 150, 'prot3': 120}  # Optional lengths dict

analyzer = ProteinEmbeddingAnalyzer(dataset)
analyzer.run_full_analysis(output_dir= Path.cwd() / 'data'/'analysis')