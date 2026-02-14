from biotite.database.rcsb import SequenceQuery, search, count
from biotite.sequence import ProteinSequence, align
from biotite.application.muscle import Muscle5App
from typing import Iterable
import numpy as np
import matplotlib.pyplot as plt
import py3Dmol
import biotite.structure as struc
import biotite.structure.io as struc_io
import biotite.database.rcsb as rcsb
import biotite.sequence as seq
from biotite.structure.io import pdbx


def find_similar_sequences(sequence, scope='protein', min_identity=0.9, max_expect_value = 1e-5):
    seq_query = SequenceQuery(sequence, scope=scope, min_identity=min_identity, max_expect_value=max_expect_value)
    similar_ids = search(seq_query)
    return similar_ids

def align_sequences(sequences: Iterable[ProteinSequence|str]) -> align.Alignment:
    sequences = [ProteinSequence(seq) if isinstance(seq, str) else seq for seq in sequences]
    app = Muscle5App(sequences)
    app.start()
    app.join()
    return app.get_alignment()


class ProteinAnalysis:
    def __init__(self, input_data, mode="id"):
        """
        mode: 'id' (PDB ID) or 'sequence' (Amino Acid String)
        """
        self.query_sequence = None
        self.pdb_id = None
        self.atom_array = None  # The full structure
        self.target_chain = None  # The specific chain matching our sequence
        self.stats = {}  # To store identity, SASA, etc.

        if mode == "id":
            self.pdb_id = input_data
            self._load_from_id(self.pdb_id)
            # If loaded by ID, we assume the first chain is the target
            # unless specified otherwise, but for now we take the whole model.
            self.target_chain = self.atom_array

        elif mode == "sequence":
            self.query_sequence = seq.ProteinSequence(input_data)
            self._search_and_load()

    def _load_from_id(self, pdb_id):
        print(f"Fetching PDB ID: {pdb_id}...")
        file_path = rcsb.fetch(pdb_id, "cif", target_path="")
        pdbx_file = pdbx.BinaryCIFFile.read(file_path)
        # Get structure (model 1)
        self.atom_array = pdbx.get_structure(pdbx_file, model=1)
        # Filter for peptide backbone only (remove water/ligands)
        self.atom_array = self.atom_array[struc.filter_amino_acids(self.atom_array)]

    def _search_and_load(self):
        print("Searching PDB for sequence match...")
        # Search for 90%+ identity matches
        query = rcsb.SequenceQuery(str(self.query_sequence), min_identity=0.9)
        results = rcsb.search(query)

        if not results:
            raise ValueError("No matches found in PDB.")

        best_id = results[0]  # Take top hit
        print(f"Match found: {best_id}")
        self.pdb_id = best_id
        self._load_from_id(best_id)

        # CRITICAL: Isolate the correct chain
        self._isolate_matching_chain()

    def _isolate_matching_chain(self):
        """
        Splits the PDB into chains and aligns the query sequence to each.
        Keeps only the chain with the highest identity.
        """
        print("Isolating correct chain...")
        chain_ids = np.unique(self.target_chain.chain_id)
        best_chain = None
        best_score = -1
        best_identity = 0

        matrix = align.SubstitutionMatrix.std_protein_matrix()

        for ch_id in chain_ids:
            # Extract chain
            current_chain = self.atom_array[self.atom_array.chain_id == ch_id]

            # Convert chain structure to sequence
            # (Note: to_sequence returns a list of seqs, we take [0])
            chain_seq = struc.to_sequence(current_chain)[0]

            # Align
            alignment = align.align_optimal(self.query_sequence, chain_seq, matrix, local=False)[0]

            # Calculate Identity
            identity = align.get_sequence_identity(alignment)

            if identity > best_score:
                best_score = identity
                best_chain = current_chain
                best_identity = identity

        print(f"Selected Chain with {best_identity * 100:.1f}% Identity")
        self.target_chain = best_chain
        self.stats['identity'] = best_identity

    def calculate_properties(self):
        """Calculates SASA, SSE, and Torsion angles."""
        if self.target_chain is None:
            raise ValueError("No structure loaded.")

        print("Calculating structural properties...")

        # 1. SASA (Residue-wise)
        # Calculate atom SASA then sum by residue
        atom_sasa = struc.sasa(self.target_chain, point_number=100)
        self.stats['sasa'] = struc.apply_residue_wise(
            self.target_chain, atom_sasa, np.sum
        )

        # 2. Secondary Structure (SSE)
        # Encoded as 'a', 'b', 'c'
        self.stats['sse'] = struc.annotate_sse(self.target_chain)

        # 3. Torsion Angles (Phi, Psi)
        # Returns tuple (phi, psi, omega)
        phi, psi, omega = struc.dihedral_backbone(self.target_chain)
        self.stats['phi'] = np.degrees(phi)  # Convert to degrees for plotting
        self.stats['psi'] = np.degrees(psi)

    def get_pdb_string(self):
        """Helper to get PDB string for the visualizer."""
        # Create a temporary file object in memory is complex,
        # easiest is to save to string using specialized IO or tempfile.
        # Here we rely on saving a temp file for 3Dmol compatibility.
        sink = pdbx.BinaryCIFFile()
        structure = pdbx.set_structure(sink, self.target_chain)
        return structure


# ---------------------------------------------------------

class ProteinVisualizer:
    def __init__(self, analysis_obj):
        self.data = analysis_obj

    def plot_properties(self):
        """Plots SASA and Torsion angles using Matplotlib."""
        if 'sasa' not in self.data.stats:
            print("Please run .calculate_properties() first.")
            return

        sasa = self.data.stats['sasa']
        phi = self.data.stats['phi']
        psi = self.data.stats['psi']
        residues = np.arange(len(sasa))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Plot SASA
        ax1.plot(residues, sasa, color='orange', label='SASA')
        ax1.fill_between(residues, sasa, alpha=0.3, color='orange')
        ax1.set_ylabel('SASA ($\AA^2$)')
        ax1.set_title(f'Structural Properties: {self.data.pdb_id}')
        ax1.grid(True, alpha=0.3)

        # Plot Torsion
        ax2.plot(residues, phi, label='Phi', alpha=0.7)
        ax2.plot(residues, psi, label='Psi', alpha=0.7)
        ax2.set_ylabel('Angle (Degrees)')
        ax2.set_xlabel('Residue Index')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def view_3d(self, style='cartoon'):
        """
        Renders interactive 3D view using py3Dmol.
        style: 'cartoon', 'stick', 'sphere'
        """
        # We need to write the structure to a string/file that py3Dmol accepts.
        # For simplicity, we save a temp PDB file.
        temp_filename = "temp_view.pdb"
        pdb_file = struc_io.PDBFile()
        pdb_file.set_structure(self.data.target_chain)
        pdb_file.write(temp_filename)

        # View Setup
        view = py3Dmol.view(width=600, height=400)
        view.addModel(open(temp_filename, 'r').read(), 'pdb')

        # Styling
        if style == 'cartoon':
            view.setStyle({'cartoon': {'color': 'spectrum'}})
        elif style == 'stick':
            view.setStyle({'stick': {}})

        view.zoomTo()
        view.show()


