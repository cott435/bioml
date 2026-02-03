from biotite.database.rcsb import SequenceQuery, search, count
from biotite.sequence import ProteinSequence, align
from biotite.application.muscle import Muscle5App
from typing import Iterable


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




