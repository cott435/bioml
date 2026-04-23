

def main(tdc=False):
    from pathlib import Path
    from data import SequenceProcessingPipeline
    import torch
    data_name = 'IEDB_Jespersen'
    model_name = 'esmc_300m'
    base_data_dir = Path(__file__).resolve().parents[1] / 'data' / 'data_files'

    if tdc:
        from data.parse import get_tdc_epitope
        raw_data = get_tdc_epitope(data_name, file_dir=base_data_dir)
    else:
        raw_data=None

    pipe = SequenceProcessingPipeline(
        data_name=data_name,
        model_name=model_name,
        sequence_kind="single",
        save_dir=base_data_dir,
    )
    device = 'cpu'# = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')

    result = pipe.run(
        df=raw_data,
        storage="lmdb",
        include_hidden_states=True,
        hidden_layers=[1, 5, 10, 15, 20, 25],
        device=device,
        esm3_tracks=("structure", "sasa")

    )

    train_ds = pipe.build_training_dataset(
        storage="lmdb",
        representation="concat",
        hidden_layers=[1, 10, 20],
    )

    d = train_ds[0]


if __name__ == "__main__":
    main()
