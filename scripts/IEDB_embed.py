

def main(tdc=False, max_block=None, embed=True):
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if embed:
        result = pipe.run(
            df=raw_data,
            storage="lmdb",
            include_hidden_states=True,
            hidden_layers=[1, 5, 10, 15, 20, 25],
            device=device,
            temperature=1e-6,
            max_block=max_block,
            embedder_kind='forge'
        )
    else:
        train_ds = pipe.build_training_dataset(
            storage="lmdb",
            representation="concat",
            hidden_layers=[1, 10, 20],
            include_structure=True,
            include_embedding=False
        )

        d = train_ds.plot(0)


if __name__ == "__main__":
    main(embed=False)
