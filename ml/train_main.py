from pathlib import Path
from data import ESMCSingleDS
from ml.datasets import ESMCTokenFrameBuilder
from ml.pipeline import MLBaselinePipeline, default_model_specs
from ml.splits import GroupKFoldSplitStrategy, SingleGroupSplitStrategy
from ml.utils import set_global_seed


def main(test_name, max_tokens=200000, include_linear=True, include_nonlinear_svm=False, include_trees=False, rolling_ave=None) -> None:
    data_name = "IEDB_Jespersen"
    model_name = "esmc_300m"
    base_dir = Path(__file__).absolute().parents[1]
    base_data_dir = base_dir / "data" / "data_files"

    seed = 42
    use_k_fold = True
    n_splits = 4
    test_size = 0.2
    enable_grid_search = True
    save_token_frame = False


    output_dir = base_dir / "experiments" / "ml_baselines" / test_name
    output_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(seed)

    dataset = ESMCSingleDS(data_name, model_name, save_dir=base_data_dir)
    token_dataset = ESMCTokenFrameBuilder(dataset, seed=seed).build(max_tokens=max_tokens, rolling_ave=rolling_ave)

    split_strategy = (
        GroupKFoldSplitStrategy(n_splits=n_splits)
        if use_k_fold
        else SingleGroupSplitStrategy(test_size=test_size, seed=seed)
    )
    pipeline = MLBaselinePipeline(
        model_specs=default_model_specs(
            grid_search=enable_grid_search,
            include_nonlinear_svm=include_nonlinear_svm,
            include_trees=include_trees,
            include_linear=include_linear
        ),
        seed=seed,
    )
    results = pipeline.run(token_dataset, split_strategy)

    results_path = output_dir / "results.xlsx"
    results.to_excel(results_path, index=False)

    ok_results = results[results["status"] == "ok"].copy()
    summary = (
        ok_results.groupby(["model", "param_set"], as_index=False).mean(numeric_only=True)
        if not ok_results.empty
        else ok_results
    )
    summary_path = output_dir / "summary.xlsx"
    summary.to_excel(summary_path, index=False)

    if save_token_frame:
        token_dataset.frame.to_parquet(output_dir / "token_frame.parquet", index=False)

    print(f"Tokens: {token_dataset.num_tokens:,} | Features: {token_dataset.num_features}")
    print(f"Saved split metrics to: {results_path}")
    print(f"Saved model summary to: {summary_path}")
    print(summary)


if __name__ == "__main__":
    main('rolling_avv', rolling_ave=5)
