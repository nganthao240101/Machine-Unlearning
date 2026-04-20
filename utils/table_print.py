def print_metric_table_like_paper(result_book, dataset_name, ks=(10, 20, 50)):
    methods_order = ["retrain", "sisa", "receraser", "grapheraser"]
    models_order = ["bpr", "wmf", "lightgcn"]

    metric_map = {(r["model"], r["method"]): r for r in result_book.metric_rows if r["dataset"] == dataset_name}

    print("\n" + "=" * 150)
    print(f"{dataset_name:^150}")
    print("=" * 150)

    header = [""] + [m.upper() for _model in models_order for m in methods_order]
    print("".join(f"{h:>12}" for h in header))

    for metric_name in ["recall", "ndcg"]:
        for idx, k in enumerate(ks):
            row_name = f"{metric_name.capitalize()}@{k}"
            row = [row_name]

            for model in models_order:
                for method in methods_order:
                    rec = metric_map.get((model, method))
                    row.append("-" if rec is None else f"{rec[metric_name][idx]:.4f}")

            print("".join(f"{x:>12}" for x in row))

    print("=" * 150)


def print_time_table_like_paper(result_book, dataset_name):
    methods_order = ["retrain", "sisa", "receraser", "grapheraser"]
    models_order = ["bpr", "wmf", "lightgcn"]

    time_map = {(r["model"], r["method"]): r for r in result_book.time_rows if r["dataset"] == dataset_name}

    print("\n" + "=" * 150)
    print(f"{dataset_name:^150}")
    print("=" * 150)

    header = [""] + [m.upper() for _model in models_order for m in methods_order]
    print("".join(f"{h:>12}" for h in header))

    rows = [
        ("Retrain", "total_time"),
        ("ShardTrain", "shard_train_time"),
        ("AggTrain", "agg_train_time"),
        ("Total", "total_time"),
    ]

    for row_name, key in rows:
        row = [row_name]
        for model in models_order:
            for method in methods_order:
                rec = time_map.get((model, method))
                row.append("-" if rec is None or key not in rec else f"{rec[key]:.1f}s")
        print("".join(f"{x:>12}" for x in row))

    print("=" * 150)