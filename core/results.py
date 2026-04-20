class ResultBook:
    def __init__(self):
        self.metric_rows = []
        self.time_rows = []

    def add_metric(self, dataset, model, method, metrics):
        self.metric_rows.append({
            "dataset": dataset,
            "model": model,
            "method": method,
            "recall": metrics["recall"],
            "precision": metrics["precision"],
            "ndcg": metrics["ndcg"],
        })

    def add_time(self, dataset, model, method, time_stats):
        row = {"dataset": dataset, "model": model, "method": method}
        row.update(time_stats)
        self.time_rows.append(row)