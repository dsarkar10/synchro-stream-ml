class TrafficController:
    DEFAULTS = {
        "buffered_linear": {
            "strategy": "Buffered Linear Ingestion",
            "safety": "high",
            "description": "Sequential processing with large memory buffer. Maximum safety against forgetting.",
            "plasticity": 0.25,
            "stability": 0.95,
            "throughput": 0.30,
        },
        "interleaved_minibatch": {
            "strategy": "Interleaved Mini-Batch",
            "safety": "medium",
            "description": "Interleaved old/new data in small batches. Balanced approach.",
            "plasticity": 0.55,
            "stability": 0.65,
            "throughput": 0.60,
        },
        "high_speed_parallel": {
            "strategy": "High-Speed Parallel",
            "safety": "low",
            "description": "Fully parallel batch ingestion. Maximum throughput, minimal retention.",
            "plasticity": 0.90,
            "stability": 0.25,
            "throughput": 0.95,
        },
    }

    def __init__(self):
        self.memory_buffer_size = 100
        self.buffer_resize_factor = 1.0

    def recommend(self, nps_score):
        if nps_score > 0.7:
            base = dict(self.DEFAULTS["buffered_linear"])
            base["buffer_resize"] = self._compute_buffer_resize(nps_score)
            return base
        elif 0.3 < nps_score <= 0.7:
            return dict(self.DEFAULTS["interleaved_minibatch"])
        else:
            return dict(self.DEFAULTS["high_speed_parallel"])

    def _compute_buffer_resize(self, nps_score):
        factor = 1.0 + (nps_score - 0.7) * 3.0
        factor = min(factor, 3.0)
        self.memory_buffer_size = int(100 * factor)
        self.buffer_resize_factor = factor
        return {
            "suggested_size": self.memory_buffer_size,
            "resize_factor": round(factor, 2),
            "message": f"Increased memory buffer by {round((factor - 1.0) * 100)}% to mitigate high NPS.",
        }

    def get_simulated_metrics(self, strategy_key):
        if strategy_key == "linear":
            return dict(self.DEFAULTS["buffered_linear"])
        elif strategy_key == "interleaved":
            return dict(self.DEFAULTS["interleaved_minibatch"])
        elif strategy_key == "parallel":
            return dict(self.DEFAULTS["high_speed_parallel"])
        return dict(self.DEFAULTS["interleaved_minibatch"])


controller = TrafficController()
