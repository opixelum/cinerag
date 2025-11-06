import evaluate
import models

evaluate.run_evaluate_retrieval(config={
    "model": {"chunk_size": 10000, "small_window": 2}
})
evaluate.run_evaluate_reply(config={
    "model": {"chunk_size": 10000, "small_window": 2}
})
