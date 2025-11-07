import evaluate
import models

model = {
    "chunk_size": 10000,
    "small_window": 1
}

evaluate.run_evaluate_retrieval(config={
    "model": model
})
evaluate.run_evaluate_reply(config={
    "model": model
})
