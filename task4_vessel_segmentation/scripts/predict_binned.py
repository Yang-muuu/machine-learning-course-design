"""Equivalent histogram-tree inference on its fitted bins for this sklearn build."""
import numpy as np


def predict_proba_binned(model, x, threads=6):
    checked = model._preprocess_X(x, reset=False)
    binned = model._bin_mapper.transform(checked)
    raw = np.zeros((len(x), model.n_trees_per_iteration_), dtype=model._baseline_prediction.dtype, order="F")
    raw += model._baseline_prediction
    model._predict_iterations(binned, model._predictors, raw, True, threads)
    return model._loss.predict_proba(raw)
