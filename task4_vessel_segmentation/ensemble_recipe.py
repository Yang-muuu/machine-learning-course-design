"""Preserve the float32 blend operations when an OOF ensemble is warm-started."""
import itertools
import numpy as np


def replay_recipe(recipe, probabilities):
    p = probabilities[recipe["initial_source"]].copy()
    for step in recipe["steps"]:
        p = p * np.float32(1 - step["alpha"]) + probabilities[step["source"]] * np.float32(step["alpha"])
    return p.astype(np.float32)


def recover_recipe(report, probabilities, expected):
    recipe = report.get("blend_recipe")
    if recipe is not None and np.array_equal(replay_recipe(recipe, probabilities), expected):
        return recipe
    history = report.get("search_history", [])
    if history and history[0]["method"] == "best_single":
        recipe = {"initial_source": history[0]["source"], "steps": [
            {"source": step["added"], "alpha": step["alpha"]} for step in history[1:]
            if step["method"] in ("global_convex_blend", "greedy_convex_blend")]}
        if np.array_equal(replay_recipe(recipe, probabilities), expected):
            return recipe
    # Early primary reports retained weights but lost the three search steps on
    # warm start. Recover only within the actual declared three-round grid;
    # acceptance requires equality at every pixel, and never uses labels.
    members = report["members"]
    if len(members) > 4:
        raise ValueError("Historical blend needs its original operation recipe")
    names = [member["name"] for member in members]
    target = np.array([member["weight"] for member in members], dtype=np.float64)
    operations = list(itertools.product(range(len(names)), [.1, .25, .5, .75]))
    for depth in range(4):
        for start in range(len(names)):
            for steps in itertools.product(operations, repeat=depth):
                weights = np.zeros(len(names), dtype=np.float64)
                weights[start] = 1.
                for index, alpha in steps:
                    weights *= 1 - alpha
                    weights[index] += alpha
                if not np.allclose(weights, target, atol=1e-12, rtol=0):
                    continue
                recipe = {"initial_source": names[start], "steps": [
                    {"source": names[index], "alpha": alpha} for index, alpha in steps]}
                if np.array_equal(replay_recipe(recipe, probabilities), expected):
                    return recipe
    raise ValueError("No declared blend recipe exactly reconstructs historical probabilities")
