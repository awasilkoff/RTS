import pickle
import gurobipy as gp
from pathlib import Path


def save_full_state(state: dict, model: gp.Model, base_path: str) -> None:
    base = Path(base_path)

    # 1. Save the model in a Gurobi-supported format
    model_path = base.with_suffix(".lp")  # or .mps, .rew, .json depending on needs
    model.write(str(model_path))

    # 2. Save the rest of the state via pickle, plus the model filename
    state_to_pickle = {**state, "_model_file": model_path.name}
    with open(base.with_suffix(".pkl"), "wb") as f:
        pickle.dump(state_to_pickle, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_full_state(base_path: str):
    base = Path(base_path)

    # 1. Load dict
    with open(base.with_suffix(".pkl"), "rb") as f:
        state = pickle.load(f)

    # 2. Recreate model from file
    model_file = state.pop("_model_file", None)
    model = None
    if model_file is not None:
        model = gp.read(str(base.with_name(model_file)))

    return state, model
