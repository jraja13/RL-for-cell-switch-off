"""
Config Generator: 
Reads cells.csv and outputs network_config.json
cells.csv columns: CellID, Latitude, Longitude, Macro

Usage:
run the command: python generate_config.py
"""

import json
import pandas as pd

# Auer et al. (2011) power parameters 
POWER_PARAMS = {
    "macro": {"p_max_watts": 40.0, "p_static_watts": 10.0},
    "micro": {"p_max_watts": 6.4,  "p_static_watts": 2.0},
}

def generate_config(cells_csv_path: str = "Datasets/Opencellid/final_topology.csv",
                    output_path: str = "network_config.json"):

    df = pd.read_csv(cells_csv_path)
    cells = []
    for _, row in df.iterrows():
        cell_id   = str(row["CellID"])
        cell_type = "macro" if cell_id.startswith("M") else "micro"
        p         = POWER_PARAMS[cell_type]

        cells.append({
            "cell_id":          cell_id,
            "cell_type":        cell_type,
            "macro_id":         str(row["Macro"]),
            "latitude":         float(row["Latitude"]),
            "longitude":        float(row["Longitude"]),
            "p_max_watts":      p["p_max_watts"],
            "p_static_watts":   p["p_static_watts"],
            "cso_candidate":    cell_type == "micro",
        })

    config = {"cells": cells}

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    n_macro = sum(1 for c in cells if c["cell_type"] == "macro")
    n_micro = sum(1 for c in cells if c["cell_type"] == "micro")

    print(f"Config saved → {output_path}")
    print(f"  Macro cells : {n_macro}")
    print(f"  Micro cells : {n_micro}  (CSO candidates)")

    return config

if __name__ == "__main__":
    generate_config()