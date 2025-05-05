import json
from typing import Dict, List, Any
from genome import EnergyGene, MoveGene, Gene


def load_genes_from_file(
        filename: str = "gene_settings.json") -> Dict[str, Gene]:
    """
    Loads gene settings from a JSON file (as shown below) and returns
    a dict mapping each gene name to the appropriate Gene object:

    {
        "size": [1.0, 4.0],
        "speed": [1.0, 5.0],
        ...
        "energy_prod": {
            "options": ["heterotroph", "autotroph", "parasite"],
            "values":  [0.1,         0.5,         1.0       ]
        },
        "move_aff": ["aquatic", "volant", "terrestrial"]
    }
    """
    gene_pool: Dict[str, Gene] = {}

    # Open and parse the JSON - Context handler will close file
    with open(filename, "r") as f:
        genes_data: Dict[str, Any] = json.load(f)

    for gene_name, settings in genes_data.items():
        if gene_name == "energy_prod":
            options: List[str] = settings["options"]
            values: List[float] = settings["values"]
            # EnergyGene(default_option, mid, min, max, all_options)
            gene_pool[gene_name] = EnergyGene(
                options[0],
                values[1],
                values[0],
                values[2],
                options
            )

        elif gene_name == "move_aff":
            options: List[str] = settings  # just a list of affinities
            # MoveGene(default_affinity, all_affinities)
            gene_pool[gene_name] = MoveGene(
                options[0],
                options
            )

        else:
            # Generic genes are two-element [min, max] lists
            try:
                min_val, max_val = settings  # type: ignore
            except (TypeError, ValueError):
                raise ValueError(
                    f"Expected two floats for '{gene_name}', got {settings!r}"
                    )
            default_val: float = (min_val + max_val) / 2.0
            # Gene(name, default, min, max)
            gene_pool[gene_name] = Gene(
                gene_name,
                default_val,
                min_val,
                max_val
            )

    return gene_pool