from screen_new import Calculator, Structure, Adsorption, InitResults
from config_models import Config
import torch
from pathlib import Path

CONFIG = Config(
    model_path="mace-small-density-agnesi-stress.model",
    device="cuda" if torch.cuda.is_available() else "cpu",
    dispersion=True,
    default_dtype="float64",
    traj_dir="/EO_mono_trajectories",
    filename="EO_mono_results.json",
    host_mp_ids=["mp-124"],
    miller_indices=[(1, 1, 1)],
    surface_supercell=[2, 2, 1],
    system_type="bimetallic",
    alloy_elements=["Pt"],
    compositions=[0.5],
    n_alloy_structures=3,
    adsorbate_rotations={"x": [90, 180]},
    add_promoters=True,
    promoters=["Ni"],
    n_promoter_atoms=1,
    n_promoter_structures=1,
    reaction_network_nodes={
        "ethylene epoxidation": {
            "ethylene + O": ["C=C", "[O]", "[O]"],
            "ethylene oxide metallocycle + O": ["[CH2]C[O]", "[O]"],
            "ethylene epoxide + O": ["C1CO1", "[O]"],
        }
    },
    reference_states={
        "ethylene epoxidation": {
            "ethylene + O": [("C=C", 1), ("O=O", 1)],
            "ethylene oxide metallocycle + O": [("C=C", 1), ("O=O", 1)],
            "ethylene epoxide + O": [("C=C", 1), ("O=O", 1)],
        }
    }
)

def main() -> None:
    calculator = Calculator.initialize(config=CONFIG)
    
    surface_structures = Structure.get_surface_structures(
        config=CONFIG,
        mp_ids=CONFIG.host_mp_ids,
        miller_indices=CONFIG.miller_indices,
        metal_type=CONFIG.system_type,
        dopants=CONFIG.alloy_elements,
        fractions=CONFIG.compositions,
        surface_supercell=CONFIG.surface_supercell,
        n_alloy_structures=CONFIG.n_alloy_structures,
        calculator=calculator,
        add_promoters=CONFIG.add_promoters,
        promoters=CONFIG.promoters,
        n_promoter_atoms=CONFIG.n_promoter_atoms,
        n_promoter_structures=CONFIG.n_promoter_structures,
        traj_dir=CONFIG.traj_dir,
    )

    adsorbate_library = Structure.create_adsorbates_from_smiles(
        CONFIG.reaction_network_nodes, config=CONFIG, calculator=calculator
    )

    adsorption_results = Adsorption.evaluate_adsorption(
        surface_structures, adsorbate_library, calculator, config=CONFIG
    )

    Calculator.save_results(InitResults(
        filename=Path(CONFIG.filename),
        data=adsorption_results,
        config=CONFIG.model_dump()
    ))

if __name__ == "__main__":
    main()
