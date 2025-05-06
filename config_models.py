from pydantic import BaseModel, field_validator
from typing import Dict, List, Optional, Tuple, Union

class ReactionNode(BaseModel):
    smiles_list: List[str]

class ReferenceState(BaseModel):
    molecules: List[Tuple[str, int]]

class Config(BaseModel):
    model_path: str
    device: str
    dispersion: bool = True
    default_dtype: str = "float64"
    traj_dir: str
    filename: str
    host_mp_ids: List[str]
    miller_indices: List[Tuple[int, int, int]]
    surface_supercell: List[int] = [1, 1, 1]
    system_type: str
    alloy_elements: List[str]
    compositions: List[float]
    n_alloy_structures: int
    adsorbate_rotations: Dict[str, List[int]]
    add_promoters: bool = False
    promoters: Optional[List[str]] = None
    n_promoter_atoms: Optional[int] = None
    n_promoter_structures: Optional[int] = None
    reaction_network_nodes: Dict[str, Dict[str, List[str]]]
    reference_states: Dict[str, Dict[str, List[Tuple[str, int]]]]
    n_adsorption_attempts: int = 15
    min_site_distance: float = 3.0
    adsorption_distance: float = 2.0
    max_adsorption_distance: float = 3.5
    site_types: List[str] = ["ontop", "bridge", "hollow", "all"]

    @field_validator("system_type")
    def validate_system_type(cls, v):
        if v not in ["monometallic", "bimetallic"]:
            raise ValueError("system_type must be 'monometallic' or 'bimetallic'")
        return v
