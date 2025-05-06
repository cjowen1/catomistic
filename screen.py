from config_models import Config
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import json
import logging
import os
import random
import re

import numpy as np
from ase import Atoms
from ase.constraints import FixBondLengths
from ase.filters import FrechetCellFilter
from ase.io import Trajectory
from ase.optimize import FIRE
from pydantic import BaseModel
from fairchem.data.oc.core import Bulk, Slab
from fairchem.data.oc.databases.pkls import BULK_PKL_PATH
from mace.calculators import MACECalculator
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.io.ase import AseAtomsAdaptor
from rdkit import Chem
from rdkit.Chem import AllChem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class RelaxationParams(BaseModel):
    fmax: float = 0.05
    steps: int = 500
    surface_only: bool = False

class InitResults(BaseModel):
    filename: Path
    data: Dict
    config: Dict[str, Any]  

    def model_dump(self) -> Dict:
        return {
            "filename": str(self.filename),
            "data": self.data,
            "config": self.config.model_dump() 
        }

class Calculator:
    @staticmethod
    def initialize(config: Config) -> MACECalculator:
        try:
            calculator = MACECalculator(
                model_paths=config.model_path,
                device=config.device, 
                dispersion=config.dispersion,
                default_dtype=config.default_dtype,
            )
            logging.info("Initialized calculator successfully")
            return calculator
        except Exception as e:
            logging.error(f"Error initializing calculator: {e}")
            raise

    @staticmethod
    def save_results(results: InitResults) -> None:
        """Save calculation results to file."""
        if results.config is None:
            raise ValueError("config must be provided")

        serializable_results = {}
        for surf_key, rxn_data in results.data.items():
            serializable_results[surf_key] = {}
            base_parts = []
            promoter_info = None

            # Process promoter information
            for part in surf_key.split("_"):
                if any(p in part for p in results.config["promoters"]):
                    promoter_match = re.match(r"([A-Z][a-z]?)(\d+\.\d+)", part)
                    if promoter_match:
                        element, coverage = promoter_match.groups()
                        promoter_info = {
                            "element": element,
                            "coverage": float(coverage),
                            "n_atoms": results.config["n_promoter_atoms"],
                        }
                else:
                    base_parts.append(part)

            # Process reaction data
            for node_key, node_data in rxn_data.items():
                serializable_results[surf_key][node_key] = {
                    "configurations": [],
                    "promoter_info": promoter_info,
                }

                for config in node_data["configurations"]:
                    config_copy = config.copy()
                    if "site_info" in config_copy:
                        for site in config_copy["site_info"]:
                            if "coordinates" in site and not isinstance(
                                site["coordinates"], list
                            ):
                                site["coordinates"] = site["coordinates"].tolist()
                    serializable_results[surf_key][node_key]["configurations"].append(
                        config_copy
                    )

        with open(results.filename, "w") as f:
            json.dump(serializable_results, f, indent=2)
        logging.info(f"Results saved to {results.filename}")

class StructureConfig(BaseModel):
    surface_name: str
    miller_indices: List[Tuple[int, int, int]]
    surface_supercell: List[int]
    metal_type: str
    dopants: Optional[List[str]] = None
    fractions: Optional[List[float]] = None
    n_alloy_structures: Optional[int] = None
    add_promoters: bool = False
    promoters: Optional[List[str]] = None 
    n_promoter_atoms: Optional[int] = None
    n_promoter_structures: Optional[int] = None
    traj_dir: Path

class SurfaceStructure(BaseModel):
    name: str
    atoms: Atoms
    miller_indices: Optional[Tuple[int, int, int]] = None
    dopants: Optional[List[str]] = None
    fraction: Optional[float] = None

    class Config:
        arbitrary_types_allowed = True

class Structure:
    @staticmethod
    def _handle_molecular_species(smiles: str, config: Config, calculator: MACECalculator) -> Dict:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        
        conformer = mol.GetConformer()
        positions = conformer.GetPositions()
        
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        atoms = Atoms(symbols, positions=positions)
        atoms.center(vacuum=15.0)
        
        relaxed = Structure.relax_structure(
            atoms,
            calculator=calculator,
            trajectory=f"mol_{Structure.sanitize_filename(smiles)}_relaxation.traj",
            traj_dir=config.traj_dir
        )
        
        return {
            "smiles": smiles,
            "conformers": [relaxed],
            "surface_constraints": [],
            "relaxed_bond_lengths": [],
            "multiplicity": 1,
        }   
    
    @staticmethod
    def relax_structure(
        atoms: Atoms,
        name: Optional[str] = None,
        calculator: Optional[MACECalculator] = None,
        surface_only_relax: bool = False,
        trajectory: Optional[str] = None,
        traj_dir: Optional[Union[Path, str]] = None,
    ) -> Optional[Atoms]:
        if not calculator:
            raise ValueError("Calculator must be provided")
        if not traj_dir:
            raise ValueError("traj_dir must be provided")

        # Convert traj_dir to Path object
        traj_dir = Path(traj_dir)
        
        atoms = atoms.copy()
        atoms.calc = calculator
        
        os.makedirs(traj_dir, exist_ok=True)
        logging.info(f"Using trajectory directory: {traj_dir}")

        traj = None
        if trajectory:
            traj_path = traj_dir / trajectory
            traj = Trajectory(str(traj_path), "a")

        try:
            if surface_only_relax:
                exp_cell_filter = FrechetCellFilter(
                    atoms, 
                    mask=[True, True, False, False, False, False]
                )
                optimizer = FIRE(exp_cell_filter, trajectory=traj)
            else:
                optimizer = FIRE(atoms, trajectory=traj)
                
            optimizer.run(fmax=0.05, steps=500)
            
            if traj:
                traj.close()
            return atoms

        except Exception as e:
            logging.error(f"Error relaxing structure: {str(e)}")
            return None

    @staticmethod  
    def add_promoters_to_surface(
        base_surface_name: str,
        base_surface_atoms: Atoms,
        promoters: List[str],
        num_atoms: int,
        n_promoter_structures: int = 1,
        calculator: Optional[MACECalculator] = None
    ) -> List[Tuple[str, Atoms]]:
        """Add promoter atoms to surface sites using energy-based selection."""
        promoted_surfaces = []

        for promoter in promoters:
            trial_structures = []
            n_trials = n_promoter_structures * 10  # Try 10x more placements than needed
            
            for struct_idx in range(n_trials):
                try:
                    promoted_atoms = base_surface_atoms.copy()
                    pmg_structure = AseAtomsAdaptor.get_structure(promoted_atoms)
                    asf = AdsorbateSiteFinder(pmg_structure)
                    ads_sites = asf.find_adsorption_sites(distance=2.0)

                    sites = ads_sites.get("hollow", [])
                    if not sites:
                        surface_atoms = [
                            site for site in pmg_structure.sites
                            if site.properties.get("surface_properties") == "surface"
                        ]
                        sites = [site.coords for site in surface_atoms]

                    if len(sites) < num_atoms:
                        logging.warning(f"Not enough sites ({len(sites)}) for {num_atoms} promoter atoms")
                        selected_sites = sites
                    else:
                        selected_sites = random.sample(sites, num_atoms)

                    for site in selected_sites:
                        promoter_atom = Atoms(
                            promoter,
                            positions=[site + np.array([0, 0, 0.5])]
                        )
                        promoted_atoms.extend(promoter_atom)

                    # Calculate energy if calculator is provided
                    if calculator:
                        promoted_atoms.calc = calculator
                        energy = promoted_atoms.get_potential_energy()
                        coverage = num_atoms / len(base_surface_atoms)
                        name = f"{base_surface_name}_{promoter}{coverage:.3f}_{struct_idx}"
                        trial_structures.append((energy, name, promoted_atoms))
                    else:
                        # If no calculator, just add the structure
                        coverage = num_atoms / len(base_surface_atoms)
                        name = f"{base_surface_name}_{promoter}{coverage:.3f}_{struct_idx}"
                        promoted_surfaces.append((name, promoted_atoms))

                except Exception as e:
                    logging.error(
                        f"Failed to add promoter {promoter} to surface "
                        f"{base_surface_name}: {str(e)}"
                    )
                    continue

            # Sort by energy and keep the lowest energy structures
            if calculator and trial_structures:
                trial_structures.sort(key=lambda x: x[0])
                promoted_surfaces.extend(
                    [(name, atoms) for _, name, atoms in trial_structures[:n_promoter_structures]]
                )

        return promoted_surfaces

    @staticmethod
    def get_surface_structures(
        config: Dict,
        mp_ids: List[str],
        miller_indices: List[Tuple[int, int, int]],
        surface_supercell: List[int] = [1, 1, 1],
        metal_type: str = "monometallic",
        dopants: Optional[List[str]] = None,
        fractions: Optional[List[float]] = None,
        n_alloy_structures: Optional[int] = None,
        calculator: Optional[MACECalculator] = None,
        add_promoters: bool = False,
        promoters: Optional[List[str]] = None,
        n_promoter_atoms: Optional[int] = None,
        n_promoter_structures: Optional[int] = None,
        traj_dir: Optional[Path] = None,
    ) -> List[SurfaceStructure]:
        """Generate surface structures with optional doping and promoters."""
        if add_promoters and not all([promoters, n_promoter_atoms, n_promoter_structures]):
            raise ValueError("Promoter parameters required when add_promoters is True")

        if metal_type == "bimetallic" and not all([dopants, fractions, n_alloy_structures]):
            raise ValueError("Doping parameters required for bimetallic surfaces")

        if traj_dir and isinstance(traj_dir, str):
            traj_dir = Path(traj_dir)

        surfaces = []

        # Create single trajectory file for pure surfaces
        pure_traj_path = None
        if traj_dir:
            os.makedirs(traj_dir, exist_ok=True)
            pure_traj_path = traj_dir / "pure_surfaces_relaxation.traj"
            pure_traj = Trajectory(str(pure_traj_path), "a")
        else:
            pure_traj = None

        for mp_id in mp_ids:
            bulk = Bulk(bulk_src_id_from_db=mp_id, bulk_db_path=BULK_PKL_PATH)

            for miller in miller_indices:
                slab = Slab.from_bulk_get_specific_millers(bulk=bulk, specific_millers=miller)
                name = f"{mp_id}_{miller}"
                base_atoms = slab[0].atoms.copy().repeat(surface_supercell)
                
                # Relax the pure surface and add to trajectory
                if calculator and pure_traj:
                    relaxed_atoms = base_atoms.copy()
                    relaxed_atoms.calc = calculator
                    try:
                        exp_cell_filter = FrechetCellFilter(
                            relaxed_atoms, 
                            mask=[True, True, False, False, False, False]
                        )
                        optimizer = FIRE(exp_cell_filter, trajectory=pure_traj)
                        optimizer.run(fmax=0.05, steps=500)
                        base_atoms = relaxed_atoms
                    except Exception as e:
                        logging.error(f"Error relaxing pure surface {name}: {str(e)}")
                
                surfaces.append(SurfaceStructure(name=name, atoms=base_atoms, miller_indices=miller))

                if metal_type == "bimetallic":
                    doped_surfaces = Structure._create_doped_surfaces(
                        base_atoms, mp_id, miller, dopants, fractions,
                        n_alloy_structures, calculator, traj_dir
                    )
                    surfaces.extend(doped_surfaces)

        if pure_traj:
            pure_traj.close()

        # Handle promoters with a separate trajectory file
        if add_promoters:
            promoter_traj_path = None
            if traj_dir:
                promoter_traj_path = traj_dir / "promoter_surfaces_relaxation.traj"
                promoter_traj = Trajectory(str(promoter_traj_path), "a")
            else:
                promoter_traj = None
                
            promoted = []
            for surface in surfaces:
                promoted_surfaces = Structure.add_promoters_to_surface(
                    surface.name,
                    surface.atoms,
                    promoters,
                    n_promoter_atoms,
                    n_promoter_structures,
                    calculator  # Pass the calculator for energy-based selection
                )
                
                for name, atoms in promoted_surfaces:
                    # Relax promoter surface and add to trajectory
                    if calculator and promoter_traj:
                        atoms.calc = calculator
                        try:
                            exp_cell_filter = FrechetCellFilter(
                                atoms, 
                                mask=[True, True, False, False, False, False]
                            )
                            optimizer = FIRE(exp_cell_filter, trajectory=promoter_traj)
                            optimizer.run(fmax=0.05, steps=500)
                        except Exception as e:
                            logging.error(f"Error relaxing promoter surface {name}: {str(e)}")
                    
                    promoted.append(SurfaceStructure(name=name, atoms=atoms))
            
            if promoter_traj:
                promoter_traj.close()
                
            surfaces.extend(promoted)
        
        # Return the final list of surface structures
        return surfaces

    @staticmethod
    def _create_doped_surfaces(
        base_atoms: Atoms,
        mp_id: str,
        miller: Tuple[int, int, int],
        dopants: List[str],
        fractions: List[float],
        n_alloy_structures: int,
        calculator: Optional[MACECalculator] = None,
        traj_dir: Optional[Path] = None
    ) -> List[SurfaceStructure]:
        """Create doped surface structures with the given dopants and fractions."""
        doped_surfaces = []
        
        total_atoms = len(base_atoms)
        
        # Create a single trajectory file for all alloy structures
        traj_path = None
        if traj_dir:
            os.makedirs(traj_dir, exist_ok=True)
            traj_path = traj_dir / f"alloy_structures_{mp_id}_{miller}_relaxation.traj"
            traj = Trajectory(str(traj_path), "a")
        else:
            traj = None
        
        for fraction in fractions:
            for dopant in dopants:
                trial_structures = []
                n_dopant_atoms = int(round(fraction * total_atoms))
                actual_fraction = n_dopant_atoms / total_atoms
                
                n_trials = n_alloy_structures * 10  # Create 10x structures and keep best ones
                
                for struct_idx in range(n_trials):
                    try:
                        doped_atoms = base_atoms.copy()
                        all_indices = list(range(total_atoms))
                        doped_indices = np.random.choice(
                            all_indices, size=n_dopant_atoms, replace=False
                        )
                        
                        symbols = list(doped_atoms.symbols)
                        for idx in doped_indices:
                            symbols[idx] = dopant
                        doped_atoms.symbols = symbols
                        
                        # Calculate energy to filter the best structures
                        if calculator:
                            doped_atoms.calc = calculator
                            energy = doped_atoms.get_potential_energy()
                            name = f"{mp_id}_{dopant}{actual_fraction:.3f}_{miller}_{struct_idx}"
                            trial_structures.append((energy, name, doped_atoms))
                    
                    except Exception as e:
                        logging.error(
                            f"Failed to generate structure {struct_idx}: {str(e)}"
                        )
                        continue
                
                # Sort by energy and keep the n_alloy_structures with lowest energy
                trial_structures.sort(key=lambda x: x[0])
                for _, name, atoms in trial_structures[:n_alloy_structures]:
                    # Relax the structure and add to trajectory
                    if calculator and traj_path:
                        atoms.calc = calculator
                        try:
                            # Use FrechetCellFilter for surface-only relaxation
                            exp_cell_filter = FrechetCellFilter(
                                atoms, 
                                mask=[True, True, False, False, False, False]
                            )
                            optimizer = FIRE(exp_cell_filter, trajectory=traj)
                            optimizer.run(fmax=0.05, steps=500)
                        except Exception as e:
                            logging.error(f"Error relaxing alloy structure {name}: {str(e)}")
                    
                    doped_surfaces.append(
                        SurfaceStructure(
                            name=name,
                            atoms=atoms,
                            miller_indices=miller,
                            dopants=[dopant],
                            fraction=actual_fraction
                        )
                    )
        
        if traj:
            traj.close()
        
        return doped_surfaces

    @staticmethod
    def create_adsorbates_from_smiles(
        reaction_network_nodes: Dict,
        config: Dict,
        calculator: Optional[MACECalculator] = None
    ) -> Dict:
        """Create and relax adsorbate structures from SMILES strings."""
        if not calculator:
            raise ValueError("Calculator required for gas phase relaxations")

        adsorbates = {}
        multiplicity_map = {"[H]": 2, "[C]": 3, "[O]": 3}

        for rxn_name, node_dict in reaction_network_nodes.items():
            adsorbates[rxn_name] = {"nodes": {}}
            
            for node_label, smiles_list in node_dict.items():
                adsorbates[rxn_name]["nodes"][node_label] = []

                for smiles in smiles_list:
                    try:
                        if smiles in multiplicity_map:
                            atoms = Structure._handle_atomic_species(smiles, multiplicity_map)
                        else:
                            atoms = Structure._handle_molecular_species(
                                smiles, config, calculator
                            )
                        
                        if atoms:
                            adsorbates[rxn_name]["nodes"][node_label].append(atoms)
                            
                    except Exception as e:
                        logging.error(f"Failed to process {smiles}: {str(e)}")
                        continue
        return adsorbates

    @staticmethod
    def _create_reference_molecule(smiles: str) -> Atoms:
        """Create reference molecule structure from SMILES."""
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        
        conformer = mol.GetConformer()
        positions = conformer.GetPositions()
        
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        atoms = Atoms(symbols, positions=positions)
        atoms.center(vacuum=15.0)
        
        return atoms

    @staticmethod
    def _handle_atomic_species(smiles: str, multiplicity_map: Dict[str, int]) -> Dict:
        """Process atomic species with correct multiplicity."""
        atoms = Atoms(smiles[1], positions=[[0, 0, 0]])
        atoms.center(vacuum=15.0)
        atoms.set_initial_magnetic_moments([multiplicity_map[smiles] - 1])
        
        return {
            "smiles": smiles,
            "conformers": [atoms],
            "surface_constraints": [],
            "relaxed_bond_lengths": [],
            "multiplicity": multiplicity_map[smiles],
        }
    
    @staticmethod
    def sanitize_filename(name: str) -> str:
        """Convert filename to safe format."""
        # Replace spaces and special characters with underscores
        safe_name = re.sub(r'[^\w\-_]', '_', name)
        # Replace multiple underscores with single underscore
        safe_name = re.sub(r'_+', '_', safe_name)
        # Remove leading/trailing underscores
        return safe_name.strip('_')
    
class AdsorptionConfig(BaseModel):
    min_site_distance: float = 2.0
    adsorption_distance: float = 2.0  # Distance above surface
    max_adsorption_distance: float = 4.0  # Maximum distance to consider
    n_adsorption_attempts: int = 15  # Increased attempts
    site_types: List[str] = ["ontop", "bridge", "hollow"]  # Added site types to consider

class AdsorptionResults(BaseModel):
   site_coordinates: List[List[float]]
   total_energy: float
   formation_energy: float
   relaxation_trajectory: str

   class Config:
       arbitrary_types_allowed = True

class Adsorption:
    @staticmethod
    def inspect_data_structure(data, level=0, max_level=3):
        """Debug helper to inspect nested data structures"""
        indent = "  " * level
        if level >= max_level:
            logging.info(f"{indent}[MAX LEVEL REACHED]")
            return
            
        if isinstance(data, dict):
            logging.info(f"{indent}Dict with keys: {list(data.keys())}")
            if level < max_level - 1:
                for k, v in data.items():
                    logging.info(f"{indent}Key: {k} →")
                    Adsorption.inspect_data_structure(v, level+1, max_level)
        elif isinstance(data, (list, tuple)):
            logging.info(f"{indent}{type(data).__name__} of length {len(data)}")
            if level < max_level - 1 and len(data) > 0:
                logging.info(f"{indent}First element:")
                Adsorption.inspect_data_structure(data[0], level+1, max_level)
        else:
            logging.info(f"{indent}Value: {type(data).__name__} = {str(data)[:100]}")    
    
    @staticmethod
    def create_surface_adsorbate_systems(
        surface_library: List[SurfaceStructure],
        adsorbate_library: Dict,
        config: AdsorptionConfig
    ) -> Dict:
        """Create surface-adsorbate systems using pymatgen's AdsorbateSiteFinder."""
        systems = {}
        
        for surface in surface_library:
            systems[surface.name] = {}
            
            try:
                # Convert ASE atoms to pymatgen structure
                pmg_structure = AseAtomsAdaptor.get_structure(surface.atoms)
                
                # Create AdsorbateSiteFinder
                asf = AdsorbateSiteFinder(pmg_structure)
                all_sites_dict = asf.find_adsorption_sites(distance=config.adsorption_distance)
                
                # Get sites, prioritizing specific site types before falling back to "all"
                all_sites = []
                for site_type in config.site_types:
                    if site_type in all_sites_dict and all_sites_dict[site_type]:
                        all_sites = all_sites_dict[site_type]
                        logging.info(f"Using {site_type} sites for {surface.name}")
                        break
                
                # Fall back to "all" sites if needed
                if not all_sites and "all" in all_sites_dict:
                    all_sites = all_sites_dict["all"]
                    logging.info(f"Falling back to 'all' sites for {surface.name}")
                
                if not all_sites:
                    logging.warning(f"No adsorption sites found for {surface.name}")
                    continue
                
                # Find the correct surface side (assume higher z-coordinate is top surface)
                atoms = surface.atoms
                z_coords = atoms.positions[:,2]
                z_mean = np.mean(z_coords)
                z_max = np.max(z_coords)
                
                # Filter sites to ensure they're only on the top of the slab
                # (higher z than average, but not too high above the maximum)
                top_sites = [site for site in all_sites 
                            if site[2] > z_mean and site[2] < z_max + 5.0]
                
                if not top_sites:
                    logging.warning(f"No suitable top sites found for {surface.name}")
                    continue
                    
                logging.info(f"Found {len(top_sites)} suitable top sites out of {len(all_sites)} total sites")
                
                for rxn_name, rxn_ads in adsorbate_library.items():
                    for node_label, species_list in rxn_ads["nodes"].items():
                        node_key = f"{rxn_name}_{node_label}"
                        systems[surface.name][node_key] = {"configurations": []}
                        
                        for attempt in range(config.n_adsorption_attempts):
                            # Find valid sites that maintain minimum distance
                            valid_sites = []
                            remaining_sites = top_sites.copy()
                            
                            while len(valid_sites) < len(species_list) and remaining_sites:
                                if not remaining_sites:
                                    break
                                    
                                site_idx = random.randrange(len(remaining_sites))
                                candidate_site = remaining_sites.pop(site_idx)
                                
                                # Check distance to existing sites
                                if not valid_sites or all(
                                    np.linalg.norm(np.array(candidate_site) - np.array(existing_site)) >= config.min_site_distance
                                    for existing_site in valid_sites
                                ):
                                    valid_sites.append(candidate_site)
                            
                            if len(valid_sites) < len(species_list):
                                continue
                                
                            try:
                                # Create the combined structure with adsorbates
                                current_atoms = surface.atoms.copy()
                                site_info = []
                                
                                # Get the highest atom z-coord in the slab (surface level)
                                surface_level = max(atom.position[2] for atom in current_atoms)
                                
                                for i, (site, species) in enumerate(zip(valid_sites, species_list)):
                                    # Convert site coordinates to regular Python list
                                    site_coords = [float(c) for c in site]
                                    
                                    # Get a conformer
                                    if not species.get("conformers", []):
                                        logging.warning(f"No conformers found for species {i}")
                                        continue
                                        
                                    conformer = species["conformers"][0].copy()
                                    
                                    # Calculate the height of the conformer
                                    conformer_height = (max(atom.position[2] for atom in conformer) - 
                                                    min(atom.position[2] for atom in conformer))
                                    
                                    # Position the adsorbate with its bottom at adsorption_distance above surface
                                    # First center the conformer at origin in xy plane
                                    conformer_center = np.mean(conformer.positions, axis=0)
                                    conformer.translate([-conformer_center[0], -conformer_center[1], 0])
                                    
                                    # Then place it at the site with proper height
                                    # Position the bottom of the adsorbate at specified distance above surface
                                    conformer_min_z = min(atom.position[2] for atom in conformer)
                                    z_offset = surface_level + config.adsorption_distance - conformer_min_z
                                    
                                    # Move to the site
                                    conformer.translate([site_coords[0], site_coords[1], z_offset])
                                    
                                    # Now add the positioned conformer to the structure
                                    current_atoms.extend(conformer)
                                    
                                    site_info.append({
                                        "coordinates": site_coords,
                                        "species_smiles": species.get("smiles", "unknown")
                                    })
                                
                                # Add configuration with proper site information
                                systems[surface.name][node_key]["configurations"].append({
                                    "surface": current_atoms,
                                    "site_info": site_info
                                })
                                    
                            except Exception as e:
                                logging.error(f"Failed to create {surface.name}_{node_key} configuration: {str(e)}")
                                continue
                
            except Exception as e:
                logging.error(f"Error processing surface {surface.name}: {str(e)}")
                continue
                    
        return systems

    @staticmethod 
    def get_reference_energies(
        surface_library: List[SurfaceStructure],
        adsorbate_library: Dict,
        calculator: MACECalculator,
        config: Config
    ) -> Dict:
        references = {
            "surfaces": {},
            "molecules": {},
            "node_references": {}
        }

        # Get surface reference energies
        for surface in surface_library:
            relaxed = Structure.relax_structure(
                surface.atoms,
                calculator=calculator,
                trajectory=f"ref_surface_{Structure.sanitize_filename(surface.name)}_relaxation.traj",
                traj_dir=config.traj_dir
            )
            references["surfaces"][surface.name] = relaxed.get_potential_energy()

        # Collect unique molecules from reference states
        ref_molecules = set()
        for rxn_name, states in config.reference_states.items():
            for state in states.values():
                ref_molecules.update(smiles for smiles, _ in state)

        # Get molecule reference energies
        for smiles in ref_molecules:
            if smiles not in references["molecules"]:
                atoms = Structure._create_reference_molecule(smiles)
                relaxed = Structure.relax_structure(
                    atoms,
                    calculator=calculator,
                    trajectory=f"ref_mol_{Structure.sanitize_filename(smiles)}_relaxation.traj",
                    traj_dir=config.traj_dir
                )
                references["molecules"][smiles] = relaxed.get_potential_energy()

        # Calculate reference energies for each node
        for rxn_name, rxn_data in adsorbate_library.items():
            references["node_references"][rxn_name] = {}
            for node_label, ref_states in config.reference_states[rxn_name].items():
                total_energy = sum(coeff * references["molecules"][smiles] for smiles, coeff in ref_states)
                references["node_references"][rxn_name][node_label] = total_energy

        return references

    @staticmethod 
    def evaluate_adsorption(
        surface_library: List[SurfaceStructure],
        adsorbate_library: Dict,
        calculator: MACECalculator,
        config: Config
    ) -> Dict:
        results = {}
        references = Adsorption.get_reference_energies(
            surface_library, adsorbate_library, calculator, config
        )
        systems = Adsorption.create_surface_adsorbate_systems(
            surface_library, adsorbate_library, config
        )

        for surface_name, nodes in systems.items():
            results[surface_name] = {}
            
            for node_key, node_data in nodes.items():
                rxn_name, node_label = node_key.split('_')[0], '_'.join(node_key.split('_')[1:])
                results[surface_name][node_key] = {"configurations": []}
                
                for config_data in node_data["configurations"]:
                    try:
                        # Simply relax the combined structure that was already created
                        atoms = config_data["surface"].copy()
                        
                        # Continue with relaxation and energy calculations
                        relaxed = Structure.relax_structure(
                            atoms,
                            calculator=calculator,
                            trajectory=f"{Structure.sanitize_filename(rxn_name)}_{Structure.sanitize_filename(node_label)}_relaxation.traj",
                            traj_dir=config.traj_dir
                        )
                        
                        energy = relaxed.get_potential_energy()
                        formation_energy = (
                            energy 
                            - references["surfaces"][surface_name]
                            - references["node_references"][rxn_name][node_label]
                        )
                        
                        results[surface_name][node_key]["configurations"].append({
                            "total_energy": energy,
                            "formation_energy": formation_energy,
                            "site_info": config_data["site_info"],
                        })
                        
                        # Save results periodically
                        Calculator.save_results(InitResults(
                            filename=Path(config.filename),
                            data=results,
                            config=config.model_dump()
                        ))
                    except Exception as e:
                        logging.error(f"Configuration evaluation failed: {str(e)}")
                        continue
        
        return results

    @staticmethod
    def _evaluate_configuration(
        config_data: Dict,
        rxn_name: str,
        node_label: str,
        adsorbate_library: Dict,
        references: Dict,
        calculator: MACECalculator,
        config: Config,
        surface_name: str
    ) -> Dict:
        atoms = config_data["surface"].copy()
        
        try:
            logging.info(f"Processing configuration: {rxn_name} - {node_label}")
            logging.info(f"Inspecting config_data structure:")
            Adsorption.inspect_data_structure(config_data)
            
            # Check if we have the expected structure
            if "species" not in config_data:
                logging.error("Missing 'species' key in config_data")
                raise ValueError("Missing 'species' key in config_data")
                
            if "site_info" not in config_data:
                logging.error("Missing 'site_info' key in config_data")
                raise ValueError("Missing 'site_info' key in config_data")
            
            # Process each site
            for site_info in config_data["site_info"]:
                # Get the correct species index - handle different types
                if "species_idx" not in site_info:
                    logging.error("Missing 'species_idx' in site_info")
                    continue
                    
                species_idx = site_info["species_idx"]
                
                # Try to convert species_idx to integer if it's a string
                if isinstance(species_idx, str):
                    try:
                        species_idx = int(species_idx)
                    except ValueError:
                        logging.error(f"Cannot convert species_idx '{species_idx}' to integer")
                        continue
                
                # Access the species with proper error handling
                species_list = config_data["species"]
                if not isinstance(species_list, (list, tuple)):
                    logging.error(f"'species' is not a list: {type(species_list)}")
                    continue
                    
                if species_idx < 0 or species_idx >= len(species_list):
                    logging.error(f"Species index {species_idx} out of range [0-{len(species_list)-1}]")
                    continue
                    
                species = species_list[species_idx]
                
                # Get conformer safely
                if not isinstance(species, dict):
                    logging.error(f"Species object is not a dictionary: {type(species)}")
                    continue
                    
                if "conformers" not in species or not species["conformers"]:
                    logging.error("No conformers found in species")
                    continue
                    
                conformer = species["conformers"][0].copy()
                
                # Get coordinates
                if not isinstance(site_info["coordinates"], (list, tuple, np.ndarray)):
                    logging.error(f"Coordinates not in expected format: {type(site_info['coordinates'])}")
                    continue
                    
                # Convert coordinates to array of floats
                coords = site_info["coordinates"]
                try:
                    coords_array = np.array([float(c) for c in coords])
                except (ValueError, TypeError) as e:
                    logging.error(f"Error converting coordinates: {str(e)}")
                    continue
                    
                # Place above surface
                z_max = max(atom.position[2] for atom in atoms)
                if coords_array[2] < z_max:
                    coords_array[2] = z_max + 2.0
                    
                conformer.translate(coords_array)
                atoms.extend(conformer)
            
            # Continue with relaxation and energy calculations
            relaxed = Structure.relax_structure(
                atoms,
                calculator=calculator,
                trajectory=f"{Structure.sanitize_filename(rxn_name)}_{Structure.sanitize_filename(node_label)}_relaxation.traj",
                traj_dir=config.traj_dir
            )
            
            energy = relaxed.get_potential_energy()
            formation_energy = (
                energy 
                - references["surfaces"][surface_name]
                - references["node_references"][rxn_name][node_label]
            )
            
            return {
                "total_energy": energy,
                "formation_energy": formation_energy,
                "site_info": config_data["site_info"],
            }
        except Exception as e:
            logging.error(f"Error in _evaluate_configuration: {str(e)}")
            raise
