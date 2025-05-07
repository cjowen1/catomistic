"""Create Pt surface using ASE."""

from ase.build import bulk, surface
from ase.io import write
from pathlib import Path

def main():
    # Create bulk Pt with FCC structure
    # a = 3.92 Å is the experimental lattice constant for Pt
    bulk_pt = bulk('Pt', 'fcc', a=3.92)
    
    # Create the output directory if it doesn't exist
    output_dir = Path("data/structures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the bulk structure
    write(str(output_dir / "Pt_bulk.cif"), bulk_pt)
    
    # Create (111) surface with 3 layers and 15 Å vacuum
    pt_111 = surface(bulk_pt, (1,1,1), layers=3, vacuum=15.0)
    write(str(output_dir / "Pt_111.cif"), pt_111)
    
    # Create (100) surface with 3 layers and 15 Å vacuum
    pt_100 = surface(bulk_pt, (1,0,0), layers=3, vacuum=15.0)
    write(str(output_dir / "Pt_100.cif"), pt_100)
    
    # Create (110) surface with 3 layers and 15 Å vacuum
    pt_110 = surface(bulk_pt, (1,1,0), layers=3, vacuum=15.0)
    write(str(output_dir / "Pt_110.cif"), pt_110)

if __name__ == "__main__":
    main() 
