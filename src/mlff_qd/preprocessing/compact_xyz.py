import numpy as np
import argparse

from mlff_qd.utils.constants import hartree_to_eV, hartree_bohr_to_eV_angstrom
from mlff_qd.utils.io import parse_positions_xyz, parse_forces_xyz, get_num_atoms

def create_stacked_xyz(pos_file, frc_file, output_file_hartree, output_file_ev):
    num_atoms = get_num_atoms(pos_file)
    positions, atom_types, total_energies = parse_positions_xyz(pos_file, num_atoms)
    forces = parse_forces_xyz(frc_file, num_atoms)
    assert positions.shape == forces.shape, "Mismatch in number of frames between positions and forces."

    total_energies_ev = [e * hartree_to_eV if e is not None else None for e in total_energies]
    forces_ev = forces * hartree_bohr_to_eV_angstrom

    print(f"Creating stacked XYZ file in Hartree: {output_file_hartree}")
    with open(output_file_hartree, "w") as f:
        for frame_idx in range(positions.shape[0]):
            energy = total_energies[frame_idx] if total_energies[frame_idx] is not None else 0.0
            f.write(f"{num_atoms}\n")
            f.write(f" {energy:.10f} \n")
            for atom, (x, y, z), (fx, fy, fz) in zip(atom_types, positions[frame_idx], forces[frame_idx]):
                f.write(f"{atom:<2} {x:>12.6f} {y:>12.6f} {z:>12.6f} {fx:>12.6f} {fy:>12.6f} {fz:>12.6f}\n")
    print(f"Stacked XYZ file saved in Hartree to {output_file_hartree}")

    print(f"Creating stacked XYZ file in eV: {output_file_ev}")
    with open(output_file_ev, "w") as f:
        for frame_idx in range(positions.shape[0]):
            energy = total_energies_ev[frame_idx] if total_energies_ev[frame_idx] is not None else 0.0
            f.write(f"{num_atoms}\n")
            f.write(f" {energy:.10f} \n")
            for atom, (x, y, z), (fx, fy, fz) in zip(atom_types, positions[frame_idx], forces_ev[frame_idx]):
                f.write(f"{atom:<2} {x:>12.6f} {y:>12.6f} {z:>12.6f} {fx:>12.6f} {fy:>12.6f} {fz:>12.6f}\n")
    print(f"Stacked XYZ file saved in eV to {output_file_ev}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and stack XYZ files.")
    parser.add_argument("--pos", type=str, required=True, help="Path to the positions XYZ file.")
    parser.add_argument("--frc", type=str, required=True, help="Path to the forces XYZ file.")
    args = parser.parse_args()

    output_file_hartree = "dataset_pos_frc_hartree.xyz"
    output_file_ev = "dataset_pos_frc_ev.xyz"

    create_stacked_xyz(args.pos, args.frc, output_file_hartree, output_file_ev)

