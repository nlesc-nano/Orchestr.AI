import numpy as np
import argparse

HARTREE_TO_EV = 27.2114
HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = 51.4221

def parse_positions_xyz(filename, num_atoms):
    print(f"Parsing positions XYZ file: {filename}")
    positions = []
    atom_types = []
    total_energies = []

    with open(filename, "r") as f:
        lines = f.readlines()
        num_lines_per_frame = num_atoms + 2

        for i in range(0, len(lines), num_lines_per_frame):
            atom_lines = lines[i + 2:i + 2 + num_atoms]
            comment_line = lines[i + 1]

            try:
                total_energy = float(comment_line.split("=")[-1].strip())
                total_energies.append(total_energy)
            except ValueError:
                total_energies.append(None)

            frame_positions = []
            for line in atom_lines:
                parts = line.split()
                atom_types.append(parts[0])
                frame_positions.append([float(x) for x in parts[1:4]])

            positions.append(frame_positions)

    return np.array(positions), atom_types[:num_atoms], total_energies

def parse_forces_xyz(filename, num_atoms):
    print(f"Parsing forces XYZ file: {filename}")
    forces = []
    with open(filename, "r") as f:
        lines = f.readlines()
        num_lines_per_frame = num_atoms + 2
        for i in range(0, len(lines), num_lines_per_frame):
            frame_forces = []
            for j in range(2, 2 + num_atoms):
                parts = lines[i + j].split()
                frame_forces.append(list(map(float, parts[1:4])))
            forces.append(frame_forces)
    return np.array(forces)

def get_num_atoms(filename):
    with open(filename, "r") as f:
        num_atoms = int(f.readline().strip())
    print(f"Number of atoms: {num_atoms}")
    return num_atoms

def create_stacked_xyz(pos_file, frc_file, output_file_hartree, output_file_ev):
    num_atoms = get_num_atoms(pos_file)
    positions, atom_types, total_energies = parse_positions_xyz(pos_file, num_atoms)
    forces = parse_forces_xyz(frc_file, num_atoms)
    assert positions.shape == forces.shape, "Mismatch in number of frames between positions and forces."

    total_energies_ev = [e * HARTREE_TO_EV if e is not None else None for e in total_energies]
    forces_ev = forces * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM

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

