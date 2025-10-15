import numpy as np
from scipy.spatial import distance_matrix
from ase.io import read, write

def estimate_padding(positions):
    """Estimate padding automatically based on nearest-neighbor distances."""
    if len(positions) < 2:
        return 3.0  # fallback if only 1 atom

    # compute pairwise distances
    dmat = distance_matrix(positions, positions)
    dmat[dmat == 0] = np.inf  # ignore self-distances
    nearest = np.min(dmat, axis=1)  # nearest neighbor per atom
    avg_nn = np.mean(nearest)       # average NN distance

    # heuristic: padding = ~2 × avg NN distance
    padding = 2.0 * avg_nn

    # clip between 3.0 and 10.0 Å for sanity
    return float(np.clip(padding, 3.0, 10.0))

def process_xyz(input_file, output_file, png_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    output_lines = []
    i = 0
    while i < len(lines):
        num_atoms_line = lines[i].strip()
        if not num_atoms_line.isdigit():
            i += 1
            continue

        num_atoms = int(num_atoms_line)
        header = lines[i+1].rstrip()

        frame_atoms = []
        frame_positions = []

        for j in range(num_atoms):
            parts = lines[i+2+j].split()
            elem = parts[0]
            pos = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            rest = parts[4:]
            frame_atoms.append((elem, rest))
            frame_positions.append(pos)

        frame_positions = np.array(frame_positions)

        # Bounding box
        mins = np.min(frame_positions, axis=0)
        maxs = np.max(frame_positions, axis=0)
        lengths = maxs - mins

        # Automatic padding
        padding = estimate_padding(frame_positions)

        # Add padding
        box_lengths = lengths + padding

        # Force cubic box
        max_length = np.max(box_lengths)
        box_lengths[:] = max_length

        # Center atoms
        frame_positions -= np.mean(frame_positions, axis=0)
        frame_positions += box_lengths / 2.0

        # Write frame
        output_lines.append(f"{num_atoms}\n")
        output_lines.append(
            f'Lattice="{box_lengths[0]:.6f} 0.0 0.0 0.0 {box_lengths[1]:.6f} 0.0 0.0 0.0 {box_lengths[2]:.6f}" '
            f'Properties=species:S:1:pos:R:3:forces:R:3 pbc="F F F" energy={header} - centered\n'
        )

        for (elem, rest), pos in zip(frame_atoms, frame_positions):
            rest_str = ' '.join(rest)
            output_lines.append(f"{elem} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} {rest_str}\n")

        i += num_atoms + 2

    with open(output_file, 'w') as f:
        f.writelines(output_lines)

    # Save PNG from ASE (first frame only)
    atoms = read(output_file, index=0)
    write(png_file, atoms)

