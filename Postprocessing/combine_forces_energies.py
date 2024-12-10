import os
import re

def find_point_directories(results_path):
    """Find all point_* directories in the results_chunk_* folder and sort them numerically."""
    point_dirs = [d for d in os.listdir(results_path) if d.startswith("point_") and os.path.isdir(os.path.join(results_path, d))]
    point_dirs.sort(key=lambda x: int(x.split('_')[-1]))  # Sort numerically by the number after "point_"
    return point_dirs

def format_forces_file(forces_content):
    """Format the forces content to match the desired structure for combined forces."""
    formatted_lines = []
    atom_lines = []

    for line in forces_content.splitlines():
        # Skip unnecessary lines (header, sum lines, and any non-numerical lines)
        if "ATOMIC FORCES" in line or "SUM OF ATOMIC FORCES" in line or "Element" in line or line.startswith("#"):
            continue

        # Extract the relevant columns (atom type and forces)
        parts = re.split(r'\s+', line.strip())
        if len(parts) >= 6 and parts[3].replace('.', '', 1).replace('-', '', 1).isdigit():  # Check if it's a valid force line
            atom_type = parts[2]  # The atom type (Element)
            forces = parts[3:6]   # The X, Y, Z forces
            formatted_line = f"{atom_type:<3} {float(forces[0]):>20.8f} {float(forces[1]):>20.8f} {float(forces[2]):>20.8f}"
            atom_lines.append(formatted_line)

    # Prepend the number of atoms as the first line, with four spaces
    if atom_lines:
        formatted_lines.append(f"    {len(atom_lines)}")  # Add four spaces before the number of atoms
        formatted_lines.extend(atom_lines)                # Atom lines

    return "\n".join(formatted_lines)  # No empty lines between structures

def combine_forces_and_energies():
    """Combine forces.xyz and energy.txt files from all chunks and points into single files."""
    combined_forces_file = "combined_forces.xyz"
    combined_energies_file = "combined_energies.txt"

    # Open the files for writing (overwrite if they exist)
    with open(combined_forces_file, 'w') as forces_out, open(combined_energies_file, 'w') as energies_out:
        # Iterate over all chunk_* directories in the current folder
        chunks = sorted([d for d in os.listdir('.') if d.startswith("chunk_") and os.path.isdir(d)])

        first_structure = True  # To handle the first structure differently (no newline before it)

        for chunk in chunks:
            results_path = os.path.join(chunk, f"results_{chunk}")  # The directory where point_* folders are stored
            if not os.path.exists(results_path):
                print(f"Warning: {results_path} not found.")
                continue

            # Find all point_* directories inside the results_* folder, sorted numerically
            point_dirs = find_point_directories(results_path)
            print(f"Processing {len(point_dirs)} points in {chunk}...")

            for point_dir in point_dirs:
                point_path = os.path.join(results_path, point_dir)

                # Combine forces.xyz
                forces_file = os.path.join(point_path, "forces.xyz")
                if os.path.exists(forces_file):
                    with open(forces_file, 'r') as f:
                        forces_content = f.read()
                        formatted_forces = format_forces_file(forces_content)

                        if first_structure:
                            forces_out.write(f"{formatted_forces}")  # Write the first structure
                            first_structure = False
                        else:
                            forces_out.write(f"\n{formatted_forces}")  # Ensure no extra newline before subsequent structures
                else:
                    print(f"Warning: {forces_file} not found in {point_path}")

                # Combine energy.txt
                energy_file = os.path.join(point_path, "energy.txt")
                if os.path.exists(energy_file):
                    with open(energy_file, 'r') as f:
                        energies_out.write(f.read())  # Append the energy value from each energy.txt file
                else:
                    print(f"Warning: {energy_file} not found in {point_path}")

    print(f"Combined forces written to {combined_forces_file}")
    print(f"Combined energies written to {combined_energies_file}")

def main():
    """Main function to combine forces and energies."""
    combine_forces_and_energies()

if __name__ == "__main__":
    main()

