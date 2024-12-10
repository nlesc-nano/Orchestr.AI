import os
import re
import shutil  # For copying files

def load_scratch_path(file):
    """Load the scratch path from the YAML configuration file."""
    import yaml
    with open(file, 'r') as f:
        params = yaml.safe_load(f)
    return params['scratch_path']

def find_cp2k_job_folders(scratch_path, chunk_name):
    """Find all cp2k_job.* folders in the scratch path, including cp2k_job for a specific chunk."""
    chunk_path = os.path.join(scratch_path, chunk_name)
    cp2k_job_folders = []
    
    for folder in os.listdir(chunk_path):
        if folder == "cp2k_job":
            cp2k_job_folders.insert(0, os.path.join(chunk_path, folder))  # Add cp2k_job first
        elif folder.startswith("cp2k_job."):
            full_path = os.path.join(chunk_path, folder)
            if os.path.isdir(full_path):
                cp2k_job_folders.append(full_path)  # Add numbered cp2k_job.* folders after

    cp2k_job_folders.sort(key=lambda x: int(x.split('.')[-1]) if '.' in x else -1)  # Sort numbered folders
    return cp2k_job_folders

def find_point_directories(results_path):
    """Find all point_* directories in the results_chunk_* folder and sort them numerically."""
    point_dirs = [d for d in os.listdir(results_path) if d.startswith("point_") and os.path.isdir(os.path.join(results_path, d))]
    point_dirs.sort(key=lambda x: int(x.split('_')[-1]))  # Sort numerically by the number after "point_"
    return point_dirs

def extract_structure_from_infile(infile):
    """Extract the structure after &COORD from the cp2k_job.*.in file."""
    structure_lines = []
    recording = False
    
    with open(infile, 'r') as f:
        for line in f:
            if "&COORD" in line:
                recording = True  # Start recording lines after &COORD
                continue
            
            if recording:
                # Stop recording if another section starts
                if "&END" in line or line.startswith('&'):
                    break
                structure_lines.append(line.strip())
    
    return structure_lines

def extract_structure_from_xyz(xyz_file):
    """Extract the first four lines of a structure from a .xyz file, skipping the header line."""
    structure_lines = []
    with open(xyz_file, 'r') as f:
        next(f)  # Skip the first header line
        for line in f:
            if line.strip():
                parts = re.split(r'\s+', line.strip())
                if len(parts) >= 4 and parts[1].replace('.', '', 1).isdigit():  # Check if it's a valid coordinate line
                    structure_lines.append(line.strip())
            if len(structure_lines) == 4:  # Only take the first four lines
                break
    return structure_lines

def normalize_structure(structure_lines, precision=3):
    """Normalize the structure lines by removing extra spaces and rounding numbers."""
    normalized_structure = []
    for line in structure_lines:
        parts = re.split(r'\s+', line.strip())  # Split by any number of spaces
        if len(parts) >= 4:  # Expecting format: atom_type x y z
            atom_type = parts[0]
            coords = [f"{round(float(coord), precision)}" for coord in parts[1:4]]  # Round to given decimal places
            normalized_structure.append([atom_type] + coords)
    
    return normalized_structure[:4]  # Only return the first four lines for comparison

def compare_structures(structure1, structure2):
    """Compare two structures based on their first four lines."""
    return structure1 == structure2

def copy_forces_file(cp2k_folder, point_dir):
    """Copy the forces.xyz file from cp2k_folder to point_dir."""
    forces_file = os.path.join(cp2k_folder, "forces.xyz")
    if os.path.exists(forces_file):
        dest_file = os.path.join(point_dir, "forces.xyz")
        shutil.copy(forces_file, dest_file)
        print(f"Copied {forces_file} to {dest_file}")
    else:
        print(f"Warning: {forces_file} not found in {cp2k_folder}")

def extract_total_energy(cp2k_folder, point_dir):
    """Extract 'Total energy' from cp2k_job*.out and save it in energy.txt in point_dir."""
    out_files = [f for f in os.listdir(cp2k_folder) if f.startswith("cp2k_job") and f.endswith(".out")]
    
    if not out_files:
        print(f"No cp2k_job*.out files found in {cp2k_folder}")
        return
    
    out_file = os.path.join(cp2k_folder, out_files[0])  # Assuming there is only one relevant .out file
    
    with open(out_file, 'r') as f:
        for line in f:
            if "Total energy:" in line:
                parts = line.split()
                if len(parts) >= 3:
                    energy = parts[-1]  # The energy value is the last part of the line
                    energy_file = os.path.join(point_dir, "energy.txt")
                    with open(energy_file, 'w') as ef:
                        ef.write(f"{energy}\n")
                    print(f"Extracted energy: {energy} from {out_file} and saved in {energy_file}")
                break

def process_all_structures_in_chunk(scratch_path, chunk_name, local_chunk_name):
    """Process all structures in a specific chunk, comparing them with cp2k_job.* folders and finding matches."""
    chunk_path = os.path.join(scratch_path, chunk_name)
    results_path = os.path.join(local_chunk_name, f"results_{local_chunk_name}")  # The directory where structure.xyz files are saved locally
    match_log_file = f"matches_log_{local_chunk_name}.txt"  # Output file for logging matches for each chunk
    
    if not os.path.isdir(chunk_path):
        print(f"Error: {chunk_name} directory not found at {chunk_path}")
        return
    
    cp2k_job_folders = find_cp2k_job_folders(scratch_path, chunk_name)  # Get cp2k_job folders inside the chunk
    if not cp2k_job_folders:
        print(f"No cp2k_job folders found in {chunk_path}")
        return
    
    point_dirs = find_point_directories(results_path)  # Find all point_* directories in the local chunk
    print(f"Found {len(cp2k_job_folders)} cp2k_job folders and {len(point_dirs)} point_* directories in {local_chunk_name}.")
    
    cp2k_idx = 0  # Keep track of where to start comparing for each structure
    matches = []  # Store matches for printing later
    
    # Open the file for writing (will overwrite if it exists)
    with open(match_log_file, 'w') as log_file:
        log_file.write(f"Matches between structure.xyz and cp2k_job.*.in files in {local_chunk_name}:\n")
        log_file.write("=" * 60 + "\n")
    
        for point_dir in point_dirs:
            point_idx = point_dir.split('_')[-1]  # Extract point index from directory name
            
            # Load the structure from structure.xyz for the current point
            xyz_file = os.path.join(results_path, point_dir, "structure.xyz")
            if not os.path.exists(xyz_file):
                print(f"Warning: {xyz_file} not found.")
                continue
            
            xyz_structure_lines = extract_structure_from_xyz(xyz_file)
            normalized_xyz_structure = normalize_structure(xyz_structure_lines)
            
            match_found = False
            
            # Compare with cp2k_job.* folders starting from the last unexplored folder
            for cp2k_folder in cp2k_job_folders[cp2k_idx:]:
                forces_file = os.path.join(cp2k_folder, "forces.xyz")
                
                # Skip folders that don't contain the forces.xyz file
                if not os.path.exists(forces_file):
                    print(f"Skipping {cp2k_folder} because forces.xyz is missing.")
                    continue

                print(f"Comparing {point_dir} in {local_chunk_name} with {cp2k_folder}")
                
                # Find the first .in file in the cp2k_job.* folder
                in_files = [f for f in os.listdir(cp2k_folder) if f.endswith('.in')]
                if not in_files:
                    print(f"No .in files found in {cp2k_folder}")
                    continue
                
                first_infile = os.path.join(cp2k_folder, in_files[0])
                cp2k_structure_lines = extract_structure_from_infile(first_infile)
                normalized_cp2k_structure = normalize_structure(cp2k_structure_lines)
                
                # Compare the structures
                if compare_structures(normalized_cp2k_structure, normalized_xyz_structure):
                    print(f"\nStructure in {xyz_file} matches the structure in {first_infile}!\n")
                    matches.append((xyz_file, first_infile))  # Log the match
                    log_file.write(f"Matched {xyz_file} with {first_infile}\n")  # Write the match to the file
                    log_file.write("-" * 60 + "\n")
                    
                    # Copy forces.xyz file to the point_* directory
                    copy_forces_file(cp2k_folder, os.path.join(results_path, point_dir))
                    
                    # Extract and save total energy to energy.txt
                    extract_total_energy(cp2k_folder, os.path.join(results_path, point_dir))
                    
                    match_found = True
                    cp2k_idx = cp2k_job_folders.index(cp2k_folder) + 1  # Move to the next folder after the match
                    break
                else:
                    print(f"\nStructure in {xyz_file} does NOT match the structure in {first_infile}.\n")
            
            if not match_found:
                print(f"No match found for {xyz_file} in the cp2k_job folders.\n")
                log_file.write(f"No match found for {xyz_file}\n")
                log_file.write("-" * 60 + "\n")

def find_all_chunks(scratch_path):
    """Find all scratch_chunk_* directories in the scratch path."""
    return sorted([d for d in os.listdir(scratch_path) if d.startswith("scratch_chunk_") and os.path.isdir(os.path.join(scratch_path, d))])

def main():
    file = "input_parameters.yml"  # Ensure this is the correct path to your YAML config
    scratch_path = load_scratch_path(file)
    
    # Find all scratch_chunk_* directories in the scratch path
    chunks = find_all_chunks(scratch_path)
    print(f"Found {len(chunks)} chunk directories: {chunks}")
    
    # Process each chunk
    for chunk in chunks:
        local_chunk_name = chunk.replace("scratch_", "")  # Convert scratch_chunk_0 to chunk_0 for the local directory
        process_all_structures_in_chunk(scratch_path, chunk, local_chunk_name)

if __name__ == "__main__":
    main()

