import os
import argparse
import re
import glob

from mlff_qd.utils.constants import hartree_to_eV, hartree_bohr_to_eV_angstrom

def extract_forces(forces_path):
    """Extract forces from forces.xyz file."""
    forces = []
    with open(forces_path, "r") as f:
        lines = f.readlines()
        start = None
        for i, line in enumerate(lines):
            if "ATOMIC FORCES in [a.u.]" in line:
                start = i + 2  # Skip header lines
                break
        if start is not None:
            for line in lines[start:]:
                if "Kind" in line or "SUM OF ATOMIC FORCES" in line or line.strip() == "":
                    continue  # Skip unwanted lines
                parts = line.split()
                if len(parts) >= 6:
                    element = parts[2]
                    x, y, z = map(float, parts[3:6])
                    forces.append((element, x, y, z))
    return forces

def extract_coordinates(coord_path):
    """Extract coordinates from cp2k_job.in file."""
    coordinates = []
    with open(coord_path, "r") as f:
        lines = f.readlines()
        start, end = None, None
        for i, line in enumerate(lines):
            if "&COORD" in line:
                start = i + 1
            elif "&END" in line and start is not None:
                end = i
                break
        if start is not None and end is not None:
            for line in lines[start:end]:
                parts = line.split()
                if len(parts) >= 4:
                    element = parts[0]
                    x, y, z = map(float, parts[1:4])
                    coordinates.append((element, x, y, z))
    return coordinates

def extract_total_energy(out_path):
    """Extract the total energy from cp2k_job.out."""
    total_energy = None
    with open(out_path, "r") as f:
        for line in f:
            if "Total energy:" in line:
                total_energy = float(line.split()[-1])
                break
    return total_energy

def get_scratch_chunk_folders():
    """Return a sorted list of folders starting with 'scratch_chunk_' in the current directory."""
    folders = [d for d in os.listdir('.') if d.startswith("scratch_chunk_") and os.path.isdir(d)]
    print(f"Found scratch_chunk folders: {folders}")
    return sorted(folders)

def get_cp2k_job_folders(scratch_folder):
    """
    Return a sorted list of CP2K job folders within a given scratch_chunk folder.
    Folders starting with 'cp2k_job' are considered.
    """
    cp2k_folders = []
    for entry in os.listdir(scratch_folder):
        full_path = os.path.join(scratch_folder, entry)
        if os.path.isdir(full_path) and entry.startswith("cp2k_job"):
            cp2k_folders.append(full_path)
    # Sort based on numeric value in folder name (if any)
    def extract_number(folder_name):
        match = re.search(r"cp2k_job(?:\.(\d+))?", os.path.basename(folder_name))
        if match and match.group(1):
            return int(match.group(1))
        return -1  # The folder "cp2k_job" without extension gets priority.
    sorted_folders = sorted(cp2k_folders, key=extract_number)
    print(f"In {scratch_folder}, found cp2k folders: {sorted_folders}")
    return sorted_folders

def get_frame_number_from_point_file(folder):
    """
    Look for point_*.wfn files within the folder.
    If more than one file exists for the same frame number, select the one with the latest modification time.
    The filename pattern is assumed to be: point_<frame_number>-RESTART.wfn
    Return the frame number (as an integer) from the selected file.
    If no such file is found, return None.
    """
    point_files = glob.glob(os.path.join(folder, "point_*-RESTART.wfn"))
    if not point_files:
        return None

    # Dictionary mapping frame number to (mod_time, filename) for the latest file
    frame_files = {}
    for pf in point_files:
        # Match the digits before the hyphen
        match = re.search(r"point_(\d+)-", os.path.basename(pf))
        if match:
            frame_num = int(match.group(1))
            mod_time = os.path.getmtime(pf)
            if frame_num not in frame_files or mod_time > frame_files[frame_num][0]:
                frame_files[frame_num] = (mod_time, pf)
    # Pick the highest frame number among those found
    highest_frame = max(frame_files.keys())
    return highest_frame

def process_cp2k_jobs():
    """
    Process CP2K job folders found in all scratch_chunk_* folders.
    Only process a folder if it contains a forces.xyz file and point_*.wfn files.
    If more than one point file exists with the same frame number, the one with
    the latest modification time is used. Duplicate frames (from restarted jobs)
    are skipped.
    """
    output_forces_hartree_file = "forces_hartree_all.xyz"
    output_forces_ev_file = "forces_ev_all.xyz"
    output_coords_hartree_file = "positions_hartree_all.xyz"
    output_coords_ev_file = "positions_ev_all.xyz"

    all_forces_hartree = []
    all_forces_ev = []
    all_coords_hartree = []
    all_coords_ev = []
    
    processed_frames = {}

    scratch_folders = get_scratch_chunk_folders()
    for scratch_folder in scratch_folders:
        cp2k_folders = get_cp2k_job_folders(scratch_folder)
        for folder in cp2k_folders:
            forces_path = os.path.join(folder, "forces.xyz")
            if not os.path.exists(forces_path):
                continue

            # Determine frame number from point_*.wfn files,
            # selecting the last generated if duplicates exist.
            frame_number = get_frame_number_from_point_file(folder)
            if frame_number is None:
                print(f"No point_*.wfn file found in {folder}. Skipping...")
                continue

            # If we have already processed this frame (from a restarted job), skip it.
            if frame_number in processed_frames:
                continue

            base_name = os.path.basename(folder)
            coord_in_file = os.path.join(folder, f"{base_name}.in")
            out_file = os.path.join(folder, f"{base_name}.out")

            if not (os.path.exists(coord_in_file) and os.path.exists(out_file)):
                print(f"Missing .in or .out file in {folder}. Skipping...")
                continue

            print(f"Processing {folder} for frame {frame_number}...")

            forces = extract_forces(forces_path)
            if not forces:
                print(f"No valid forces found in {forces_path}. Skipping...")
                continue

            # Create XYZ block for forces with fixed-width formatting.
            forces_hartree_block = f"{len(forces)}\nFrame = {frame_number}, units = Hartree/Bohr"
            forces_ev_block = f"{len(forces)}\nFrame = {frame_number}, units = eV/Å"
            for e, x, y, z in forces:
                forces_hartree_block += f"\n{e} {x:12.8f} {y:12.8f} {z:12.8f}"
                forces_ev_block += f"\n{e} {x * hartree_bohr_to_eV_angstrom:12.8f} {y * hartree_bohr_to_eV_angstrom:12.8f} {z * hartree_bohr_to_eV_angstrom:12.8f}"
            all_forces_hartree.append(forces_hartree_block)
            all_forces_ev.append(forces_ev_block)

            # Process coordinates and total energy.
            coordinates = extract_coordinates(coord_in_file)
            total_energy = extract_total_energy(out_file)
            if total_energy is None:
                print(f"Warning: No total energy found in {out_file}. Skipping...")
                continue

            title_hartree = f"Frame = {frame_number}, units = Angstrom, E = {total_energy:.8f}"
            coords_block = f"{len(coordinates)}\n{title_hartree}"
            for e, x, y, z in coordinates:
                coords_block += f"\n{e} {x:12.8f} {y:12.8f} {z:12.8f}"
            
            all_coords_hartree.append((total_energy, coordinates, coords_block))
            all_coords_ev.append((total_energy * hartree_to_eV, coordinates, coords_block))

            processed_frames[frame_number] = folder

    # Write outputs in ordered XYZ format (frames concatenated without extra blank lines)
    with open(output_forces_hartree_file, "w") as f:
        f.write("\n".join(all_forces_hartree))
    with open(output_forces_ev_file, "w") as f:
        f.write("\n".join(all_forces_ev))
    with open(output_coords_hartree_file, "w") as f:
        f.write("\n".join(entry[2] for entry in all_coords_hartree))
    with open(output_coords_ev_file, "w") as f:
        f.write("\n".join(entry[2] for entry in all_coords_ev))

    total_frames = len(processed_frames)
    print(f"Processed {total_frames} unique frames. Outputs saved to:")
    print(f"  {output_forces_hartree_file}")
    print(f"  {output_forces_ev_file}")
    print(f"  {output_coords_hartree_file}")
    print(f"  {output_coords_ev_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CP2K job outputs from scratch_chunk folders.")
    args = parser.parse_args()
    process_cp2k_jobs()
