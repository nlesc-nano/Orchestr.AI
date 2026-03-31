import os
import subprocess
import logging
import numpy as np

def convert_to_schnetpack_db(input_path: str, scratch_dir: str, atomrefs_dict: dict = None) -> str:
    """
    Converts an .npz or .xyz dataset to SchNetPack .db format.
    """
    if input_path.endswith(".db"):
        return input_path

    db_path = os.path.join(scratch_dir, "dataset.db")
    if os.path.exists(db_path):
        logging.info(f"[SchNetPack Wrapper] DB already exists at {db_path}, removing old one.")
        os.remove(db_path)

    from schnetpack.data import ASEAtomsData
    from ase import Atoms
    from ase.data import atomic_numbers

    # Dynamically map atomrefs if provided by user
    my_atomrefs = None
    if atomrefs_dict:
        # Create an array large enough to hold all elements up to Oganesson (118)
        energy_refs = [0.0] * 119
        for key, val in atomrefs_dict.items():
            # If the user provides a symbol like 'Cl', look up its atomic number
            if isinstance(key, str) and key in atomic_numbers:
                z = atomic_numbers[key]
            else:
                z = int(key)
            energy_refs[z] = float(val)
        
        my_atomrefs = {"energy": energy_refs}
        logging.info(f"[SchNetPack Wrapper] Built atomrefs for {len(atomrefs_dict)} elements.")

    ds = ASEAtomsData.create(
        datapath=db_path,
        distance_unit="Ang",
        property_unit_dict={"energy": "eV", "forces": "eV/Ang"},
        atomrefs=my_atomrefs,
    )

    atoms_list = []
    property_list = []

    if input_path.endswith(".npz"):
        npz = np.load(input_path, allow_pickle=True)
        def pick(*cands):
            for k in cands:
                if k in npz.files:
                    return k
            raise KeyError(f"None of these keys found: {cands} in {input_path}")
        
        kZ = pick("z", "Z", "numbers", "atomic_numbers")
        kR = pick("R", "positions", "pos")
        kE = pick("E", "energy", "energies")
        kF = pick("F", "forces", "force")

        Z = np.asarray(npz[kZ])
        R = np.asarray(npz[kR])
        E = np.asarray(npz[kE])
        F = np.asarray(npz[kF])

        for i in range(len(R)):
            atoms_list.append(Atoms(numbers=Z, positions=R[i]))
            property_list.append({
                "energy": np.array([E[i]], dtype=np.float64),
                "forces": np.array(F[i], dtype=np.float64)
            })
    elif input_path.endswith(".xyz"):
        from ase.io import read
        frames = read(input_path, index=":")
        for atoms in frames:
            e = atoms.info.get("energy", atoms.info.get("dft_energy", 0.0))
            f = atoms.arrays.get("forces", atoms.arrays.get("dft_force", np.zeros((len(atoms), 3))))
            atoms_list.append(atoms)
            property_list.append({
                "energy": np.array([e], dtype=np.float64),
                "forces": np.array(f, dtype=np.float64)
            })
    else:
        raise ValueError(f"Unsupported file extension for DB conversion: {input_path}")

    ds.add_systems(property_list=property_list, atoms_list=atoms_list)
    logging.info(f"[SchNetPack Wrapper] Converted {input_path} to {db_path} (N={len(ds)})")
    
    return db_path

def run_schnetpack_training(config_yaml_path: str):
    """
    Run SchNetPack training using the spktrain CLI with a generated Hydra config.
    """
    if not os.path.exists(config_yaml_path):
        raise FileNotFoundError(f"Config file not found: {config_yaml_path}")

    config_dir = os.path.dirname(os.path.abspath(config_yaml_path))
    config_name = os.path.basename(config_yaml_path).replace(".yaml", "")

    logging.info(f"[SchNetPack Wrapper] Starting spktrain with config: {config_name} from {config_dir}")

    cmd = [
        "spktrain",
        f"--config-dir={config_dir}",
        f"--config-name={config_name}"
    ]

    try:
        subprocess.run(cmd, check=True)
        logging.info("[SchNetPack Wrapper] Training completed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"[SchNetPack Wrapper] Training failed with return code {e.returncode}")
        raise