import os
import subprocess
import logging
import numpy as np

def convert_to_schnetpack_db(input_path: str, scratch_dir: str, atomrefs_dict: dict = None) -> str:
    """
    Converts an .npz or .xyz dataset to SchNetPack .db format.

    Supports:
      1) Standard single-target format:
         - energy
         - forces

      2) Spin-combined multi-target format:
         - E_singlet
         - E_triplet
         - f_singlet
         - f_triplet
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
        energy_refs = [0.0] * 119
        for key, val in atomrefs_dict.items():
            if isinstance(key, str) and key in atomic_numbers:
                z = atomic_numbers[key]
            else:
                z = int(key)
            energy_refs[z] = float(val)

        # Only standard "energy" atomrefs are supported here
        my_atomrefs = {"energy": energy_refs}
        logging.info(f"[SchNetPack Wrapper] Built atomrefs for {len(atomrefs_dict)} elements.")

    atoms_list = []
    property_list = []

    # -------------------------
    # NPZ path: keep standard single-target behavior
    # -------------------------
    if input_path.endswith(".npz"):
        ds = ASEAtomsData.create(
            datapath=db_path,
            distance_unit="Ang",
            property_unit_dict={"energy": "eV", "forces": "eV/Ang"},
            atomrefs=my_atomrefs,
        )

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
                "forces": np.array(F[i], dtype=np.float64),
            })

        ds.add_systems(property_list=property_list, atoms_list=atoms_list)
        logging.info(f"[SchNetPack Wrapper] Converted {input_path} to {db_path} (N={len(ds)})")
        return db_path

    # -------------------------
    # XYZ path: support standard OR spin-combined extxyz
    # -------------------------
    elif input_path.endswith(".xyz"):
        from ase.io import read

        frames = read(input_path, index=":")
        if len(frames) == 0:
            raise ValueError(f"No frames found in XYZ file: {input_path}")

        first = frames[0]

        is_spin_combined = (
            "E_singlet" in first.info and
            "E_triplet" in first.info and
            "f_singlet" in first.arrays and
            "f_triplet" in first.arrays
        )

        if is_spin_combined:
            logging.info("[SchNetPack Wrapper] Detected spin-combined XYZ format.")
            ds = ASEAtomsData.create(
                datapath=db_path,
                distance_unit="Ang",
                property_unit_dict={
                    "E_singlet": "eV",
                    "E_triplet": "eV",
                    "f_singlet": "eV/Ang",
                    "f_triplet": "eV/Ang",
                },
                atomrefs=None,  # custom multi-target case
            )

            # -------------------------------------------------
            # Compute mean offsets for the two energy targets
            # -------------------------------------------------
            singlet_energies = []
            triplet_energies = []

            for atoms in frames:
                if "E_singlet" not in atoms.info or "E_triplet" not in atoms.info:
                    raise KeyError("Spin-combined XYZ is missing E_singlet or E_triplet in atoms.info")
                singlet_energies.append(float(atoms.info["E_singlet"]))
                triplet_energies.append(float(atoms.info["E_triplet"]))

            mean_E_singlet = float(np.mean(singlet_energies))
            mean_E_triplet = float(np.mean(triplet_energies))

            logging.info(
                f"[SchNetPack Wrapper] Energy means: "
                f"E_singlet_mean={mean_E_singlet:.10f}, "
                f"E_triplet_mean={mean_E_triplet:.10f}"
            )

            # -------------------------------------------------
            # Store mean-centered energies in the DB
            # -------------------------------------------------
            for atoms in frames:
                if "E_singlet" not in atoms.info or "E_triplet" not in atoms.info:
                    raise KeyError("Spin-combined XYZ is missing E_singlet or E_triplet in atoms.info")
                if "f_singlet" not in atoms.arrays or "f_triplet" not in atoms.arrays:
                    raise KeyError("Spin-combined XYZ is missing f_singlet or f_triplet in atoms.arrays")

                e_s = float(atoms.info["E_singlet"]) - mean_E_singlet
                e_t = float(atoms.info["E_triplet"]) - mean_E_triplet
                f_s = atoms.arrays["f_singlet"]
                f_t = atoms.arrays["f_triplet"]

                atoms_list.append(atoms)
                property_list.append({
                    "E_singlet": np.array([e_s], dtype=np.float64),
                    "E_triplet": np.array([e_t], dtype=np.float64),
                    "f_singlet": np.array(f_s, dtype=np.float64),
                    "f_triplet": np.array(f_t, dtype=np.float64),
                })

        else:
            logging.info("[SchNetPack Wrapper] Detected standard single-target XYZ format.")
            ds = ASEAtomsData.create(
                datapath=db_path,
                distance_unit="Ang",
                property_unit_dict={"energy": "eV", "forces": "eV/Ang"},
                atomrefs=my_atomrefs,
            )

            for atoms in frames:
                e = atoms.info.get("energy", atoms.info.get("dft_energy", 0.0))
                f = atoms.arrays.get("forces", atoms.arrays.get("dft_force", np.zeros((len(atoms), 3))))
                atoms_list.append(atoms)
                property_list.append({
                    "energy": np.array([e], dtype=np.float64),
                    "forces": np.array(f, dtype=np.float64),
                })

        ds.add_systems(property_list=property_list, atoms_list=atoms_list)
        logging.info(f"[SchNetPack Wrapper] Converted {input_path} to {db_path} (N={len(ds)})")
        return db_path

    else:
        raise ValueError(f"Unsupported file extension for DB conversion: {input_path}")


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