from scm.plams import Molecule
from nanoCAT.recipes import replace_surface
import random
from periodictable import elements

def compute_surface_indices_with_replace_surface_dynamic(
    input_file, surface_atom_types, f=1.0, surface_replaced_file="surface_replaced.xyz"
):
    """
    Dynamically replace surface atoms with random elements from the periodic table that are not in the molecule.

    Parameters:
        input_file (str): Path to the input .xyz file.
        surface_atom_types (list): List of atom types considered as surface atoms to be replaced.
        f (float): Fraction of surface atoms to replace (default: 1.0).
        surface_replaced_file (str): Path to the output .xyz file with surface atoms replaced.

    Returns:
        surface_indices (list): Indices of the replaced surface atoms in the structure.
        replaced_atom_types (list): Updated atom types after replacement.
    """
    print(f"Reading molecule from {input_file}...")
    mol_original = Molecule(input_file)  # Load the original molecule
    mol_updated = mol_original.copy()  # Create a copy to track cumulative changes

    # Identify atom types in the molecule
    molecule_atom_types = {atom.symbol for atom in mol_original}

    # Generate a list of replacement elements from the periodic table
    available_elements = [el.symbol for el in elements if el.symbol not in molecule_atom_types]

    # Prepare replacements dynamically
    replacements = [
        (atom_type, random.choice(available_elements)) for atom_type in surface_atom_types
    ]
    print(f"Dynamic replacements: {replacements}")

    surface_indices = []  # Collect indices of replaced surface atoms
    for i, (original_symbol, replacement_symbol) in enumerate(replacements):
        print(f"Replacing surface atoms: {original_symbol} -> {replacement_symbol} (f={f})...")

        # Create a new molecule for this replacement
        mol_new = replace_surface(mol_updated, symbol=original_symbol, symbol_new=replacement_symbol, f=f)

        # Update `mol_updated` to incorporate the changes
        mol_updated = mol_new.copy()

        # Identify the replaced atoms in the molecule
        for idx, atom in enumerate(mol_new):
            if atom.symbol == replacement_symbol:
                surface_indices.append(idx)

        print(f"Replacement {i+1}: {len(surface_indices)} surface atoms replaced so far.")

    # Save the final updated molecule to the output file
    print(f"Writing modified molecule with replacements to {surface_replaced_file}...")
    mol_updated.write(surface_replaced_file)

    # Extract updated atom types
    replaced_atom_types = [atom.symbol for atom in mol_updated]

    print(f"Surface replacements completed. {len(surface_indices)} surface atoms identified and replaced.")
    return surface_indices, replaced_atom_types
