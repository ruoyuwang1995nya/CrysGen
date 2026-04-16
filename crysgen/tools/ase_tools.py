from typing import List, Sequence, Tuple

from ase.atoms import Atoms

from .base_tools import Tools


@Tools.register("element_filter")
def element_filter(
    atoms_ls: Sequence[Atoms],
    elements_not_allowed: Sequence[str] | None = None,
    elements_required: Sequence[str] | None = None,
) -> Tuple[List[Atoms], List[bool]]:
    """Filter a list of ASE ``Atoms`` by element content.

    Returns the filtered list and a boolean mask aligned to the input order.
    An ``Atoms`` is kept if it contains all ``elements_required`` (when given)
    and contains none of ``elements_not_allowed`` (when given).
    """

    not_allowed = set(elements_not_allowed or [])
    required = set(elements_required or [])

    filtered: List[Atoms] = []
    mask: List[bool] = []

    for atoms in atoms_ls:
        symbols = set(atoms.get_chemical_symbols())

        if not_allowed and symbols & not_allowed:
            mask.append(False)
            continue

        if required and not required.issubset(symbols):
            mask.append(False)
            continue

        mask.append(True)
        filtered.append(atoms)

    return filtered, mask


@Tools.register("multi_element_filter")
def multi_element_filter(
    atoms_ls: Sequence[Atoms],
    **kwargs
) -> Tuple[List[Atoms], List[bool]]:
    """Filter out structures with only a single element type.

    Excludes elemental structures (e.g., pure Si, pure Fe) and keeps only
    compounds with multiple element types.

    Args:
        atoms_ls: Sequence of ASE Atoms objects to filter.

    Returns:
        Tuple of (filtered_atoms_list, boolean_mask).
        An ``Atoms`` is kept only if it contains more than one element type.
    """
    filtered: List[Atoms] = []
    mask: List[bool] = []

    for atoms in atoms_ls:
        unique_elements = set(atoms.get_chemical_symbols())
        
        if len(unique_elements) > 1:
            mask.append(True)
            filtered.append(atoms)
        else:
            mask.append(False)

    return filtered, mask


@Tools.register("oxidation_state_balance_filter")
def oxidation_state_balance_filter(
    atoms_ls: Sequence[Atoms],
    tolerance: float = 1e-3,
) -> Tuple[List[Atoms], List[bool]]:
    """Filter a list of ASE ``Atoms`` by oxidation state balance.

    Materials with balanced oxidation states (sum close to zero) are likely
    to be ionic insulators/semiconductors, while unbalanced materials are
    likely to be metallic or unstable.

    Args:
        atoms_ls: Sequence of ASE Atoms objects to filter.
        tolerance: Tolerance for oxidation state balance (default: 1e-3).
                  Oxidation states summing to within ±tolerance are considered balanced.

    Returns:
        Tuple of (filtered_atoms_list, boolean_mask).
        An ``Atoms`` is kept if its oxidation states can be determined and balance.
    """
    from pymatgen.analysis.bond_valence import BVAnalyzer
    from pymatgen.io.ase import AseAtomsAdaptor
    

    filtered: List[Atoms] = []
    mask: List[bool] = []
    adaptor = AseAtomsAdaptor()
    bv_analyzer = BVAnalyzer()

    for atoms in atoms_ls:
        try:
            # Convert ASE Atoms to pymatgen Structure
            structure = adaptor.get_structure(atoms)
            
            # Try to get oxidation states using bond valence analyzer
            try:
                structure_with_oxi = bv_analyzer.get_oxi_state_decorated_structure(structure)
                
                # Check if oxidation states balance
                # Calculate total charge
                total_charge = sum(
                    site.specie.oxi_state 
                    for site in structure_with_oxi
                )
                
                # If charge balances (close to zero), keep the structure
                if abs(total_charge) <= tolerance:
                    mask.append(True)
                    filtered.append(atoms)
                else:
                    # Unbalanced charge suggests metallic or problematic structure
                    mask.append(False)
                    
            except (ValueError, TypeError):
                # If BVAnalyzer fails, try composition-based oxidation state guessing
                composition = structure.composition
                try:
                    oxi_state_guesses = composition.oxi_state_guesses()
                    if oxi_state_guesses:
                        # Use the first (most likely) guess
                        oxi_states = oxi_state_guesses[0]
                        # Check balance
                        total_charge = sum(
                            oxi_states[el] * composition[el]
                            for el in oxi_states
                        )
                        if abs(total_charge) <= tolerance:
                            mask.append(True)
                            filtered.append(atoms)
                        else:
                            mask.append(False)
                    else:
                        # No valid oxidation state guess - likely metallic
                        mask.append(False)
                except (ValueError, TypeError):
                    # Cannot determine oxidation states - exclude as potentially metallic
                    mask.append(False)
                    
        except Exception:
            # Any other error - exclude the structure
            mask.append(False)

    return filtered, mask


if __name__ == "__main__":
    # Simple test case for Si
    import numpy as np
    
    # Create a simple silicon structure (diamond cubic)
    si_atoms = Atoms(
        symbols=['Si'] * 8,
        positions=[
            [0.00, 0.00, 0.00],
            [0.25, 0.25, 0.25],
            [0.50, 0.50, 0.00],
            [0.75, 0.75, 0.25],
            [0.50, 0.00, 0.50],
            [0.75, 0.25, 0.75],
            [0.00, 0.50, 0.50],
            [0.25, 0.75, 0.75],
        ],
        cell=[5.43, 5.43, 5.43],
        pbc=True
    )
    
    # Test oxidation state balance filter
    filtered, mask = oxidation_state_balance_filter([si_atoms])
    
    print(f"Test: Silicon structure")
    print(f"Passed filter: {mask[0]}")
    print(f"Number of structures kept: {len(filtered)}")
    
    if mask[0]:
        print("✓ Silicon structure correctly identified as non-metallic (balanced oxidation states)")
    else:
        print("✗ Silicon structure incorrectly filtered out")



