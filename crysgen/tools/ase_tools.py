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



