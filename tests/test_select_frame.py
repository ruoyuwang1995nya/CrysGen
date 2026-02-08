from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from ase import Atoms
from ase.io import read, write
from dflow.python import OPIO

import crysgen.tools.ase_tools  # ensure element_filter is registered
from crysgen.op.select_frame import SelectFrame


class TestSelectFrame(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_element_filter(self) -> None:
        atoms_keep = Atoms("HO", positions=[[0, 0, 0], [0, 0, 1]])
        atoms_drop = Atoms("NaCl", positions=[[0, 0, 0], [0, 0, 2]])
        structures_path = self.tmp_path / "structures.extxyz"
        write(structures_path, [atoms_keep, atoms_drop], format="extxyz")

        config = {
            "selectors": {
                "element_filter": {
                    "elements_required": ["O"],
                    "elements_not_allowed": ["Na"],
                }
            }
        }

        op = SelectFrame()

        prev_cwd = os.getcwd()
        os.chdir(self.tmp_path)
        try:
            output = op.execute(OPIO({"structures": structures_path, "config": config}))
        finally:
            os.chdir(prev_cwd)

        selected_indices = output["selected_indices"]
        self.assertEqual(selected_indices, [0])

        selected_structures = read(self.tmp_path / output["selected_structures"], ":")
        self.assertEqual(len(selected_structures), 1)
        self.assertEqual(set(selected_structures[0].get_chemical_symbols()), {"H", "O"})


if __name__ == "__main__":
    unittest.main()