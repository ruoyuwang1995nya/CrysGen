from .foo import *
from .vasp import PrepVasp, RunVasprun
from .ase import PrepFpASE, RunFpASE, ASEInputs


fp_styles = {
    "vasp": {
        "prep": PrepVasp,
        "run": RunVasprun,
        "inputs":VaspInputs
    },
    "ase": {
        "prep": PrepFpASE,
        "run": RunFpASE,
        "inputs": ASEInputs
    },
    "foo":{
        "prep": PrepFoo,
        "run": RunFoo,
        "inputs": FooInputs
    }
}