"""
This is an example to perform CSP based on a reference crystal.
The structures with good matches will be output to *-matched.cif.
Supports molecule input as .xyz or .smi.
"""
from pyxtal import pyxtal
from pyxtal.optimize import WFS, DFS, QRS
from pyxtal.molecule import pyxtal_molecule, compare_mol_connectivity
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gen", dest="gen", type=int, default=1,
                        help="Number of generation, default: 1")
    parser.add_argument("-p", "--pop", dest="pop", type=int, default=10,
                        help="Population size, default: 10")
    parser.add_argument("-n", "--ncpu", dest="ncpu", type=int, default=1,
                        help="cpu number, default: 1")
    parser.add_argument("-a", "--algo", dest="algo", default="WFS",
                        help="algorithm, default: WFS")
    parser.add_argument("--mlp", dest="mlp", default="MACE",
                        help="MLP backend for xyz-only relaxation (MACE/ANI/UMA), default: MACE")
    parser.add_argument("--preopt", dest="preopt", action="store_true",
                        help="preoptimize the lattice and rotation")
    parser.add_argument("--parameters", dest="parameters", default="parameters.xml",
                        help="forcefield parameter xml file, default: parameters.xml")
    parser.add_argument("--ffstyle", dest="ffstyle", default="gaff",
                        help="forcefield style, default: gaff")

    # New: molecule + reference inputs
    parser.add_argument("--mol", dest="mol", default="CTMTNA.xyz",
                        help="Molecule input (.xyz or .smi), default: CTMTNA.xyz")
    parser.add_argument("--smiles", dest="smiles", default=None,
                        help="Optional SMILES for force-field mode; not needed for .xyz MLP-only runs")
    parser.add_argument("--xyz-only", dest="xyz_only", action="store_true",
                        help="Force MLP-only mode (selected automatically for .xyz without --smiles)")
    parser.add_argument("--nconf", dest="nconf", type=int, default=1,
                        help="Number of conformers generated from --smiles, default: 1")
    parser.add_argument("--niter-conf", dest="niter_conf", type=int, default=5,
                        help="Embedding iterations for conformer generation, default: 5")
    parser.add_argument("--conf-tol", dest="conf_tol", type=float, default=0.5,
                        help="RMSD tolerance for unique conformers, default: 0.5")
    parser.add_argument("--uff", dest="use_uff", action="store_true",
                        help="Use UFF (instead of MMFF) in conformer generation")
    parser.add_argument("--seed", dest="seed", default=None,
                        help="Optional reference crystal CIF used only for match checking")
    parser.add_argument("--wdir", dest="wdir", default="ctmtna-simple",
                        help="Working directory, default: ctmtna-simple")
    parser.add_argument("--sg", dest="sg", type=int, nargs="+", default=[61],
                        help="Space group list, default: 61")
    # add CLI arg
    parser.add_argument(
        "--active-sites",
        dest="active_sites",
        type=int,
        nargs=3,
        metavar=("DONOR", "ACCEPTOR", "H"),
        default=None,
        help="Optional atom indices for active sites, e.g. --active-sites 11 12 20",
    )

    options = parser.parse_args()

    # build active_sites only if provided
    active_sites = None
    if options.active_sites is not None:
        d, a, h = options.active_sites
        active_sites = [[d], [a], [h]]

    mol_paths = [x.strip() for x in options.mol.split(",") if len(x.strip()) > 0]
    if len(mol_paths) == 0:
        raise ValueError("Please provide at least one molecule path via --mol")
    mol_ext = os.path.splitext(mol_paths[0])[1].lower()
    for mpath in mol_paths:
        ext = os.path.splitext(mpath)[1].lower()
        if ext != ".smi" and not os.path.exists(mpath):
            raise FileNotFoundError(f"Cannot find molecule file: {mpath}")
    if options.seed is not None and not os.path.exists(options.seed):
        raise FileNotFoundError(f"Cannot find reference CIF: {options.seed}")

    xyz_only = options.xyz_only or (
        options.smiles is None
        and all(os.path.splitext(path)[1].lower() == ".xyz" for path in mol_paths)
    )

    # Optimizer currently requires a valid SMILES string internally
    if xyz_only:
        if options.algo != "WFS":
            raise ValueError("xyz-only mode currently supports --algo WFS only")
        smiles_opt = options.smiles if options.smiles is not None else "xyz_only"
    elif options.smiles is not None:
        smiles_opt = options.smiles
    elif mol_ext == ".smi":
        smiles_opt = os.path.splitext(os.path.basename(mol_paths[0]))[0]
    else:
        raise ValueError(
            "For .xyz input, please also provide --smiles '<SMILES>' "
            "so optimizer torsion/FF initialization can proceed."
        )

    # Build molecule pool:
    # - single geometry from --mol
    # - or multiple conformers generated from --smiles
    if xyz_only:
        mol_objs = [pyxtal_molecule(mpath, active_sites=active_sites) for mpath in mol_paths]
        molecules_for_opt = [mol_objs]
        seed_molecules = [mol_objs[0]]
    elif options.nconf > 1:
        if options.smiles is None:
            raise ValueError("Please provide --smiles when --nconf > 1")
        confs = pyxtal_molecule.get_conformers_from_smiles(
            options.smiles,
            N_iter=options.niter_conf,
            N_conf=options.nconf,
            tol=options.conf_tol,
            use_uff=options.use_uff,
        )
        if len(confs) == 0:
            raise RuntimeError("No conformers were generated from --smiles")
        if active_sites is not None:
            for m in confs:
                m.active_sites = active_sites
        molecules_for_opt = [confs]
        seed_molecules = [confs[0]]
    else:
        m1 = pyxtal_molecule(mol_paths[0], active_sites=active_sites)
        # For xyz input, verify topology matches the SMILES-based template used
        # by optimizer internals (torsion/FF typing).
        if mol_ext == ".xyz":
            m_smi = pyxtal_molecule(smiles_opt + ".smi")
            ok, _ = compare_mol_connectivity(m1.mol, m_smi.mol, ignore_HH=True)
            if not ok:
                raise ValueError(
                    "Input .xyz connectivity does not match --smiles template. "
                    "HTOCSP force-field/torsion setup is SMILES-driven, so xyz-only "
                    "input is not currently sufficient for robust optimization. "
                    "Use --mol '<SMILES>.smi' (or --nconf from SMILES), or provide "
                    "an xyz generated with the same atom/bond topology convention."
                )
        molecules_for_opt = [[m1]]
        seed_molecules = [m1]

    # A reference CIF is optional. It is used for match checking, not for
    # defining the molecule; molecular geometry comes from --mol.
    pmg = None
    if options.seed is not None:
        xtal = pyxtal(molecular=True)
        xtal.from_seed(options.seed, molecules=seed_molecules)
        pmg = xtal.to_pymatgen()

    # Sampling
    fun = globals().get(options.algo)
    if fun is None:
        raise ValueError(f"Unknown algorithm: {options.algo}. Choose from WFS, DFS, QRS.")

    run_kwargs = dict(
        tag="csp_run",
        N_gen=options.gen,
        N_pop=options.pop,
        N_cpu=options.ncpu,
        mlp=options.mlp,
        skip_mlp=False if xyz_only else True,
        ff_style=options.ffstyle,
        ff_parameters=options.parameters,
        molecules=molecules_for_opt,
        xyz_only=xyz_only,
    )
    if options.algo != "QRS":
        run_kwargs["pre_opt"] = options.preopt
    if xyz_only and options.algo == "WFS":
        run_kwargs["fracs"] = [1.0, 0.0]

    go = fun(
        smiles_opt,
        options.wdir,
        options.sg,
        **run_kwargs,
    )

    go.run(ref_pmg=pmg)
    if pmg is not None:
        go.print_matches(header="Ref_match")
    go.plot_results()
