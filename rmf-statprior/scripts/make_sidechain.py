"""
Side-chain dihedral angles (chi1, chi2) of the Top500 chains.

The PDB entries and chains are taken from the source column of data/top500/aggregated_angles.tsv (e.g. 119L:ChainA:ASN2).
Each entry is downloaded once from RCSB into data/sidechain/pdb/, parsed without external dependencies, and chi1 and
chi2 are computed for the 14 residue types that have both. Only atoms without alternate locations (or altloc A) with
occupancy 1 and B-factor below --max_b are used. Output: data/sidechain/chi12.tsv (source, resname, chi1, chi2 in
degrees).

    python scripts/make_sidechain.py
"""
import argparse, math, os, sys, time, urllib.request
import numpy as np, pandas as pd

CHI = {  # residue: (chi1 atoms, chi2 atoms)
    "ARG": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD")),
    "ASN": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")),
    "ASP": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")),
    "GLN": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD")),
    "GLU": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD")),
    "HIS": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "ND1")),
    "ILE": (("N", "CA", "CB", "CG1"), ("CA", "CB", "CG1", "CD1")),
    "LEU": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "LYS": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD")),
    "MET": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "SD")),
    "PHE": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "PRO": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD")),
    "TRP": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "TYR": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
}
CLASSES = sorted(CHI)


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1; w = b2 - np.dot(b2, b1) * b1
    return math.degrees(math.atan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def parse_pdb(text, chain, max_b):
    """residues of one chain: {(resnum, icode): (resname, {atom: xyz})}, first model only."""
    res = {}
    for line in text.splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM"):
            continue
        if line[21] != chain:
            continue
        alt = line[16]
        if alt not in (" ", "A"):
            continue
        occ = float(line[54:60] or 1); b = float(line[60:66] or 0)
        if occ < 0.99 and alt == " ":
            continue
        if b > max_b:
            continue
        name = line[12:16].strip(); rn = line[17:20]; key = (int(line[22:26]), line[26])
        xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        res.setdefault(key, (rn, {}))[1][name] = xyz
    return res


def chis(res):
    out = []
    for (num, icode), (rn, atoms) in res.items():
        if rn not in CHI:
            continue
        a1, a2 = CHI[rn]
        if all(a in atoms for a in a1 + a2):
            out.append((num, rn, dihedral(*[atoms[a] for a in a1]), dihedral(*[atoms[a] for a in a2])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--angles", default="data/top500/aggregated_angles.tsv")
    ap.add_argument("--out_dir", default="data/sidechain")
    ap.add_argument("--max_b", type=float, default=40.0)
    ap.add_argument("--test_pdb", default=None, help="parse one local PDB file (chain A) and print chi angles")
    a = ap.parse_args()
    if a.test_pdb:
        for row in chis(parse_pdb(open(a.test_pdb).read(), "A", a.max_b))[:8]:
            print(row)
        return
    src = pd.read_csv(a.angles, sep="\t", header=None, usecols=[0])[0].str.split(":", expand=True)
    pairs = sorted(set(zip(src[0], src[1].str.replace("Chain", ""))))
    print(f"{len(pairs)} chains in {len(set(p for p, _ in pairs))} PDB entries", flush=True)
    os.makedirs(os.path.join(a.out_dir, "pdb"), exist_ok=True)
    rows, failed = [], []
    for k, (pdb, chain) in enumerate(pairs):
        path = os.path.join(a.out_dir, "pdb", f"{pdb}.pdb")
        if not os.path.exists(path):
            try:
                urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb}.pdb", path)
                time.sleep(0.05)
            except Exception as e:
                failed.append(pdb); continue
        for num, rn, c1, c2 in chis(parse_pdb(open(path).read(), chain, a.max_b)):
            rows.append((f"{pdb}:Chain{chain}:{rn}{num}", rn, c1, c2))
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(pairs)} chains, {len(rows)} residues", flush=True)
    df = pd.DataFrame(rows, columns=["source", "resname", "chi1", "chi2"])
    df.to_csv(os.path.join(a.out_dir, "chi12.tsv"), sep="\t", index=False)
    print(f"written {len(df)} residues; failed downloads: {len(failed)} {failed[:10]}")
    print(df["resname"].value_counts().to_string())


if __name__ == "__main__":
    main()
