from utils import compute_contact_mask, read_plddt, read_pae
import os
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Process PAE and pLDDT data for peptide-MHC structures")
    parser.add_argument("--pdb_path", type=str, required=True, help="Path to the PDB file")
    parser.add_argument("--pae_path", type=str, required=True, help="Path to the PAE file")
    parser.add_argument("--plddt_path", type=str, required=True, help="Path to the pLDDT file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save results")
    parser.add_argument("--pep_len", type=int, required=True, help="Peptide length")
    parser.add_argument("--mhc_len", type=int, required=True, help="MHC chain length")
    parser.add_argument("--radius", type=float, default=10.0, help="Contact radius in Angstroms")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    contact_mask = compute_contact_mask(args.pep_len, args.pdb_path, args.radius)
    pae_peptide = read_pae(args.pae_path, args.mhc_len, args.pep_len, contact_mask)
    plddt_peptide = read_plddt(args.plddt_path, args.mhc_len, args.pep_len)
    np.save(os.path.join(args.output_dir, 'peptide_pae.npy'), pae_peptide)
    np.save(os.path.join(args.output_dir, 'peptide_plddt.npy'), plddt_peptide)
    print(f"Saved outputs to {args.output_dir}")