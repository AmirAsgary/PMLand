import sys
sys.path.append('../')
from utils import calculate_aa_frequency_matrix_from_fasta
import os
import numpy as np
import pandas as pd
import argparse
from scipy.special import logsumexp
original_dir = os.getcwd()
os.chdir('/home/amir/amir/hermes/hermes')
sys.path.insert(0, '/home/amir/amir/hermes/hermes')
from hermes.inference import run_hermes_on_pdbfile_or_pyrosetta_pose
os.chdir(original_dir)

def parse_chain_and_sites(value):
    """Parse 'P:1,2,7 A:2,5,7' to [('P', ['1', '2', '7']), ('A', ['2', '5', '7'])]"""
    result = []
    for item in value.split():
        chain, sites = item.split(':')
        sites_list = sites.split(',')
        result.append((chain, sites_list))
    return result

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate amino acid frequency matrix from FASTA")
    parser.add_argument("--input_fasta", type=str, required=True, help="Path to input FASTA file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save output files")
    parser.add_argument("--pdb_path", type=str, required=True, help="Path to PDB file")
    parser.add_argument("--chain_and_sites_list", type=parse_chain_and_sites, required=True, help="Chain and sites, e.g. 'P:1,2,3,4,5 A:4,7,9'")
    parser.add_argument("--prefix", default="", type="str")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    # mpnn scores
    pep_freq = calculate_aa_frequency_matrix_from_fasta(args.input_fasta)
    mpnn_score = np.log(pd.DataFrame(pep_freq, index=list('ACDEFGHIKLMNPQRSTVWY')))
    mpnn_score.to_csv(os.path.join(args.output_path, f'{args.prefix}mpnn_score.csv'))
    # hermes scores
    hermes_df, embeddings = run_hermes_on_pdbfile_or_pyrosetta_pose('hermes_py_050', args.pdb_path, chain_and_sites_list=args.chain_and_sites_list, request=['probas'])
    hermes_df_prob = hermes_df.iloc[:, 5:]
    log_hermes_df_prob = np.log(hermes_df_prob)
    log_hermes_df_prob.columns = [i.split('_')[1] for i in log_hermes_df_prob.columns]
    log_hermes_df_prob = log_hermes_df_prob[list('ACDEFGHIKLMNPQRSTVWY')]
    hermes_score = log_hermes_df_prob.T
    hermes_score.to_csv(os.path.join(args.output_path, f'{args.prefix}hermes_score.csv'))
    # combined score
    log_p1 = mpnn_score.to_numpy()
    log_p2 = hermes_score.to_numpy()
    log_arithmetic_mean = logsumexp(np.stack([log_p1, log_p2]), axis=0) - np.log(2)
    combined_score = pd.DataFrame(log_arithmetic_mean, index=list('ACDEFGHIKLMNPQRSTVWY'))
    combined_score.to_csv(os.path.join(args.output_path, f'{args.prefix}combined_score.csv'))
    print(f"Saved outputs to {args.output_path}")