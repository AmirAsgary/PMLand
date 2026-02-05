import sys
sys.path.append('src/')
from utils import compute_contact_mask, read_plddt, read_pae, calculate_aa_frequency_matrix_from_fasta, extract_peptide_from_fasta, get_cpl_score
import os
import numpy as np
import pandas as pd
import json
import argparse
import shutil
import logging
from datetime import datetime
from scipy.special import logsumexp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

class PeptideScoringPipeline:
    def __init__(self, args):
        self.input_dir = args.input_dir
        self.output_dir = args.output_dir
        self.input_df_path = args.df
        self.args = args
        self.failed_jobs = []
        assert 'protienmpnn' in os.listdir(self.input_dir), f'{self.input_dir} ::: \n{os.listdir(self.input_dir)}'
        assert 'alphafold' in os.listdir(self.input_dir), f'{self.input_dir} ::: \n{os.listdir(self.input_dir)}'
        os.makedirs(self.output_dir, exist_ok=True)
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging to both file and console."""
        log_file = os.path.join(self.output_dir, f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Pipeline started. Log file: {log_file}")
        self.logger.info(f"Arguments: {vars(self.args)}")

    def _handle_error(self, error_msg, exception=None):
        """Handle errors based on force_run mode."""
        full_msg = f"{error_msg}: {exception}" if exception else error_msg
        self.failed_jobs.append(full_msg)
        self.logger.error(full_msg)
        if not self.args.force_run:
            raise RuntimeError(full_msg)
        return None

    def _read_df(self, df_path=None):
        path = df_path if df_path else self.input_df_path
        df = pd.read_csv(path)
        required_cols = ['peptide', 'mhc_seq', 'mhc_type', 'anchors', 'id', 'name']
        if not all(col in df.columns for col in required_cols):
            df = pd.read_csv(path, sep='\t')
            if not all(col in df.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df.columns]
                raise ValueError(f"Missing required columns: {missing}")
        self.logger.info(f"Loaded dataframe with {len(df)} rows from {path}")
        return df

    def pae_and_plddt(self, id, mhc_seq, pep_seq, name):
        alphafold_path = os.path.join(self.input_dir, 'alphafold', id)
        pdb = [i for i in os.listdir(alphafold_path) if i.endswith('.pdb')][0]
        plddt = [i for i in os.listdir(alphafold_path) if i.endswith('plddt.npy')][0]
        pae = [i for i in os.listdir(alphafold_path) if i.endswith('predicted_aligned_error.npy')][0]
        pdb_path = os.path.join(alphafold_path, pdb)
        plddt_path = os.path.join(alphafold_path, plddt)
        pae_path = os.path.join(alphafold_path, pae)
        pep_len = len(pep_seq)
        mhc_len = len(mhc_seq)
        radius = self.args.radius
        output_dir = os.path.join(self.output_dir, name, id)
        os.makedirs(output_dir, exist_ok=True)
        contact_mask = compute_contact_mask(pep_seq, pdb_path, radius)
        pae_peptide = read_pae(pae_path, mhc_len, pep_len, contact_mask)
        plddt_peptide = read_plddt(plddt_path, mhc_len, pep_len)
        np.save(os.path.join(output_dir, 'peptide_pae.npy'), pae_peptide)
        np.save(os.path.join(output_dir, 'peptide_plddt.npy'), plddt_peptide)
        self.logger.info(f"Computed PAE/pLDDT for {name}/{id}: PAE_mean={np.mean(pae_peptide):.3f}, pLDDT_mean={np.mean(plddt_peptide):.3f}")
        return pae_peptide, plddt_peptide

    def score_matrix(self, id, peptide, name):
        protienmpnn_path = os.path.join(self.input_dir, 'protienmpnn', id)
        comb_scores = []
        output_path = os.path.join(self.output_dir, name, id)
        os.makedirs(output_path, exist_ok=True)
        for path in os.listdir(protienmpnn_path):
            fasta_dir = os.path.join(protienmpnn_path, path, 'peptide_design', 'seqs')
            fasta_files = [i for i in os.listdir(fasta_dir) if i.endswith('.fa')]
            if not fasta_files:
                continue
            fasta_path = os.path.join(fasta_dir, fasta_files[0])
            pdb_dir = os.path.join(protienmpnn_path, path, 'multichain_pdb')
            pdb_files = os.listdir(pdb_dir)
            if not pdb_files:
                continue
            pdb_path = os.path.join(pdb_dir, pdb_files[0])
            chain_and_sites_list = 'P:' + ','.join([str(i) for i in range(1, len(peptide) + 1)])
            prefix = path + '_'
            pep_freq = calculate_aa_frequency_matrix_from_fasta(fasta_path)
            mpnn_score = np.log(pd.DataFrame(pep_freq, index=list('ACDEFGHIKLMNPQRSTVWY')))
            mpnn_score.to_csv(os.path.join(output_path, f'{prefix}mpnn_score.csv'))
            try:
                original_dir = os.getcwd()
                hermes_path = '/home/amir/amir/hermes/hermes'
                if os.path.exists(hermes_path):
                    os.chdir(hermes_path)
                    sys.path.insert(0, hermes_path)
                from hermes.inference import run_hermes_on_pdbfile_or_pyrosetta_pose
                os.chdir(original_dir)
                chain_sites = self._parse_chain_and_sites(chain_and_sites_list)
                hermes_df, _ = run_hermes_on_pdbfile_or_pyrosetta_pose('hermes_py_050', pdb_path, chain_and_sites_list=chain_sites, request=['probas'])
                hermes_df_prob = hermes_df.iloc[:, 5:]
                log_hermes_df_prob = np.log(hermes_df_prob)
                log_hermes_df_prob.columns = [i.split('_')[1] for i in log_hermes_df_prob.columns]
                log_hermes_df_prob = log_hermes_df_prob[list('ACDEFGHIKLMNPQRSTVWY')]
                hermes_score = log_hermes_df_prob.T
                hermes_score.to_csv(os.path.join(output_path, f'{prefix}hermes_score.csv'))
                log_p1 = mpnn_score.to_numpy()
                log_p2 = hermes_score.to_numpy()
                log_arithmetic_mean = logsumexp(np.stack([log_p1, log_p2]), axis=0) - np.log(2)
                combined_score = pd.DataFrame(log_arithmetic_mean, index=list('ACDEFGHIKLMNPQRSTVWY'))
                combined_score.to_csv(os.path.join(output_path, f'{prefix}combined_score.csv'))
                comb_scores.append(combined_score.to_numpy())
                self.logger.info(f"Computed MPNN+Hermes scores for {name}/{id}/{path}")
            except ImportError:
                self.logger.warning(f"Hermes not available, using MPNN score only for {id}/{path}")
                mpnn_score.to_csv(os.path.join(output_path, f'{prefix}combined_score.csv'))
                comb_scores.append(mpnn_score.to_numpy())
        if len(comb_scores) == 0:
            self.logger.warning(f"No scores computed for {name}/{id}")
            return None
        if len(comb_scores) > 1:
            log_arithmetic_mean = logsumexp(np.stack(comb_scores), axis=0) - np.log(len(comb_scores))
            combined_score = pd.DataFrame(log_arithmetic_mean, index=list('ACDEFGHIKLMNPQRSTVWY'))
            combined_score.to_csv(os.path.join(output_path, 'combined_score.csv'))
        else:
            combined_score = pd.DataFrame(comb_scores[0], index=list('ACDEFGHIKLMNPQRSTVWY'))
            combined_score.to_csv(os.path.join(output_path, 'combined_score.csv'))
        self.logger.info(f"Saved combined score matrix for {name}/{id}")
        return combined_score

    def _parse_chain_and_sites(self, value):
        """Parse 'P:1,2,7 A:2,5,7' to [('P', ['1', '2', '7']), ('A', ['2', '5', '7'])]"""
        result = []
        for item in value.split():
            chain, sites = item.split(':')
            sites_list = sites.split(',')
            result.append((chain, sites_list))
        return result

    def combine_and_score_peptides(self, df_with_pae_plddt, name):
        """Combine scores and score peptides for a given name."""
        subset_df = df_with_pae_plddt[df_with_pae_plddt['name'] == name].copy()
        if len(subset_df) == 0:
            self.logger.warning(f"No rows found for name={name}")
            return None, None
        min_pae = subset_df['pae'].min()
        output_path = os.path.join(self.output_dir, name)
        os.makedirs(output_path, exist_ok=True)
        if self.args.pae_threshold:
            subset_df = subset_df[subset_df['pae'] <= self.args.pae_threshold]
            self.logger.info(f"Filtered by PAE threshold {self.args.pae_threshold}: {len(subset_df)} rows remaining")
        elif self.args.top_pae_percent:
            assert self.args.top_pae_percent <= 1.0
            subset_df = subset_df.sort_values('pae')
            n_keep = max(int(self.args.top_pae_percent * len(subset_df)), 1)
            subset_df = subset_df.iloc[:n_keep]
            self.logger.info(f"Filtered by top {self.args.top_pae_percent*100}% PAE: {len(subset_df)} rows remaining")
        if len(subset_df) == 0:
            raise ValueError(f"No valid pae for structures found, try using top_pae_percent or adjust pae threshold. Minimum pae: {min_pae}")
        SCORES = {}
        PEPTIDE_POOLS = {}
        for _, row in subset_df.iterrows():
            id = str(row['id'])
            peptide = str(row['peptide'])
            pep_len = len(peptide)
            pep_len_key = str(pep_len)
            comb_path = os.path.join(self.output_dir, name, id, 'combined_score.csv')
            if not os.path.exists(comb_path):
                self.logger.warning(f"combined_score.csv not found for {name}/{id}")
                continue
            comb_score_df = pd.read_csv(comb_path, index_col=0)
            comb_score = comb_score_df.to_numpy()
            protienmpnn_path = os.path.join(self.input_dir, 'protienmpnn', id)
            peptide_pools = []
            for path in os.listdir(protienmpnn_path):
                fasta_dir = os.path.join(protienmpnn_path, path, 'peptide_design', 'seqs')
                fasta_files = [i for i in os.listdir(fasta_dir) if i.endswith('.fa')]
                if not fasta_files:
                    continue
                fasta_path = os.path.join(fasta_dir, fasta_files[0])
                peptides = extract_peptide_from_fasta(fasta_path)
                peptide_pools.extend(peptides)
            peptide_pools = list(set(peptide_pools))
            if pep_len_key in PEPTIDE_POOLS:
                PEPTIDE_POOLS[pep_len_key] = list(set(PEPTIDE_POOLS[pep_len_key] + peptide_pools))
            else:
                PEPTIDE_POOLS[pep_len_key] = peptide_pools
            if pep_len_key in SCORES:
                SCORES[pep_len_key].append(comb_score)
            else:
                SCORES[pep_len_key] = [comb_score]
        FINAL_DF = []
        FINAL_SCORES = {}
        for key in SCORES.keys():
            scores_stack = np.stack(SCORES[key])
            key_comb_score = logsumexp(scores_stack, axis=0) - np.log(scores_stack.shape[0])
            n_positions = key_comb_score.shape[1]
            key_comb_score_df = pd.DataFrame(key_comb_score, columns=[str(i+1) for i in range(n_positions)], index=list('ACDEFGHIKLMNPQRSTVWY'))
            peptide_pool = list(set(PEPTIDE_POOLS[key]))
            self.logger.info(f"Scoring {len(peptide_pool)} unique {key}-mer peptides for {name}")
            pep_scores = []
            for pep in peptide_pool:
                try:
                    _, score = get_cpl_score(pep, key_comb_score_df)
                    pep_scores.append(float(score))
                except Exception as e:
                    self.logger.warning(f"Could not score peptide {pep}: {e}")
                    pep_scores.append(np.nan)
            final_df = pd.DataFrame({
                'scores': pep_scores,
                'peptides': peptide_pool,
                'kmer': [int(key)] * len(peptide_pool),
                'name': [name] * len(peptide_pool)
            })
            FINAL_SCORES[key] = key_comb_score.tolist()
            FINAL_DF.append(final_df)
            if self.args.visualize:
                self._visualize_score_matrix(key_comb_score_df, name, key, output_path)
        if not FINAL_DF:
            return None, None
        FINAL_DF = pd.concat(FINAL_DF, ignore_index=True)
        FINAL_SCORES['aa_index'] = list('ACDEFGHIKLMNPQRSTVWY')
        FINAL_DF.to_csv(os.path.join(output_path, 'peptide_score.csv'), index=False)
        with open(os.path.join(output_path, 'scores.json'), 'w') as f:
            json.dump(FINAL_SCORES, f)
        self.logger.info(f"Saved peptide scores for {name}: {len(FINAL_DF)} peptides")
        return FINAL_DF, FINAL_SCORES

    def _visualize_score_matrix(self, score_df, name, kmer, output_path):
        """Visualize combined score matrix as heatmap."""
        fig, ax = plt.subplots(figsize=(max(8, score_df.shape[1] * 0.8), 10))
        sns.heatmap(score_df, cmap='viridis', annot=True if score_df.shape[1] <= 15 else False, fmt='.2f', ax=ax, cbar_kws={'label': 'Log Probability'})
        ax.set_xlabel('Position', fontsize=12)
        ax.set_ylabel('Amino Acid', fontsize=12)
        ax.set_title(f'Combined Score Matrix - {name} ({kmer}-mer)', fontsize=14)
        plt.tight_layout()
        save_path = os.path.join(output_path, f'score_matrix_{kmer}mer.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved score matrix visualization: {save_path}")

    def _visualize_pae_plddt_distributions(self, df):
        """Visualize PAE and pLDDT distributions for each name."""
        names = df['name'].unique()
        n_names = len(names)
        if n_names == 0:
            return
        fig, axes = plt.subplots(2, max(n_names, 1), figsize=(max(5 * n_names, 6), 8), squeeze=False)
        for i, name in enumerate(names):
            subset = df[df['name'] == name]
            # PAE distribution
            ax_pae = axes[0, i]
            if len(subset) == 1:
                ax_pae.bar([0], subset['pae'].values, color='steelblue', width=0.5)
                ax_pae.set_xticks([0])
                ax_pae.set_xticklabels([subset['id'].values[0]], rotation=45, ha='right', fontsize=8)
            else:
                sns.histplot(subset['pae'].dropna(), kde=True if len(subset) > 2 else False, ax=ax_pae, color='steelblue')
            ax_pae.set_xlabel('PAE (mean)', fontsize=10)
            ax_pae.set_ylabel('Count' if len(subset) > 1 else 'Value', fontsize=10)
            ax_pae.set_title(f'{name}\nPAE Distribution (n={len(subset)})', fontsize=11)
            ax_pae.axvline(subset['pae'].mean(), color='red', linestyle='--', label=f'Mean: {subset["pae"].mean():.2f}')
            ax_pae.legend(fontsize=8)
            # pLDDT distribution
            ax_plddt = axes[1, i]
            if len(subset) == 1:
                ax_plddt.bar([0], subset['plddt'].values, color='forestgreen', width=0.5)
                ax_plddt.set_xticks([0])
                ax_plddt.set_xticklabels([subset['id'].values[0]], rotation=45, ha='right', fontsize=8)
            else:
                sns.histplot(subset['plddt'].dropna(), kde=True if len(subset) > 2 else False, ax=ax_plddt, color='forestgreen')
            ax_plddt.set_xlabel('pLDDT (mean)', fontsize=10)
            ax_plddt.set_ylabel('Count' if len(subset) > 1 else 'Value', fontsize=10)
            ax_plddt.set_title(f'{name}\npLDDT Distribution (n={len(subset)})', fontsize=11)
            ax_plddt.axvline(subset['plddt'].mean(), color='red', linestyle='--', label=f'Mean: {subset["plddt"].mean():.2f}')
            ax_plddt.legend(fontsize=8)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'pae_plddt_distributions.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved PAE/pLDDT distribution plot: {save_path}")

    def _visualize_pae_vs_plddt_scatter(self, df):
        """Scatter plot of PAE vs pLDDT colored by name."""
        fig, ax = plt.subplots(figsize=(8, 6))
        names = df['name'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
        for i, name in enumerate(names):
            subset = df[df['name'] == name]
            ax.scatter(subset['pae'], subset['plddt'], label=name, alpha=0.7, s=60, c=[colors[i]], edgecolors='black', linewidths=0.5)
        ax.set_xlabel('PAE (mean)', fontsize=12)
        ax.set_ylabel('pLDDT (mean)', fontsize=12)
        ax.set_title('PAE vs pLDDT by Name', fontsize=14)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'pae_vs_plddt_scatter.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved PAE vs pLDDT scatter plot: {save_path}")

    def _visualize_peptide_score_distribution(self, all_results_df):
        """Visualize peptide score distributions."""
        if all_results_df is None or len(all_results_df) == 0:
            return
        names = all_results_df['name'].unique()
        n_names = len(names)
        fig, axes = plt.subplots(1, max(n_names, 1), figsize=(max(5 * n_names, 6), 5), squeeze=False)
        for i, name in enumerate(names):
            subset = all_results_df[all_results_df['name'] == name]
            ax = axes[0, i]
            if len(subset) == 1:
                ax.bar([0], subset['scores'].values, color='coral', width=0.5)
                ax.set_xticks([0])
                ax.set_xticklabels([subset['peptides'].values[0]], rotation=45, ha='right', fontsize=8)
            else:
                sns.histplot(subset['scores'].dropna(), kde=True if len(subset) > 2 else False, ax=ax, color='coral')
            ax.set_xlabel('Peptide Score', fontsize=10)
            ax.set_ylabel('Count' if len(subset) > 1 else 'Value', fontsize=10)
            ax.set_title(f'{name}\nScore Distribution (n={len(subset)})', fontsize=11)
            if len(subset) > 0:
                ax.axvline(subset['scores'].mean(), color='red', linestyle='--', label=f'Mean: {subset["scores"].mean():.2f}')
                ax.legend(fontsize=8)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'peptide_score_distributions.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved peptide score distribution plot: {save_path}")

    def _cleanup_intermediate_files(self, df):
        """Remove intermediate files if not in dirty_mode."""
        if self.args.dirty_mode:
            self.logger.info("Dirty mode enabled: keeping all intermediate files")
            return
        self.logger.info("Cleaning up intermediate files...")
        for name in df['name'].unique():
            name_dir = os.path.join(self.output_dir, name)
            if not os.path.exists(name_dir):
                continue
            for id_folder in os.listdir(name_dir):
                id_path = os.path.join(name_dir, id_folder)
                if os.path.isdir(id_path):
                    for fname in os.listdir(id_path):
                        fpath = os.path.join(id_path, fname)
                        if '_mpnn_score.csv' in fname or '_hermes_score.csv' in fname or '_combined_score.csv' in fname:
                            os.remove(fpath)
                            self.logger.debug(f"Removed: {fpath}")
        self.logger.info("Cleanup complete")

    def _write_summary_log(self, df, all_results):
        """Write summary statistics to log."""
        self.logger.info("=" * 60)
        self.logger.info("PIPELINE SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total samples processed: {len(df)}")
        self.logger.info(f"Unique names: {df['name'].nunique()}")
        valid_pae = df['pae'].dropna()
        valid_plddt = df['plddt'].dropna()
        if len(valid_pae) > 0:
            self.logger.info(f"PAE - Mean: {valid_pae.mean():.3f}, Std: {valid_pae.std():.3f}, Min: {valid_pae.min():.3f}, Max: {valid_pae.max():.3f}")
        if len(valid_plddt) > 0:
            self.logger.info(f"pLDDT - Mean: {valid_plddt.mean():.3f}, Std: {valid_plddt.std():.3f}, Min: {valid_plddt.min():.3f}, Max: {valid_plddt.max():.3f}")
        if all_results is not None and len(all_results) > 0:
            valid_scores = all_results['scores'].dropna()
            self.logger.info(f"Total peptides scored: {len(all_results)}")
            if len(valid_scores) > 0:
                self.logger.info(f"Score - Mean: {valid_scores.mean():.3f}, Std: {valid_scores.std():.3f}")
        if self.failed_jobs:
            self.logger.info("-" * 60)
            self.logger.info(f"FAILED JOBS ({len(self.failed_jobs)}):")
            for job in self.failed_jobs:
                self.logger.info(f"  - {job}")
        else:
            self.logger.info("All jobs completed successfully!")
        self.logger.info("=" * 60)

    def run(self, df_path=None):
        df = self._read_df(df_path)
        PAE = []
        PLDDT = []
        self.logger.info(f"Processing {len(df)} rows...")
        for idx, row in df.iterrows():
            id = row['id']
            mhc_seq = row['mhc_seq'].replace('/', '')
            pep_seq = row['peptide']
            name = row['name']
            self.logger.info(f"Processing [{idx+1}/{len(df)}]: {name}/{id}")
            try:
                pae, plddt = self.pae_and_plddt(id, mhc_seq, pep_seq, name)
                pae_mean, plddt_mean = np.mean(pae), np.mean(plddt)
                self.score_matrix(id, pep_seq, name)
            except Exception as e:
                self._handle_error(f"Error processing {name}/{id}", e)
                pae_mean, plddt_mean = np.nan, np.nan
            PAE.append(pae_mean)
            PLDDT.append(plddt_mean)
        df['pae'] = PAE
        df['plddt'] = PLDDT
        df[['pae', 'plddt', 'id', 'name']].to_csv(os.path.join(self.output_dir, 'pae_plddt_filtering.csv'), index=False)
        self.logger.info(f"Saved PAE/pLDDT results to {os.path.join(self.output_dir, 'pae_plddt_filtering.csv')}")
        if self.args.visualize:
            self.logger.info("Generating PAE/pLDDT visualizations...")
            self._visualize_pae_plddt_distributions(df)
            self._visualize_pae_vs_plddt_scatter(df)
        all_results = []
        for name in df['name'].unique():
            self.logger.info(f"Combining scores for {name}...")
            try:
                final_df, final_scores = self.combine_and_score_peptides(df, name)
                if final_df is not None:
                    all_results.append(final_df)
            except Exception as e:
                self._handle_error(f"Error combining scores for {name}", e)
        combined_results = None
        if all_results:
            combined_results = pd.concat(all_results, ignore_index=True)
            combined_results.to_csv(os.path.join(self.output_dir, 'all_peptide_scores.csv'), index=False)
            self.logger.info(f"Saved combined results to {os.path.join(self.output_dir, 'all_peptide_scores.csv')}")
            if self.args.visualize:
                self._visualize_peptide_score_distribution(combined_results)
        self._cleanup_intermediate_files(df)
        self._write_summary_log(df, combined_results)
        self.logger.info("Pipeline complete!")
        return df

def parse_args():
    parser = argparse.ArgumentParser(description="Peptide scoring and evaluation pipeline")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing protienmpnn and alphafold folders")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory folder")
    parser.add_argument("--df", type=str, required=True, help="Path to input CSV/TSV file with peptide data")
    parser.add_argument("--radius", type=float, default=10.0, help="Contact radius in Angstroms")
    parser.add_argument("--pae_threshold", type=float, default=None, help="PAE threshold for filtering")
    parser.add_argument("--top_pae_percent", type=float, default=None, help="Top percentage of PAE to keep (0-1)")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--dirty_mode", action="store_true", help="Keep intermediate files in output directory")
    parser.add_argument("--force_run", action="store_true", help="Continue running even if errors occur")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    pipeline = PeptideScoringPipeline(args)
    pipeline.run()