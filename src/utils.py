import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy, spearmanr
import random
import subprocess
import tempfile
from multiprocessing import Pool, cpu_count
from pathlib import Path

protmpnn_alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
protmpnn_alpha_2 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
            'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','GAP']
protmpnn_alpha_3 = {s:i for i,s in enumerate(protmpnn_alpha_1)}

def sample_low_likelihood_sequences(reference_seq, seq_similarity_percent, n, log_prob, aa_index, 
                                     seed=None, allowed_positions=None):
    """
    Sample sequences with lower log-likelihood than reference sequence.
    
    Parameters:
    -----------
    reference_seq : str
        Reference peptide sequence (e.g., "AAGIGILTV")
    seq_similarity_percent : float
        Minimum sequence similarity percentage (0-100)
    n : int
        Number of sequences to sample
    log_prob : numpy array
        Log probability matrix (20 amino acids x L positions)
    aa_index : pandas Index
        Amino acid labels (e.g., df.index)
    seed : int, optional
        Random seed for reproducibility. If None, no seed is set.
    allowed_positions : list of int, optional
        List of positions (0-indexed) that are allowed to be mutated.
        If None, all positions can be mutated.
    
    Returns:
    --------
    sequences : list of tuples [(sequence, score, similarity), ...]
    log_probs_list : list of lists - per-position log probabilities for each sequence
    """
    
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    L = len(reference_seq)
    aa_list = aa_index.tolist()
    
    # Determine which positions can be mutated
    if allowed_positions is None:
        allowed_positions = list(range(L))
    else:
        # Validate allowed_positions
        allowed_positions = [p for p in allowed_positions if 0 <= p < L]
        if len(allowed_positions) == 0:
            print(f"Error: No valid positions in allowed_positions")
            return [], []
    
    num_mutable_positions = len(allowed_positions)
    
    # Calculate reference sequence score
    ref_score = 0
    ref_log_probs = []
    for i, aa in enumerate(reference_seq):
        aa_row = aa_index.get_loc(aa)
        log_p = log_prob[aa_row, i]
        ref_score += log_p
        ref_log_probs.append(log_p)
    
    print(f"Reference sequence: {reference_seq}")
    print(f"Reference log-likelihood: {ref_score:.4f}")
    print(f"Reference log-probs: {[f'{x:.3f}' for x in ref_log_probs]}")
    print(f"Allowed positions for mutation: {allowed_positions}")
    
    # Calculate maximum positions we can mutate based on similarity threshold
    min_matches = int(np.ceil(L * seq_similarity_percent / 100))
    max_mutations_by_similarity = L - min_matches
    
    # Limit by available mutable positions
    max_mutations = min(max_mutations_by_similarity, num_mutable_positions)
    
    if max_mutations == 0:
        print(f"Error: Cannot mutate any positions with {seq_similarity_percent}% similarity requirement "
              f"and {num_mutable_positions} allowed positions")
        return [], []
    
    print(f"Max mutations allowed: {max_mutations} (similarity limit: {max_mutations_by_similarity}, "
          f"position limit: {num_mutable_positions})")
    
    sampled_sequences = []
    sampled_log_probs = []
    attempts = 0
    max_attempts = n * 1000  # Prevent infinite loop
    
    while len(sampled_sequences) < n and attempts < max_attempts:
        attempts += 1
        
        # Randomly decide how many positions to mutate (1 to max_mutations)
        num_mutations = np.random.randint(1, max_mutations + 1)
        
        # Randomly select which positions to mutate FROM ALLOWED POSITIONS ONLY
        positions_to_mutate = np.random.choice(allowed_positions, size=num_mutations, replace=False)
        
        # Build the new sequence
        new_seq = list(reference_seq)
        new_score = 0
        new_log_probs = []
        
        for i in range(L):
            if i in positions_to_mutate:
                # Get reference amino acid at this position
                ref_aa = reference_seq[i]
                ref_aa_row = aa_index.get_loc(ref_aa)
                ref_log_prob_val = log_prob[ref_aa_row, i]
                
                # Find amino acids with LOWER log probability (more negative)
                all_log_probs = log_prob[:, i]
                worse_aa_indices = np.where(all_log_probs < ref_log_prob_val)[0]
                
                if len(worse_aa_indices) == 0:
                    # No worse amino acid available at this position, keep reference
                    new_seq[i] = ref_aa
                    log_p = ref_log_prob_val
                    new_score += log_p
                    new_log_probs.append(log_p)
                else:
                    # Sample from worse amino acids, weighted by how much worse they are
                    worse_log_probs = all_log_probs[worse_aa_indices]
                    
                    # Convert to weights: more negative values get higher weight
                    weights = np.max(worse_log_probs) - worse_log_probs + 1e-8
                    weights = weights / np.sum(weights)
                    
                    # Sample an amino acid
                    chosen_idx = np.random.choice(worse_aa_indices, p=weights)
                    new_seq[i] = aa_list[chosen_idx]
                    log_p = all_log_probs[chosen_idx]
                    new_score += log_p
                    new_log_probs.append(log_p)
            else:
                # Keep reference amino acid
                ref_aa_row = aa_index.get_loc(reference_seq[i])
                new_seq[i] = reference_seq[i]
                log_p = log_prob[ref_aa_row, i]
                new_score += log_p
                new_log_probs.append(log_p)
        
        new_seq_str = ''.join(new_seq)
        
        # Check if score is lower than reference (more negative)
        if new_score < ref_score:
            # Calculate actual similarity
            similarity = sum(1 for a, b in zip(reference_seq, new_seq_str) if a == b) / L * 100
            sampled_sequences.append((new_seq_str, new_score, similarity))
            sampled_log_probs.append(new_log_probs)
    
    if attempts >= max_attempts:
        print(f"Warning: Only sampled {len(sampled_sequences)} sequences after {max_attempts} attempts")
    
    print(f"Successfully sampled {len(sampled_sequences)} sequences in {attempts} attempts")
    
    # Sort by score (most negative first) and keep log_probs aligned
    sorted_indices = sorted(range(len(sampled_sequences)), key=lambda i: sampled_sequences[i][1])
    sampled_sequences = [sampled_sequences[i] for i in sorted_indices]
    sampled_log_probs = [sampled_log_probs[i] for i in sorted_indices]
    
    return sampled_sequences, sampled_log_probs




# BLOSUM62 matrix
BLOSUM62 = {
    'A': {'A': 4, 'R': -1, 'N': -2, 'D': -2, 'C': 0, 'Q': -1, 'E': -1, 'G': 0, 'H': -2, 'I': -1, 'L': -1, 'K': -1, 'M': -1, 'F': -2, 'P': -1, 'S': 1, 'T': 0, 'W': -3, 'Y': -2, 'V': 0},
    'R': {'A': -1, 'R': 5, 'N': 0, 'D': -2, 'C': -3, 'Q': 1, 'E': 0, 'G': -2, 'H': 0, 'I': -3, 'L': -2, 'K': 2, 'M': -1, 'F': -3, 'P': -2, 'S': -1, 'T': -1, 'W': -3, 'Y': -2, 'V': -3},
    'N': {'A': -2, 'R': 0, 'N': 6, 'D': 1, 'C': -3, 'Q': 0, 'E': 0, 'G': 0, 'H': 1, 'I': -3, 'L': -3, 'K': 0, 'M': -2, 'F': -3, 'P': -2, 'S': 1, 'T': 0, 'W': -4, 'Y': -2, 'V': -3},
    'D': {'A': -2, 'R': -2, 'N': 1, 'D': 6, 'C': -3, 'Q': 0, 'E': 2, 'G': -1, 'H': -1, 'I': -3, 'L': -4, 'K': -1, 'M': -3, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -4, 'Y': -3, 'V': -3},
    'C': {'A': 0, 'R': -3, 'N': -3, 'D': -3, 'C': 9, 'Q': -3, 'E': -4, 'G': -3, 'H': -3, 'I': -1, 'L': -1, 'K': -3, 'M': -1, 'F': -2, 'P': -3, 'S': -1, 'T': -1, 'W': -2, 'Y': -2, 'V': -1},
    'Q': {'A': -1, 'R': 1, 'N': 0, 'D': 0, 'C': -3, 'Q': 5, 'E': 2, 'G': -2, 'H': 0, 'I': -3, 'L': -2, 'K': 1, 'M': 0, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -2, 'Y': -1, 'V': -2},
    'E': {'A': -1, 'R': 0, 'N': 0, 'D': 2, 'C': -4, 'Q': 2, 'E': 5, 'G': -2, 'H': 0, 'I': -3, 'L': -3, 'K': 1, 'M': -2, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -3, 'Y': -2, 'V': -2},
    'G': {'A': 0, 'R': -2, 'N': 0, 'D': -1, 'C': -3, 'Q': -2, 'E': -2, 'G': 6, 'H': -2, 'I': -4, 'L': -4, 'K': -2, 'M': -3, 'F': -3, 'P': -2, 'S': 0, 'T': -2, 'W': -2, 'Y': -3, 'V': -3},
    'H': {'A': -2, 'R': 0, 'N': 1, 'D': -1, 'C': -3, 'Q': 0, 'E': 0, 'G': -2, 'H': 8, 'I': -3, 'L': -3, 'K': -1, 'M': -2, 'F': -1, 'P': -2, 'S': -1, 'T': -2, 'W': -2, 'Y': 2, 'V': -3},
    'I': {'A': -1, 'R': -3, 'N': -3, 'D': -3, 'C': -1, 'Q': -3, 'E': -3, 'G': -4, 'H': -3, 'I': 4, 'L': 2, 'K': -3, 'M': 1, 'F': 0, 'P': -3, 'S': -2, 'T': -1, 'W': -3, 'Y': -1, 'V': 3},
    'L': {'A': -1, 'R': -2, 'N': -3, 'D': -4, 'C': -1, 'Q': -2, 'E': -3, 'G': -4, 'H': -3, 'I': 2, 'L': 4, 'K': -2, 'M': 2, 'F': 0, 'P': -3, 'S': -2, 'T': -1, 'W': -2, 'Y': -1, 'V': 1},
    'K': {'A': -1, 'R': 2, 'N': 0, 'D': -1, 'C': -3, 'Q': 1, 'E': 1, 'G': -2, 'H': -1, 'I': -3, 'L': -2, 'K': 5, 'M': -1, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -3, 'Y': -2, 'V': -2},
    'M': {'A': -1, 'R': -1, 'N': -2, 'D': -3, 'C': -1, 'Q': 0, 'E': -2, 'G': -3, 'H': -2, 'I': 1, 'L': 2, 'K': -1, 'M': 5, 'F': 0, 'P': -2, 'S': -1, 'T': -1, 'W': -1, 'Y': -1, 'V': 1},
    'F': {'A': -2, 'R': -3, 'N': -3, 'D': -3, 'C': -2, 'Q': -3, 'E': -3, 'G': -3, 'H': -1, 'I': 0, 'L': 0, 'K': -3, 'M': 0, 'F': 6, 'P': -4, 'S': -2, 'T': -2, 'W': 1, 'Y': 3, 'V': -1},
    'P': {'A': -1, 'R': -2, 'N': -2, 'D': -1, 'C': -3, 'Q': -1, 'E': -1, 'G': -2, 'H': -2, 'I': -3, 'L': -3, 'K': -1, 'M': -2, 'F': -4, 'P': 7, 'S': -1, 'T': -1, 'W': -4, 'Y': -3, 'V': -2},
    'S': {'A': 1, 'R': -1, 'N': 1, 'D': 0, 'C': -1, 'Q': 0, 'E': 0, 'G': 0, 'H': -1, 'I': -2, 'L': -2, 'K': 0, 'M': -1, 'F': -2, 'P': -1, 'S': 4, 'T': 1, 'W': -3, 'Y': -2, 'V': -2},
    'T': {'A': 0, 'R': -1, 'N': 0, 'D': -1, 'C': -1, 'Q': -1, 'E': -1, 'G': -2, 'H': -2, 'I': -1, 'L': -1, 'K': -1, 'M': -1, 'F': -2, 'P': -1, 'S': 1, 'T': 5, 'W': -2, 'Y': -2, 'V': 0},
    'W': {'A': -3, 'R': -3, 'N': -4, 'D': -4, 'C': -2, 'Q': -2, 'E': -3, 'G': -2, 'H': -2, 'I': -3, 'L': -2, 'K': -3, 'M': -1, 'F': 1, 'P': -4, 'S': -3, 'T': -2, 'W': 11, 'Y': 2, 'V': -3},
    'Y': {'A': -2, 'R': -2, 'N': -2, 'D': -3, 'C': -2, 'Q': -1, 'E': -2, 'G': -3, 'H': 2, 'I': -1, 'L': -1, 'K': -2, 'M': -1, 'F': 3, 'P': -3, 'S': -2, 'T': -2, 'W': 2, 'Y': 7, 'V': -1},
    'V': {'A': 0, 'R': -3, 'N': -3, 'D': -3, 'C': -1, 'Q': -2, 'E': -2, 'G': -3, 'H': -3, 'I': 3, 'L': 1, 'K': -2, 'M': 1, 'F': -1, 'P': -2, 'S': -2, 'T': 0, 'W': -3, 'Y': -1, 'V': 4}
}

def blosum62_similarity(seq1, seq2):
    """Calculate BLOSUM62 similarity score between two sequences."""
    score = 0
    for aa1, aa2 in zip(seq1, seq2):
        score += BLOSUM62.get(aa1, {}).get(aa2, -4)  # -4 for unknown amino acids
    return score

def calculate_similarity_matrix(sequences):
    """Calculate pairwise BLOSUM62 similarity matrix efficiently."""
    n = len(sequences)
    similarity_matrix = np.zeros((n, n))
    
    # Only calculate upper triangle (matrix is symmetric)
    for i in range(n):
        similarity_matrix[i, i] = blosum62_similarity(sequences[i], sequences[i])
        for j in range(i + 1, n):
            sim = blosum62_similarity(sequences[i], sequences[j])
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim
    
    return similarity_matrix

def cluster_and_select_worst(sequences, scores, num_clusters=None, distance_threshold=None):
    """
    Cluster sequences by BLOSUM62 similarity and select worst score from each cluster.
    
    Parameters:
    -----------
    sequences : list of str
        List of peptide sequences
    scores : list of float
        Corresponding log-likelihood scores for each sequence
    num_clusters : int, optional
        Number of clusters to form. If None, use distance_threshold
    distance_threshold : float, optional
        Distance threshold for forming clusters. If None, use num_clusters
        Recommended: 10-50 depending on sequence similarity
    
    Returns:
    --------
    selected_sequences : list of str
        One sequence per cluster (with worst score)
    selected_scores : list of float
        Corresponding scores
    selected_indices : list of int
        Original indices of selected sequences
    cluster_labels : np.array
        Cluster assignment for each sequence
    """
    
    if len(sequences) == 0:
        return [], [], [], np.array([])
    
    if len(sequences) == 1:
        return [sequences[0]], [scores[0]], [0], np.array([0])
    
    print(f"\nClustering {len(sequences)} sequences...")
    
    # Calculate similarity matrix
    similarity_matrix = calculate_similarity_matrix(sequences)
    print(f"Similarity matrix range: [{similarity_matrix.min():.2f}, {similarity_matrix.max():.2f}]")
    
    # Convert similarity to distance (higher similarity = lower distance)
    # Use max similarity - current similarity
    max_sim = similarity_matrix.max()
    distance_matrix = max_sim - similarity_matrix
    
    # Ensure diagonal is zero (self-distance)
    np.fill_diagonal(distance_matrix, 0)
    
    # Convert to condensed distance matrix for hierarchical clustering
    condensed_dist = squareform(distance_matrix, checks=False)
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_dist, method='average')
    
    # Get cluster labels
    if num_clusters is not None:
        cluster_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
        print(f"Formed {num_clusters} clusters (requested)")
    elif distance_threshold is not None:
        cluster_labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')
        n_clusters = len(np.unique(cluster_labels))
        print(f"Formed {n_clusters} clusters (threshold={distance_threshold})")
    else:
        # Default: use distance threshold of 20
        distance_threshold = 20
        cluster_labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')
        n_clusters = len(np.unique(cluster_labels))
        print(f"Formed {n_clusters} clusters (default threshold={distance_threshold})")
    
    # Select worst (most negative) score from each cluster
    selected_sequences = []
    selected_scores = []
    selected_indices = []
    
    unique_clusters = np.unique(cluster_labels)
    print(f"\nCluster sizes:")
    
    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        cluster_idx = np.where(cluster_mask)[0]
        cluster_scores = [scores[i] for i in cluster_idx]
        
        # Find worst (most negative) score in cluster
        worst_idx_in_cluster = np.argmin(cluster_scores)
        original_idx = cluster_idx[worst_idx_in_cluster]
        
        selected_sequences.append(sequences[original_idx])
        selected_scores.append(scores[original_idx])
        selected_indices.append(original_idx)
        
        print(f"  Cluster {cluster_id}: {len(cluster_idx)} sequences, worst score: {scores[original_idx]:.4f}")
    
    return selected_sequences, selected_scores, selected_indices, cluster_labels


def get_cpl_score(sequence, log_prob_df):
    VAL = []
    for i, s in enumerate(sequence):
        VAL.append(log_prob_df[str(i+1)][s])
    return VAL, np.sum(VAL)

def get_mpnn_score(sequence, log_prob_arr, alphabet=protmpnn_alpha_3):
    VAL = []
    for i, s in enumerate(sequence):
        aa_idx = alphabet[s]
        log_prob = log_prob_arr[i][aa_idx] #(S,20) -> (20) -> 1
        VAL.append(log_prob)
    return VAL, np.sum(VAL)
    

from numba import njit, prange
from numba.typed import List

def sample_peptides(n, k, min_diff=1, constraints=None):
    """
    Sample n peptides of length k with minimum hamming distance min_diff.
    
    Args:
        n: number of peptides to sample
        k: length of peptides
        min_diff: minimum hamming distance between any two peptides
        constraints: dict mapping positions (int) to allowed amino acids (list of chars)
                    e.g., {0: ['A', 'C'], 5: ['G', 'K', 'R']}
    """
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}
    
    # Build allowed indices per position for numba
    allowed_per_pos = List()
    for pos in range(k):
        if constraints and pos in constraints:
            allowed = np.array([aa_to_idx[aa] for aa in constraints[pos]], dtype=np.int8)
        else:
            allowed = np.arange(20, dtype=np.int8)
        allowed_per_pos.append(allowed)
    
    @njit
    def hamming_check(peptides_int, candidate, min_diff):
        for i in range(len(peptides_int)):
            diff = 0
            for j in range(len(candidate)):
                if peptides_int[i, j] != candidate[j]:
                    diff += 1
                    if diff >= min_diff:
                        break
            if diff < min_diff:
                return False
        return True
    
    @njit
    def generate_candidate(allowed_per_pos, k):
        candidate = np.empty(k, dtype=np.int8)
        for pos in range(k):
            allowed = allowed_per_pos[pos]
            candidate[pos] = allowed[np.random.randint(0, len(allowed))]
        return candidate
    
    @njit
    def sample_loop(n, k, min_diff, allowed_per_pos):
        peptides_int = np.empty((n, k), dtype=np.int8)
        count = 0
        while count < n:
            candidate = generate_candidate(allowed_per_pos, k)
            if count == 0 or hamming_check(peptides_int[:count], candidate, min_diff):
                peptides_int[count] = candidate
                count += 1
        return peptides_int
    
    peptides_int = sample_loop(n, k, min_diff, allowed_per_pos)
    peptides = [''.join(amino_acids[i] for i in pep) for pep in peptides_int]
    
    return peptides


def kl_divergence(arr1, arr2, eps=1e-10):
    arr1 = np.clip(arr1, eps, 1)
    arr2 = np.clip(arr2, eps, 1)
    return np.sum(arr1 * np.log(arr1 / arr2), axis=0)



def entropy_difference(arr1, arr2, eps=1e-10):
    """
    Entropy difference per position: H(arr1) - H(arr2)

    Parameters
    ----------
    arr1, arr2 : np.ndarray
        Shape (20, L), categorical probabilities
    eps : float
        Numerical stability

    Returns
    -------
    np.ndarray
        Entropy difference per position, shape (L,)
    """
    arr1 = np.clip(arr1, eps, 1)
    arr2 = np.clip(arr2, eps, 1)

    H1 = -np.sum(arr1 * np.log(arr1), axis=0)
    H2 = -np.sum(arr2 * np.log(arr2), axis=0)

    return H1 - H2
##########################################



def metropolis_accept_logprob(log_prob_current, log_prob_proposed, temperature):
    """
    Metropolis-Hastings for LOG-PROBABILITIES (Higher is Better).
    """
    # 1. Calculate Delta (Proposed - Current)
    # If Proposed (-5.0) > Current (-10.0) --> Delta is +5.0 (Improvement)
    delta_log_prob = log_prob_proposed - log_prob_current
    # 2. Acceptance Logic
    # If the new state is BETTER (Higher LogProb), always accept.
    if delta_log_prob > 0:
        return True
    # 3. If the new state is WORSE (Lower LogProb), accept probabilistically.
    # We want exp(delta / T). 
    # Since delta is negative here, exp(negative) gives a probability < 1.
    acceptance_probability = np.exp(delta_log_prob / temperature)
    random_draw = np.random.uniform(0, 1)
    return random_draw < acceptance_probability

def exponential_cooling(T_initial=1., T_final=0.01, n_steps=1000, current_step=1):
    """
    Exponential decay: T(t) = T_initial * (T_final/T_initial)^(t/n_steps)
    
    This is the most common schedule.
    """
    decay_rate = (T_final / T_initial) ** (1.0 / n_steps)
    return T_initial * (decay_rate ** current_step)

def sample_peptide(mpnn_probs, mpnn_logp, original_seq, best_mutation=True, num_mutation=2, fix_positions=None, log_dict=None, temperature=1, Entropy_thr=2.):
    sequence = []
    if not fix_positions: fix_positions = []
    peptide_length = mpnn_probs.shape[0]
    mpnn_probs, mpnn_logp = mpnn_probs[:,:20], mpnn_logp[:,:20]
    allowed_positions = np.arange(peptide_length)
    if fix_positions: allowed_positions = [i for i in allowed_positions if i not in fix_positions]
    np.random.shuffle(allowed_positions)
    original_seq = np.array(list(original_seq))
    best_original_seq_idx = np.argmax(mpnn_probs, axis=1)
    sequence = original_seq.copy()
    S = entropy(mpnn_probs, axis=1)
    metropolis = False
    original_mpnn_score, original_mpnn_score_sum = get_mpnn_score(original_seq, mpnn_logp)

    if not best_mutation: # MCMC
        while metropolis != True:
            np.random.shuffle(allowed_positions)
            for i in allowed_positions[:num_mutation]:
                aa_idx = np.random.choice(20, p=mpnn_probs[i])
                sequence[i] = protmpnn_alpha_1[aa_idx]
            sequence_mpnn_score, sequence_mpnn_score_sum = get_mpnn_score(''.join(list(sequence)), mpnn_logp)
            metropolis = metropolis_accept_logprob(original_mpnn_score_sum, sequence_mpnn_score_sum, temperature)
        sequence = ''.join(list(sequence))
    
    else: # only select best mutation
        # Sort based on Entropy
        S_arg_sorted = np.argsort(S)
        S_arg_sorted = [i for i in S_arg_sorted if i in allowed_positions]
        number = 0
        for _, pos in enumerate(S_arg_sorted):
            aa_old = original_seq[pos]
            aa_idx = best_original_seq_idx[pos]
            aa_new = protmpnn_alpha_1[aa_idx]
            if aa_old == aa_new or S[pos] > Entropy_thr: 
                fix_positions.append(pos)
                continue
            else:
                number += 1
            sequence[pos] = aa_new
            if number == num_mutation: break
        sequence = ''.join(list(sequence))
        sequence_mpnn_score, _ = get_mpnn_score(sequence, mpnn_logp)
    
    if not log_dict: 
        log_dict = {
            'mpnn_probs': [],
            'fix_positions': [],
            'original_seq': [],
            'original_mpnn_score': [],
            'entropy': [],
            'new_seq': [],
            'new_mpnn_score': []
        }
    log_dict['mpnn_probs'].append([list(i) for i in mpnn_probs])
    log_dict['fix_positions'].append(fix_positions)
    log_dict['original_seq'].append(''.join(list(original_seq)))
    log_dict['original_mpnn_score'].append(original_mpnn_score)
    log_dict['entropy'].append(list(S))
    log_dict['new_seq'].append(sequence)
    log_dict['new_mpnn_score'].append(sequence_mpnn_score)

    return sequence, log_dict, fix_positions


def plot_log_prob_heatmaps(log_prob_df1, log_prob_df2, xticklabels1, xticklabels2, 
                            yticklabels, title1='Log Probabilities', 
                            title2='Log Probabilities 2', figsize=(16, 5), outpath=None):
    """
    Plot three heatmaps: two log probability heatmaps and a delta rank heatmap.
    
    Parameters:
    -----------
    log_prob_df1 : pd.DataFrame
        First log probability dataframe
    log_prob_df2 : pd.DataFrame
        Second log probability dataframe
    xticklabels1 : list
        X-axis labels for first and third heatmaps
    xticklabels2 : list
        X-axis labels for second heatmap
    yticklabels : list
        Y-axis labels (amino acids)
    title1 : str
        Title for first heatmap
    title2 : str
        Title for second heatmap
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # First heatmap - Log probabilities 1
    sns.heatmap(log_prob_df1, cmap='viridis', cbar_kws={'label': 'Log Probability'}, 
                ax=axes[0], fmt='.2f', annot_kws={'size': 8}, annot=True)
    axes[0].set_yticks(np.arange(len(yticklabels)) + 0.5)
    axes[0].set_yticklabels(yticklabels, rotation=0)
    axes[0].set_xticks(np.arange(len(xticklabels1)) + 0.5)
    axes[0].set_xticklabels(xticklabels1, rotation=45, ha='right')
    axes[0].set_xlabel('Columns')
    axes[0].set_ylabel('Amino Acids')
    axes[0].set_title(title1)
    
    # Second heatmap - Log probabilities 2
    sns.heatmap(log_prob_df2, cmap='viridis', annot=True, fmt='.2f', 
                annot_kws={'size': 8}, cbar_kws={'label': 'Log Probability'}, ax=axes[1])
    axes[1].set_xticks(np.arange(len(xticklabels2)) + 0.5)
    axes[1].set_xticklabels(xticklabels2, rotation=45, ha='right')
    axes[1].set_ylabel('Amino Acids')
    axes[1].set_title(title2)
    
    # Calculate ranks (rank 1 = highest value)
    ranks1 = log_prob_df1.rank(ascending=False).astype(int)
    ranks2 = log_prob_df2.rank(ascending=False).astype(int)
    ranks2.index = ranks1.index
    
    # Delta ranks
    delta_ranks = np.abs(ranks1.to_numpy() - ranks2.to_numpy())
    
    # Third heatmap - Delta ranks
    sns.heatmap(delta_ranks, cmap='RdBu_r', center=0, 
                cbar_kws={'label': 'Δ Rank (1st - 2nd)'}, ax=axes[2], 
                annot=True, fmt='d', annot_kws={'size': 8})
    axes[2].set_yticks(np.arange(len(yticklabels)) + 0.5)
    axes[2].set_yticklabels(yticklabels, rotation=0)
    axes[2].set_xticks(np.arange(len(xticklabels1)) + 0.5)
    axes[2].set_xticklabels(xticklabels1, rotation=45, ha='right')
    axes[2].set_xlabel('Columns')
    axes[2].set_ylabel('Amino Acids')
    axes[2].set_title('Δ Rank (positive = higher rank in 1st)')
    
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath)
    plt.show()
    
    return fig, axes



def top5_overlap_fraction(ref, tgt, t=3):
    """
    Parameters
    ----------
    ref : np.ndarray or pd.DataFrame
        Shape (20, L), reference log-probabilities
    tgt : np.ndarray or pd.DataFrame
        Shape (20, L), target log-probabilities

    Returns
    -------
    np.ndarray
        Array of length L with overlap fraction per position
    """
    # Convert to numpy (works for DataFrame or ndarray)
    ref = np.asarray(ref)
    tgt = np.asarray(tgt)

    assert ref.shape == tgt.shape, "Reference and target must have same shape"
    assert ref.shape[0] == 20, "Expected 20 amino acids (rows)"

    L = ref.shape[1]
    overlap = np.zeros(L)

    for i in range(L):
        ref_top5 = set(np.argsort(ref[:, i])[:t])
        tgt_top5 = set(np.argsort(tgt[:, i])[:t])

        overlap[i] = len(ref_top5 & tgt_top5) / 5.0

    return overlap


@njit
def _sample_peptides_numba(probs, b):
    """
    Core sampling function with numba.
    Returns indices array of shape (b, n_pos).
    """
    n_aa, n_pos = probs.shape
    result = np.empty((b, n_pos), dtype=np.int32)
    
    for i in range(b):
        for pos in range(n_pos):
            # Sample from categorical distribution
            r = np.random.random()
            cumsum = 0.0
            for aa in range(n_aa):
                cumsum += probs[aa, pos]
                if r < cumsum:
                    result[i, pos] = aa
                    break
            else:
                result[i, pos] = n_aa - 1
    
    return result


def sample_peptides_prob(arr_freq, b, amino_acids='ACDEFGHIKLMNPQRSTVWY', constraints=None):
    """
    Sample peptides based on position-wise amino acid probabilities.
    
    Parameters:
    -----------
    arr_freq : np.ndarray
        Frequency array of shape (n_aa, n_pos).
    b : int
        Number of peptides to sample.
    amino_acids : str
        String of amino acids corresponding to arr_freq rows.
    constraints : dict, optional
        Dictionary where keys are positions (0-indexed) and values are strings
        of allowed amino acids for that position.
        Example: {0: 'AC', 2: 'DEF'}
    
    Returns:
    --------
    list of str
        Sampled peptide sequences.
    """
    aa_list = list(amino_acids)
    
    # Apply constraints by zeroing out disallowed amino acids
    if constraints is not None:
        arr_freq = arr_freq.copy()  # Don't modify original
        aa_to_idx = {aa: idx for idx, aa in enumerate(aa_list)}
        for pos, allowed_aas in constraints.items():
            allowed_indices = {aa_to_idx[aa] for aa in allowed_aas if aa in aa_to_idx}
            for idx in range(len(aa_list)):
                if idx not in allowed_indices:
                    arr_freq[idx, pos] = 0.0
    
    # Normalize to probabilities (after constraints applied)
    probs = arr_freq / arr_freq.sum(axis=0, keepdims=True)
    
    indices = _sample_peptides_numba(probs, b)
    peptides = [''.join(aa_list[idx] for idx in row) for row in indices]
    
    return peptides



@njit
def _sample_best_peptides_numba(top_indices, top_probs, b, n_pos, top_k, 
                                constraint_indices, constraint_counts, seed):
    """
    top_indices: (n_pos, top_k)
    top_probs: (n_pos, top_k)
    constraint_indices: (n_pos, max_allowed_aas) - padded with -1
    constraint_counts: (n_pos,) - how many allowed AAs at this pos (0 if no constraint)
    seed: random seed (-1 means no seeding)
    """
    if seed >= 0:
        np.random.seed(seed)
    
    result = np.empty((b, n_pos), dtype=np.int32)
    
    for i in range(b):
        for pos in range(n_pos):
            # FAST PATH: If position is constrained, pick randomly from the list
            num_allowed = constraint_counts[pos]
            if num_allowed > 0:
                # Purely random choice from the constrained list
                r_idx = np.random.randint(0, num_allowed)
                result[i, pos] = constraint_indices[pos, r_idx]
            
            # SLOW PATH: Standard top-k sampling
            else:
                r = np.random.random()
                cumsum = 0.0
                for j in range(top_k):
                    cumsum += top_probs[pos, j]
                    if r < cumsum:
                        result[i, pos] = top_indices[pos, j]
                        break
                else:
                    result[i, pos] = top_indices[pos, -1]
    return result


def sample_best_peptides(arr_freq, b, amino_acids='ACDEFGHIKLMNPQRSTVWY', top_k=3, 
                         unique=True, constraints=None, seed=None):
    """
    Sample peptides from frequency array.
    
    Args:
        arr_freq: (n_aa, n_pos) frequency array
        b: number of peptides to sample
        amino_acids: amino acid alphabet
        top_k: number of top amino acids to consider at each position
        unique: if True, return only unique peptides
        constraints: dict mapping position -> list of allowed amino acids
        seed: random seed for reproducibility (None for no seeding)
    
    Returns:
        list of peptide strings
    """
    n_aa, n_pos = arr_freq.shape
    aa_list = list(amino_acids)
    aa_to_idx = {aa: idx for idx, aa in enumerate(aa_list)}

    # --- PRE-PROCESS CONSTRAINTS ---
    constraint_counts = np.zeros(n_pos, dtype=np.int32)
    max_c = max([len(v) for v in constraints.values()]) if constraints else 1
    constraint_indices = np.full((n_pos, max_c), -1, dtype=np.int32)

    if constraints:
        for pos, allowed_aas in constraints.items():
            idxs = [aa_to_idx[a] for a in allowed_aas if a in aa_to_idx]
            constraint_counts[pos] = len(idxs)
            for j, val in enumerate(idxs):
                constraint_indices[pos, j] = val

    # --- PRE-CALCULATE TOP-K ---
    top_indices = np.empty((n_pos, top_k), dtype=np.int32)
    top_probs = np.empty((n_pos, top_k), dtype=np.float64)

    for pos in range(n_pos):
        col = arr_freq[:, pos]
        t_idx = np.argsort(col)[-top_k:]
        top_indices[pos, :] = t_idx
        probs = col[t_idx]
        p_sum = np.sum(probs)
        top_probs[pos, :] = probs / (p_sum if p_sum > 0 else 1.0)

    # Convert seed: None -> -1 for numba (no seeding)
    numba_seed = seed if seed is not None else -1

    # --- EXECUTE ---
    if unique:
        peptides = set()
        batch_count = 0
        while len(peptides) < b:
            batch_size = min(b * 2, 10000)
            # Increment seed for each batch to avoid duplicates but maintain reproducibility
            current_seed = (numba_seed + batch_count) if numba_seed >= 0 else -1
            indices = _sample_best_peptides_numba(top_indices, top_probs, batch_size, n_pos, top_k, 
                                                   constraint_indices, constraint_counts, current_seed)
            for row in indices:
                peptides.add(''.join(aa_list[idx] for idx in row))
                if len(peptides) >= b: 
                    break
            batch_count += 1
        return list(peptides)[:b]
    else:
        indices = _sample_best_peptides_numba(top_indices, top_probs, b, n_pos, top_k, 
                                               constraint_indices, constraint_counts, numba_seed)
        return [''.join(aa_list[idx] for idx in row) for row in indices]
    
import numpy as np
import pandas as pd

def combine_log_probs_poe(log_distributions, weights=None, amino_acids=None):
    """
    Combines log-probability distributions using the Product of Experts logic.
    
    Args:
        log_distributions (list): List of DataFrames or np.arrays (20, L) of log-probs.
        weights (list, optional): Scalar weights for each expert model.
        amino_acids (list, optional): Required if passing numpy arrays.
        
    Returns:
        pd.DataFrame: Normalized probability distribution of shape (20, L).
    """
    weighted_logs = []
    
    for i, dist in enumerate(log_distributions):
        # Extract data and labels
        if isinstance(dist, pd.DataFrame):
            if amino_acids is None: amino_acids = dist.index.tolist()
            arr = dist.to_numpy()
        else:
            if amino_acids is None:
                raise ValueError("List of amino acids required for numpy array inputs.")
            arr = np.array(dist)
        
        # PoE logic: Weighted addition of log probabilities
        w = weights[i] if weights is not None else 1.0
        weighted_logs.append(w * arr)

    # Summing logs is equivalent to multiplying probabilities
    total_log_p = np.sum(weighted_logs, axis=0)

    # Normalize across the amino acid axis (axis 0) using Log-Sum-Exp trick
    # This converts the combined log-scores back into a valid [0, 1] distribution
    max_val = np.max(total_log_p, axis=0)
    exp_p = np.exp(total_log_p - max_val)
    normalized_p = exp_p / np.sum(exp_p, axis=0)

    return pd.DataFrame(
        normalized_p, 
        index=amino_acids, 
        columns=[f"pos_{j+1}" for j in range(normalized_p.shape[1])]
    )


METHOD_COLORS = {
    'Hermes (τ=0.00)': '#1f77b4',
    'Hermes (τ=0.50)': '#ff7f0e',
    'MPNN Cond (all)': '#2ca02c',
    'MPNN Cond (obo)': '#d62728',
    'MPNN Uncond': '#9467bd'
}


def plot_scatter_methods(methods, ps_cpl, title=''):
    """
    Plot scatter plots for each method vs PS-CPL scores.
    """
    fig, axes = plt.subplots(1, 7, figsize=(22, 4))
    
    for idx, (name, scores_arr) in enumerate(methods.items()):
        ax = axes[idx]
        rho, pval = spearmanr(scores_arr, ps_cpl)
        ax.scatter(scores_arr, ps_cpl, alpha=0.5, s=25, edgecolors="black", 
                   linewidths=0.3, color=METHOD_COLORS.get(name, 'gray'))
        ax.set_xlabel(f"{name} score", fontsize=11)
        ax.set_ylabel("PS-CPL score", fontsize=11)
        ax.set_title(f"{name}\nρ = {rho:.3f}", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    return fig




@njit(parallel=True)
def calculate_aa_frequency_matrix_njit(peptides, aa_order):
    """
    Calculate the frequency matrix using Numba njit for millions of sequences.
    
    Parameters:
    peptides (np.ndarray): 2D array of integers (each row is a peptide, each col is a position).
    aa_order (str): String of amino acid order (20 chars).
    
    Returns:
    np.ndarray: Frequency matrix of shape (20, length).
    """
    n_seqs, length = peptides.shape
    n_aa = len(aa_order)
    freq_matrix = np.zeros((n_aa, length), dtype=np.float32)
    
    for pos in range(length):
        for seq_idx in range(n_seqs):
            aa_idx = peptides[seq_idx, pos]
            if 0 <= aa_idx < n_aa:
                freq_matrix[aa_idx, pos] += 1
    
    freq_matrix /= n_seqs
    return freq_matrix

def calculate_aa_frequency_matrix(peptides):
    if not peptides:
        return np.zeros((20, 0))
    
    length = len(peptides[0])
    aa_order = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_idx = {aa: i for i, aa in enumerate(aa_order)}
    
    # Convert peptides to a 2D NumPy array of indices
    seq_array = np.array([[aa_to_idx.get(aa, -1) for aa in pep] for pep in peptides], dtype=np.int8)
    
    return calculate_aa_frequency_matrix_njit(seq_array, aa_order)

def calculate_aa_frequency_matrix_from_fasta(fasta_file):
    with open(fasta_file, 'r') as f:
        peptide_list = []
        for line in f:
            if line.startswith('>'): pass
            else: peptide_list.append(line.strip())
    pep_freq = calculate_aa_frequency_matrix(peptide_list)
    #pairwise = calculate_pairwise_cooccurrence(peptide_list)
    return pep_freq #, pairwise 


def extract_peptide_from_fasta(fasta_file):
    with open(fasta_file, 'r') as f:
        peptide_list = []
        for line in f:
            if line.startswith('>'): pass
            else: peptide_list.append(line.strip())

    return peptide_list

@njit(parallel=True)
def calculate_pairwise_cooccurrence_tensor_njit(peptides, n_aa=20):
    """
    Calculate pairwise co-occurrence tensor using njit.
    Returns a 4D tensor (pos1, pos2, aa1, aa2).
    
    Parameters:
    peptides (np.ndarray): 2D array of integers (n_seqs, length).
    n_aa (int): Number of amino acids (20).

    Returns:
    np.ndarray: 4D tensor of shape (length, length, n_aa, n_aa).
    """
    n_seqs, length = peptides.shape
    # Shape: (pos1, pos2, aa1, aa2)
    cooccurrence_tensor = np.zeros((length, length, n_aa, n_aa), dtype=np.float32)
    
    for pos1 in range(length):
        # We process the upper triangle. 
        # If you need the full symmetric matrix, you can mirror it later or calculate both loops.
        # Here we calculate only pos2 >= pos1.
        for pos2 in prange(pos1, length):  
            for seq_idx in range(n_seqs):
                idx1 = peptides[seq_idx, pos1]
                idx2 = peptides[seq_idx, pos2]
                if 0 <= idx1 < n_aa and 0 <= idx2 < n_aa:
                    cooccurrence_tensor[pos1, pos2, idx1, idx2] += 1
            
            # Normalize in place
            if n_seqs > 0:
                factor = 1.0 / n_seqs
                for i in range(n_aa):
                    for j in range(n_aa):
                        cooccurrence_tensor[pos1, pos2, i, j] *= factor

    # Optionally fill the lower triangle for symmetry:
    # cooccurrence_tensor[pos2, pos1] = cooccurrence_tensor[pos1, pos2]
    # (But usually only the upper triangle or specific pairs are needed).
                        
    return cooccurrence_tensor

def calculate_pairwise_cooccurrence(peptides):
    """
    Calculates pairwise co-occurrence of amino acids across all position pairs.
    
    Returns:
    np.ndarray: A 4D tensor of shape (L, L, 20, 20).
                tensor[i, j, a, b] is the probability of amino acid 'a' at position 'i'
                and amino acid 'b' at position 'j'.
    """
    if not peptides:
        return np.zeros((0, 0, 20, 20))
    
    length = len(peptides[0])
    aa_order = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_idx = {aa: i for i, aa in enumerate(aa_order)}
    
    # Convert peptides to 2D integer array
    seq_array = np.array([[aa_to_idx.get(aa, -1) for aa in pep] for pep in peptides], dtype=np.int8)
    
    # Call jitted function
    tensor_4d = calculate_pairwise_cooccurrence_tensor_njit(seq_array, n_aa=20)
    
    return tensor_4d



from Bio.PDB import PDBParser
import numpy as np
import os

# Custom three-letter to one-letter amino acid code conversion
THREE_TO_ONE = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    # Non-standard/modified amino acids
    'MSE': 'M',  # Selenomethionine
    'SEC': 'U',  # Selenocysteine
    'PYL': 'O',  # Pyrrolysine
}

def three_to_one(three_letter_code):
    """Convert three-letter amino acid code to one-letter code."""
    return THREE_TO_ONE.get(three_letter_code.upper(), 'X')


def compute_contact_mask(peptide, pdb_file, radius):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    ca_atoms = []
    residue_names = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    ca_atoms.append(residue['CA'].get_coord())
                    residue_names.append(residue.get_resname())
    
    ca_coords = np.array(ca_atoms)
    
    diff = ca_coords[:, np.newaxis, :] - ca_coords[np.newaxis, :, :]
    distance_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
    if isinstance(peptide, int): n = peptide
    else:
        n = len(peptide)
    last_n_resnames = residue_names[-n:]
    last_n_sequence = ''.join([three_to_one(res) for res in last_n_resnames])
    
    if last_n_sequence != peptide:
        raise ValueError(f"Peptide '{peptide}' does not match last {n} residues '{last_n_sequence}' in {pdb_file}")
    
    peptide_distances = distance_matrix
    min_distances = np.min(peptide_distances, axis=0)
    contact_mask = (min_distances <= radius).astype(int)
    
    return contact_mask


def read_pae(pae_path, mhc_seq, peptide_seq, contact_mask):
    pae = np.load(pae_path)
    if isinstance(mhc_seq, int): mhc_len = mhc_seq
    else: mhc_len = len(mhc_seq)
    if isinstance(peptide_seq, int): pep_len = peptide_seq
    else: pep_len = len(peptide_seq)
    pae = pae[:mhc_len+pep_len, :mhc_len+pep_len]
    pae = pae * contact_mask
    p1 = pae[mhc_len:mhc_len+pep_len , :] #(p, N))
    p2 = pae[: , mhc_len:mhc_len+pep_len] #(N, p)
    pp = (p1.transpose() + p2)/2 #(N,p)
    pp = np.mean(pp, axis=0) #(p,)
    return pp


def read_plddt(plddt_path, mhc_seq, peptide_seq):
    plddt = np.load(plddt_path)
    if isinstance(mhc_seq, int): mhc_len = mhc_seq
    else: mhc_len = len(mhc_seq)
    if isinstance(peptide_seq, int): pep_len = peptide_seq
    else: pep_len = len(peptide_seq)
    plddt = plddt[:mhc_len+pep_len]
    peplddt = plddt[mhc_len:mhc_len+pep_len]
    return peplddt


def compute_contact_masks(df, pdb_dir, radius=10.):
    per_res = []
    all = []
    plddt_per_res = []
    plddt_all = []
    
    for idx, row in df.iterrows():
        peptide = row['peptide']
        pdb_file = row['pdb_file']
        mhc_seq = row['mhc_seq']
        mhc_seq = mhc_seq.replace('/', '')
        if pdb_dir:
            pdb_path = os.path.join(pdb_dir, pdb_file)
        else:
            pdb_path = pdb_file
        pae_path = row['pae_path']
        plddt_path = row['plddt_path']
        
        contact_mask = compute_contact_mask(peptide, pdb_path, radius)
        pae_peptide = read_pae(pae_path, mhc_seq, peptide, contact_mask) #(p,)
        pep_lddt = read_plddt(plddt_path, mhc_seq, peptide)
        per_res.append(pae_peptide)
        all.append(np.mean(pae_peptide))
        plddt_per_res.append(pep_lddt)
        plddt_all.append(np.mean(pep_lddt))
        
    return per_res, all, plddt_per_res, plddt_all


def peptide_list_from_fasta(fasta_path, n=None):
    peptide_list = []
    with open(fasta_path, 'r') as f:
        for line in f:
            if line.startswith('>'): pass
            else: peptide_list.append(line.strip().replace('\n',''))
    if n:
        sampled = random.sample(peptide_list, n)
        return sampled
    else:
        return peptide_list
    
def exclude_peptide_from_list(peptide_list, constraints):
    if not peptide_list or not constraints:
        return list(peptide_list)
    requirements = sorted(
        [(pos, set(allowed)) for pos, allowed in constraints.items()],
        key=lambda x: len(x[1])
    )
    filtered_list = []
    for peptide in peptide_list:
        is_valid = True
        for pos, allowed_set in requirements:
            if peptide[pos] not in allowed_set:
                is_valid = False
                break  # Optimization 3: Stop checking this peptide immediately
        if is_valid:
            filtered_list.append(peptide)
            
    return filtered_list



def split_fasta(fasta_file, n_chunks):
    """Split FASTA file into n_chunks smaller files"""
    from Bio import SeqIO

    sequences = list(SeqIO.parse(fasta_file, "fasta"))
    chunk_size = len(sequences) // n_chunks + 1

    chunk_files = []
    for i in range(n_chunks):
        chunk = sequences[i * chunk_size:(i + 1) * chunk_size]
        if not chunk:
            break

        chunk_file = tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False)
        SeqIO.write(chunk, chunk_file.name, "fasta")
        chunk_files.append(chunk_file.name)
        chunk_file.close()

    return chunk_files


def run_netmhcpan_parallel(peptide_fasta, allele_list, output, mhc_type,
                           netmhcipan_path='/home/amir/amir/ParseFold/PMGen/netMHCIpan-4.1/netMHCpan',
                           netmhciipan_path='/home/amir/amir/ParseFold/PMGen/netMHCIIpan-4.3/netMHCIIpan',
                           n_jobs=None, verbose=False, length='9'):
    """
    Run NetMHCpan in parallel by splitting the input FASTA

    Args:
        n_jobs: Number of parallel jobs (default: use all CPU cores)
    """
    assert mhc_type in [1, 2]

    if n_jobs is None:
        n_jobs = cpu_count()
    print(length)
    # Split FASTA into chunks
    if verbose:
        print(f"Splitting FASTA into {n_jobs} chunks...")
    chunk_files = split_fasta(peptide_fasta, n_jobs)

    # Prepare arguments for each chunk
    temp_outputs = []
    args_list = []
    for i, chunk_file in enumerate(chunk_files):
        temp_output = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        temp_output.close()
        temp_outputs.append(temp_output.name)

        args_list.append((
            chunk_file, allele_list, temp_output.name, mhc_type,
            netmhcipan_path, netmhciipan_path, length
        ))

    # Run in parallel
    if verbose:
        print(f"Running NetMHCpan on {len(chunk_files)} chunks in parallel...")
    with Pool(processes=n_jobs) as pool:
        pool.map(run_netmhcpan_chunk, args_list)

    # Merge outputs
    if verbose:
        print("Merging results...")
    with open(output, 'w') as outfile:
        for i, temp_output in enumerate(temp_outputs):
            with open(temp_output, 'r') as infile:
                if i == 0:
                    # Include header from first file
                    outfile.write(infile.read())
                else:
                    # Skip header lines for subsequent files
                    lines = infile.readlines()
                    # Find where data starts (after header lines starting with #)
                    data_start = 0
                    for j, line in enumerate(lines):
                        if not line.startswith('#') and not line.startswith('-'):
                            data_start = j
                            break
                    outfile.writelines(lines[data_start:])

    # Cleanup temporary files
    if verbose:
        print("Cleaning up temporary files...")
    for chunk_file in chunk_files:
        Path(chunk_file).unlink()
    for temp_output in temp_outputs:
        Path(temp_output).unlink()
    if verbose:
        print(f"Done! Results saved to {output}")



def run_netmhcpan_chunk(args):
    """Run NetMHCpan on a single chunk"""
    peptide_fasta, allele_list, output, mhc_type, netmhcipan_path, netmhciipan_path, length = args

    if mhc_type == 1:
        cmd = [str(netmhcipan_path), '-f', str(peptide_fasta),
               '-BA', '-a', str(allele_list[0]), '-l', length]
    elif mhc_type == 2:
        final_allele = ""
        for allele in allele_list:
            if 'H-2' in allele or 'DRB' in allele:
                final_allele = allele
            else:
                if 'DQA' in allele or 'DPA' in allele:
                    final_allele += allele
                if 'DQB' in allele or 'DPB' in allele:
                    final_allele += f'-{allele.replace("HLA-", "")}'
        cmd = [str(netmhciipan_path), '-f', str(peptide_fasta),
               '-BA', '-u', '-s', '-length', length,
               '-inptype', '0', '-a', str(final_allele)]

    with open(output, 'w') as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE)
        
        if result.returncode != 0:
            print(f"Command failed: {' '.join(cmd)}")
            print(f"Return code: {result.returncode}")
            print(f"Stderr: {result.stderr.decode('utf-8')}")
            raise subprocess.CalledProcessError(result.returncode, cmd, stderr=result.stderr)

    return output

import re

def parse_netmhcpan_file(file_path):
    """
    Optimized parser for NetMHCpan output files.
    Uses efficient regex patterns and minimizes string operations.
    """
    # Read file once
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Pre-compile regex patterns (compiled once, reused many times)
    dash_pattern = re.compile(r'^-{50,}')
    pos_pattern = re.compile(r'^\s*Pos\s+')
    data_pattern = re.compile(r'^\s*\d+\s+')
    whitespace_splitter = re.compile(r'\s+')

    tables = []
    i = 0
    n_lines = len(lines)
    while i < n_lines:
        line = lines[i]
        # Check if this is a header line
        if pos_pattern.match(line):
            # Extract columns
            columns = whitespace_splitter.split(line.strip())
            if columns[-1] == 'BindLevel':
                columns = columns[:-1]
            n_cols = len(columns)
            i += 1
            # Skip the dashed line after header
            if i < n_lines and dash_pattern.match(lines[i]):
                i += 1
            # Collect data rows
            data = []
            while i < n_lines:
                line = lines[i]
                # Stop at next separator or end
                if dash_pattern.match(line) or not line.strip():
                    break
                # Check if it's a data row
                if data_pattern.match(line):
                    row = whitespace_splitter.split(line.strip())[:n_cols]
                    data.append(row)
                i += 1
            # Create DataFrame if we have data
            if data:
                df = pd.DataFrame(data, columns=columns)
                # Determine MHC type and convert/sort
                if 'Aff(nM)' in df.columns:
                    # MHC-I
                    df['Score_EL'] = pd.to_numeric(df['Score_EL'], errors='coerce')
                    df['Aff(nM)'] = pd.to_numeric(df['Aff(nM)'], errors='coerce')
                    df.sort_values(['Aff(nM)', 'Score_EL'], ascending=[True, False], inplace=True)
                elif 'Affinity(nM)' in df.columns:
                    # MHC-II
                    df['Score_EL'] = pd.to_numeric(df['Score_EL'], errors='coerce')
                    df['Affinity(nM)'] = pd.to_numeric(df['Affinity(nM)'], errors='coerce')
                    df.sort_values(['Affinity(nM)', 'Score_EL'], ascending=[True, False], inplace=True)

                tables.append(df)
        else:
            i += 1

    if not tables:
        raise ValueError(f"debugging message: {file_path} could not be parsed by parse_netmhcpan_file")

    return pd.concat(tables, ignore_index=True)

from scipy import stats
def corr_plot(x, y, xlabel='%Rank_EL', ylabel='ps_cpl_score', title=None, figsize=(7, 6)):
    """
    Create a scatter plot with log-scale x-axis and Spearman correlation.
    
    Args:
        x: array-like, x-axis data (will be log-scaled)
        y: array-like, y-axis data
        xlabel: str, label for x-axis
        ylabel: str, label for y-axis
        title: str, optional title for the plot
        figsize: tuple, figure size
    
    Returns:
        fig, ax, corr, p_val
    """
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = np.array(x)[mask]
    y_clean = np.array(y)[mask]
    
    # Calculate Spearman correlation
    corr, p_val = stats.spearmanr(x_clean, y_clean)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(x_clean, y_clean, s=25, c='black', alpha=0.5)
    ax.set_xscale('log')
    
    # Labels and ticks
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(axis='both', labelsize=18)
    
    # Remove grids
    ax.grid(False)
    
    # Add correlation and p-value on top
    if p_val < 0.001:
        p_text = f'p < 0.001'
    else:
        p_text = f'p = {p_val:.3f}'
    
    ax.set_title(f'Spearman ρ = {corr:.3f}, {p_text}', fontsize=20)
    
    if title:
        ax.text(0.5, 1.08, title, transform=ax.transAxes, fontsize=20, 
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    return fig, ax, corr, p_val



def calculate_enrichment_curve(all_ranks, selected_ranks, n_points=1000):
    """
    Calculate cumulative enrichment curve.
    
    X-axis: Percentile of full EL rank distribution
    Y-axis: Fraction of selected peptides with EL rank <= that percentile
    
    Args:
        all_ranks: array of EL ranks for ALL peptides
        selected_ranks: array of EL ranks for SELECTED peptides only
        n_points: number of points for the curve
    
    Returns:
        x: percentiles (0-100)
        y: fraction of selected peptides below each percentile threshold
        auc: area under curve (0.5 = random, 1.0 = perfect enrichment)
    """
    all_ranks = np.array(all_ranks)
    selected_ranks = np.array(selected_ranks)
    
    # Get percentiles of the full distribution
    percentiles = np.percentile(all_ranks, np.linspace(0, 100, n_points))
    percentiles = np.array([0] + list(percentiles))
    
    # For each percentile threshold, what fraction of selected peptides are below it?
    y = np.array([np.mean(selected_ranks <= t) for t in percentiles])
    
    # Normalize x to 0-1 for AUC calculation
    x_norm = np.linspace(0, 1, len(y))
    auc_val = np.trapz(y, x_norm)
    
    # X in percentile scale for plotting
    x = np.linspace(0, 100, len(y))
    
    return x, y, auc_val


def plot_enrichment_roc(combined_df, selections_dict, title='Enrichment Analysis', 
                        save_path=None, figsize=(10, 8)):
    """
    Plot cumulative enrichment curves for multiple selections.
    
    Args:
        combined_df: DataFrame with all peptides and '%Rank_EL' column
        selections_dict: dict of {label: {'mask': boolean_array, 'color': str}}
        title: plot title
        save_path: path to save figure
        figsize: figure size
    
    Returns:
        fig, auc_results
    """
    all_ranks = combined_df['%Rank_EL'].astype(float).values
    
    fig, ax = plt.subplots(figsize=figsize)
    
    auc_results = {}
    
    for label, data in selections_dict.items():
        mask = data['mask']
        color = data.get('color', None)
        
        # Get EL ranks of selected peptides only
        selected_ranks = all_ranks[mask]
        
        x, y, auc_val = calculate_enrichment_curve(all_ranks, selected_ranks)
        auc_results[label] = auc_val
        
        ax.plot(x, y * 100, label=f'{label} (AUC={auc_val:.3f})', color=color, linewidth=2)
    
    # Random baseline (diagonal)
    ax.plot([0, 100], [0, 100], 'k--', label='Random (AUC=0.500)', linewidth=1)
    
    ax.set_xlabel('Percentile of Full EL Rank Distribution', fontsize=16)
    ax.set_ylabel('% of Selected Peptides Below Threshold', fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.legend(fontsize=11, loc='lower right')
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, auc_results



def generate_statistical_report(report_dict, save_path):
    """
    Generate and save a statistical report.
    
    Args:
        report_dict: dict containing all statistics to report
        save_path: path to save the report
    """
    with open(save_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("STATISTICAL REPORT: PS-CPL vs Combined Scoring Analysis\n")
        f.write("=" * 70 + "\n\n")
        
        # Parameters
        f.write("PARAMETERS\n")
        f.write("-" * 70 + "\n")
        for key, val in report_dict.get('parameters', {}).items():
            f.write(f"  {key}: {val}\n")
        f.write("\n")
        
        # Correlation Analysis
        f.write("=" * 70 + "\n")
        f.write("CORRELATION ANALYSIS (Spearman)\n")
        f.write("=" * 70 + "\n\n")
        
        for sampling_method, corr_data in report_dict.get('correlations', {}).items():
            f.write(f"{sampling_method}:\n")
            for score_type, (corr, pval) in corr_data.items():
                f.write(f"  {score_type} vs EL Rank: ρ = {corr:.4f}, p = {pval:.2e}\n")
            f.write("\n")
        
        # Sample Sizes
        f.write("=" * 70 + "\n")
        f.write("SAMPLE SIZES\n")
        f.write("=" * 70 + "\n\n")
        
        for key, val in report_dict.get('sample_sizes', {}).items():
            f.write(f"  {key}: {val}\n")
        f.write("\n")
        
        # Enrichment Analysis (AUC Results)
        f.write("=" * 70 + "\n")
        f.write("ENRICHMENT ANALYSIS (AUC)\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"{'Selection Method':<40} {'AUC':<12}\n")
        f.write("-" * 52 + "\n")
        
        for label, auc_val in report_dict.get('auc_results', {}).items():
            f.write(f"{label:<40} {auc_val:<12.4f}\n")
        f.write("-" * 52 + "\n\n")
        
        # Interpretation
        f.write("INTERPRETATION\n")
        f.write("-" * 70 + "\n")
        f.write("Correlation:\n")
        f.write("  - Positive Spearman ρ: higher score correlates with higher EL rank\n")
        f.write("    (expected, since less negative score = better, lower EL rank = better)\n")
        f.write("  - Stronger |ρ| indicates better predictive power\n\n")
        f.write("Enrichment AUC:\n")
        f.write("  - AUC = 0.5: Random selection (no enrichment)\n")
        f.write("  - AUC > 0.5: Selected peptides enriched for strong binders\n")
        f.write("  - AUC = 1.0: Perfect enrichment (all selected are best binders)\n")
        f.write("\n")
    
    print(f"Report saved: {save_path}")



def plot_correlation_boxplot(corr_data, title='Correlation Coefficients Across Seeds', 
                             save_path=None, figsize=(6, 4)):
    """
    Plot boxplot of correlation coefficients across seeds.
    
    Args:
        corr_data: dict of {label: list of correlation values}
        title: plot title
        save_path: path to save figure
        figsize: figure size
    
    Returns:
        fig, ax
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    labels = list(corr_data.keys())
    data = [corr_data[label] for label in labels]
    
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    for patch, color in zip(bp['boxes'], colors[:len(labels)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Spearman ρ', fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.tick_params(axis='both', labelsize=10)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_auc_boxplot(auc_data, title='AUC Across Seeds', save_path=None, figsize=(5, 4)):
    """
    Plot boxplot of AUC values across seeds.
    
    Args:
        auc_data: dict of {label: list of AUC values}
        title: plot title
        save_path: path to save figure
        figsize: figure size
    
    Returns:
        fig, ax
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    labels = list(auc_data.keys())
    data = [auc_data[label] for label in labels]
    
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    for patch, color in zip(bp['boxes'], colors[:len(labels)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.axhline(y=0.5, color='k', linestyle='--', linewidth=1, label='Random')
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.tick_params(axis='both', labelsize=10)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax.set_ylim([0.3, 1.0])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_averaged_enrichment_curve(all_curves_dict, title='Averaged Enrichment Curves',
                                    save_path=None, figsize=(6, 5)):
    """
    Plot averaged enrichment curves with standard deviation bands.
    
    Args:
        all_curves_dict: dict of {label: {'curves': list of (x, y) tuples, 'color': str}}
        title: plot title
        save_path: path to save figure
        figsize: figure size
    
    Returns:
        fig, ax, mean_aucs
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    mean_aucs = {}
    
    for label, data in all_curves_dict.items():
        curves = data['curves']  # list of (x, y) tuples
        color = data.get('color', None)
        
        # Stack all y values (assuming same x for all)
        y_stack = np.array([y for x, y in curves])
        x = curves[0][0]  # assume all have same x
        
        mean_y = np.mean(y_stack, axis=0)
        std_y = np.std(y_stack, axis=0)
        
        # Calculate mean AUC
        x_norm = np.linspace(0, 1, len(mean_y))
        mean_auc = np.trapz(mean_y, x_norm)
        mean_aucs[label] = mean_auc
        
        ax.plot(x, mean_y * 100, label=f'{label} (AUC={mean_auc:.3f})', 
                color=color, linewidth=2)
        ax.fill_between(x, (mean_y - std_y) * 100, (mean_y + std_y) * 100, 
                        color=color, alpha=0.2)
    
    # Random baseline
    ax.plot([0, 100], [0, 100], 'k--', label='Random (AUC=0.500)', linewidth=1)
    
    ax.set_xlabel('Percentile of Full EL Rank Distribution', fontsize=12)
    ax.set_ylabel('% of Selected Peptides Below Threshold', fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9, loc='lower right')
    ax.tick_params(axis='both', labelsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax, mean_aucs


def generate_comprehensive_report(report_dict, save_path):
    """
    Generate comprehensive statistical report after multi-seed analysis.
    
    Args:
        report_dict: dict containing all statistics
        save_path: path to save the report
    """
    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE STATISTICAL REPORT: Multi-Seed Analysis\n")
        f.write("=" * 80 + "\n\n")
        
        # Parameters
        f.write("PARAMETERS\n")
        f.write("-" * 80 + "\n")
        for key, val in report_dict.get('parameters', {}).items():
            f.write(f"  {key}: {val}\n")
        f.write("\n")
        
        # Sample sizes
        f.write("=" * 80 + "\n")
        f.write("SAMPLE SIZES (per seed)\n")
        f.write("=" * 80 + "\n\n")
        for key, val in report_dict.get('sample_sizes', {}).items():
            if isinstance(val, list):
                f.write(f"  {key}: {np.mean(val):.1f} ± {np.std(val):.1f}\n")
            else:
                f.write(f"  {key}: {val}\n")
        f.write("\n")
        
        # Correlation Analysis
        f.write("=" * 80 + "\n")
        f.write("CORRELATION ANALYSIS (Spearman ρ) - Mean ± Std across seeds\n")
        f.write("=" * 80 + "\n\n")
        
        corr_data = report_dict.get('correlations', {})
        f.write(f"{'Method':<40} {'Mean ρ':<12} {'Std':<12} {'Min':<12} {'Max':<12}\n")
        f.write("-" * 80 + "\n")
        for label, values in corr_data.items():
            mean_v = np.mean(values)
            std_v = np.std(values)
            min_v = np.min(values)
            max_v = np.max(values)
            f.write(f"{label:<40} {mean_v:<12.4f} {std_v:<12.4f} {min_v:<12.4f} {max_v:<12.4f}\n")
        f.write("\n")
        
        # AUC Analysis
        f.write("=" * 80 + "\n")
        f.write("ENRICHMENT ANALYSIS (AUC) - Mean ± Std across seeds\n")
        f.write("=" * 80 + "\n\n")
        
        auc_data = report_dict.get('auc_results', {})
        f.write(f"{'Selection Method':<40} {'Mean AUC':<12} {'Std':<12} {'Min':<12} {'Max':<12}\n")
        f.write("-" * 80 + "\n")
        for label, values in auc_data.items():
            mean_v = np.mean(values)
            std_v = np.std(values)
            min_v = np.min(values)
            max_v = np.max(values)
            f.write(f"{label:<40} {mean_v:<12.4f} {std_v:<12.4f} {min_v:<12.4f} {max_v:<12.4f}\n")
        f.write("\n")
        
        # Top peptides analysis
        f.write("=" * 80 + "\n")
        f.write("TOP PEPTIDES ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        top_pep_stats = report_dict.get('top_peptide_stats', {})
        for method, stats in top_pep_stats.items():
            f.write(f"{method}:\n")
            f.write(f"  Total unique peptides: {stats['total_unique']}\n")
            f.write(f"  Peptides appearing in all seeds: {stats['in_all_seeds']}\n")
            f.write(f"  Peptides appearing in >50% seeds: {stats['in_majority_seeds']}\n")
            f.write("\n")
        
        # Interpretation
        f.write("=" * 80 + "\n")
        f.write("INTERPRETATION\n")
        f.write("=" * 80 + "\n\n")
        f.write("Correlation:\n")
        f.write("  - Positive Spearman ρ: higher score correlates with higher EL rank\n")
        f.write("  - Lower variance across seeds indicates robust scoring\n\n")
        f.write("Enrichment AUC:\n")
        f.write("  - AUC = 0.5: Random selection (no enrichment)\n")
        f.write("  - AUC > 0.5: Selected peptides enriched for strong binders\n")
        f.write("  - AUC = 1.0: Perfect enrichment\n")
        f.write("  - Lower variance across seeds indicates robust selection\n\n")
        f.write("Top Peptides:\n")
        f.write("  - Peptides appearing in all/most seeds are most reliable candidates\n")
        f.write("\n")
    
    print(f"Report saved: {save_path}")


def corr_plot(x, y, xlabel='%Rank_EL', ylabel='ps_cpl_score', title=None, figsize=(5, 4)):
    """
    Create a scatter plot with log-scale x-axis and Spearman correlation.
    
    Args:
        x: array-like, x-axis data (will be log-scaled)
        y: array-like, y-axis data
        xlabel: str, label for x-axis
        ylabel: str, label for y-axis
        title: str, optional title for the plot (overrides default correlation title)
        figsize: tuple, figure size
    
    Returns:
        fig, ax, corr, p_val
    """
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = np.array(x)[mask]
    y_clean = np.array(y)[mask]
    
    # Calculate Spearman correlation
    corr, p_val = stats.spearmanr(x_clean, y_clean)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(x_clean, y_clean, s=25, c='black', alpha=0.5)
    ax.set_xscale('log')
    
    # Labels and ticks
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    
    # Remove grids
    ax.grid(False)
    
    # Add correlation and p-value
    if p_val < 0.001:
        p_text = f'p < 0.001'
    else:
        p_text = f'p = {p_val:.3f}'
    
    corr_text = f'Spearman ρ = {corr:.3f}, {p_text}'
    
    if title:
        ax.set_title(f'{title}\n{corr_text}', fontsize=12)
    else:
        ax.set_title(corr_text, fontsize=14)
    
    plt.tight_layout()
    
    return fig, ax, corr, p_val