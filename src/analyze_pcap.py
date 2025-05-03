import pyshark
import numpy as np
import matplotlib.pyplot as plt
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from dtw import dtw

def pcap_to_sequence(pcap_file):
    """
    Converts a PCAP file into a simple sequence.
    For this example, we map:
      - TCP packets -> 'T'
      - UDP packets -> 'U'
      - ICMP packets -> 'I'
      - Other protocols -> 'X'
    """
    capture = pyshark.FileCapture(pcap_file)
    sequence = []
    
    for packet in capture:
        try:
            if 'TCP' in packet:
                sequence.append('T')
            elif 'UDP' in packet:
                sequence.append('U')
            elif 'ICMP' in packet:
                sequence.append('I')
            else:
                sequence.append('X')
        except Exception as e:
            # In case the packet doesn’t have expected layers
            sequence.append('X')
    capture.close()
    return "".join(sequence)

# -------------------------------------------------
# Global Alignment (Needleman-Wunsch)
# -------------------------------------------------

def needleman_wunsch_align(seq1, seq2):
    """
    Uses Biopython's pairwise2 to perform a full sequence (global)
    alignment. Here we use a simple scoring: +1 for a match, 0 for a mismatch.
    You can modify the scoring matrix as needed.
    """
    alignments = pairwise2.align.globalxx(seq1, seq2)
    # Here, globalxx uses 1 for match and 0 for mismatch/gap penalties.
    print(format_alignment(*alignments[0]))
    return alignments

# -------------------------------------------------
# Local Alignment (Smith-Waterman)
# -------------------------------------------------

def smith_waterman_align(seq1, seq2):
    """
    Uses Biopython's pairwise2 for local alignment.
    This will pick out the best matching sub-sequences between the two sequences.
    """
    alignments = pairwise2.align.localxx(seq1, seq2)
    for alignment in alignments:
        print(format_alignment(*alignment))
    return alignments

# -------------------------------------------------
# Time-Series Matching (DTW)
# -------------------------------------------------

def dtw_distance(seq1, seq2, visualize=False):
    """
    Computes DTW distance between two sequences (after mapping to numerical arrays)
    and, if visualize is True, saves a plot of the accumulated cost matrix with
    the warping path.
    """
    # Convert sequences to numerical arrays using a simple mapping.
    mapping = {'T': 1, 'U': 2, 'I': 3, 'X': 4}
    vec1 = np.array([mapping.get(ch, 0) for ch in seq1])
    vec2 = np.array([mapping.get(ch, 0) for ch in seq2])
    
    # Compute DTW with internals kept.
    dtw_result = dtw(vec1, vec2, keep_internals=True)
    
    # Access the distance attributes.
    global_distance = dtw_result.distance  # Minimum global distance.
    normalized_distance = getattr(dtw_result, 'normalizedDistance', None)
    print("Global distance:", global_distance)
    if normalized_distance is not None:
        print("Normalized distance:", normalized_distance)
    
    # For visualization, we use the costMatrix and the warping path.
    if visualize:
        try:
            cost_matrix = dtw_result.costMatrix
            # Construct a warping path from index1 and index2:
            path = list(zip(dtw_result.index1, dtw_result.index2))
        except AttributeError:
            print("The required attributes for visualization are not available.")
            cost_matrix, path = None, None

        if cost_matrix is not None and path is not None:
            plt.figure(figsize=(8, 6))
            plt.imshow(cost_matrix, origin='lower', cmap='viridis', aspect='auto')
            path_x, path_y = zip(*path)
            plt.plot(path_y, path_x, color='red', linewidth=2, label='Warping Path')
            plt.xlabel("Reference Sequence Index")
            plt.ylabel("Query Sequence Index")
            plt.title("DTW Accumulated Cost Matrix & Warping Path")
            plt.colorbar(label="Accumulated Cost")
            plt.legend()
            plt.tight_layout()
            plt.savefig('dtw_plot.png')
            plt.close()
            print(f"Visualization saved as dtw_plot.png")
    
    return global_distance

# -------------------------------------------------
# Euclidean Matching for Feature Vectors
# -------------------------------------------------

def euclidean_distance_matching(vec1, vec2):
    """
    Computes the Euclidean distance between two vectors.
    This function assumes vec1 and vec2 are numpy arrays; you would
    typically generate these feature vectors from the PCAP.
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must be the same length for direct Euclidean matching.")
    return np.linalg.norm(vec1 - vec2)


if __name__ == "__main__":
    attacker_pcap = "../misc/attacker.pcap"
    target_pcap = "../misc/target.pcap"
    
    attacker_seq = pcap_to_sequence(attacker_pcap)
    target_seq = pcap_to_sequence(target_pcap)
    
    print("Attacker Sequence:", attacker_seq)
    print("Defender Sequence:", target_seq)
    
    print("\n--- Global Alignment (Needleman-Wunsch) ---")
    needleman_wunsch_align(attacker_seq, target_seq)
    
    print("\n--- Local Alignment (Smith-Waterman) ---")
    smith_waterman_align(attacker_seq, target_seq)
    
    dtw_dist = dtw_distance(attacker_seq, target_seq, visualize=True)
    print("\n--- DTW Distance ---")
    print("DTW Distance:", dtw_dist)
    
    # Example numerical vectors for Euclidean matching:
    # (In practice these would come from more detailed analysis.)
    feat_attacker = np.array([1, 2, 3, 4])
    feat_target = np.array([1, 2, 2, 4])
    
    try:
        euc_dist = euclidean_distance_matching(feat_attacker, feat_target)
        print("\n--- Euclidean Distance Matching ---")
        print("Euclidean Distance:", euc_dist)
    except ValueError as ve:
        print("Error in Euclidean matching:", ve)
