import pyshark
import numpy as np
import matplotlib.pyplot as plt
from Bio.Align import PairwiseAligner
from dtw import dtw
from itertools import islice
from scapy.all import rdpcap

def extract_features(pcap_file):
    """
    Extracts features from a PCAP file.
    Features include packet lengths and inter-arrival times as floats.
    """
    packets = rdpcap(pcap_file)
    packet_lengths = []
    inter_arrival_times = []

    for i, packet in enumerate(packets):
        packet_lengths.append(len(packet))
        if i > 0:
            # Convert times to float to avoid issues with Scapy's EDecimal type.
            inter_arrival_times.append(float(packet.time) - float(packets[i - 1].time))

    return np.array(packet_lengths), np.array(inter_arrival_times)

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
    Uses Bio.Align.PairwiseAligner to perform a full sequence (global)
    alignment. This example sets up a simplified scoring system:
      - +1 for a match
      - 0 for a mismatch or gap
    """
    aligner = PairwiseAligner()
    aligner.mode = "global"
    # Set scoring to mimic globalxx from pairwise2: +1 for match,
    # 0 for mismatch and gaps. (PairwiseAligner requires gap scores.)
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = 0
    aligner.extend_gap_score = 0

    alignments = aligner.align(seq1, seq2)
    for alignment in islice(alignments, 5):
        print(alignment)
    
    return alignments

# -------------------------------------------------
# Local Alignment (Smith-Waterman)
# -------------------------------------------------

def smith_waterman_align(seq1, seq2):
    """
    Uses Bio.Align.PairwiseAligner for local (Smith–Waterman) alignment.
    This function picks out the best matching sub-sequences between the two sequences.
    """
    aligner = PairwiseAligner()
    aligner.mode = "local"

    # Configure the scoring system:
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = -1
    aligner.extend_gap_score = 0

    alignments = aligner.align(seq1, seq2)

    for alignment in islice(alignments, 5):
        print(alignment)
        
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
    This function assumes vec1 and vec2 are numpy arrays
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must be the same length for direct Euclidean matching.")
    return np.linalg.norm(vec1 - vec2)

def detect_anomaly_from_features(baseline_features, test_features, length_threshold=5, time_threshold=0.5):
    """
    Detects anomalies by comparing the baseline feature vectors with the test feature vectors.
    A threshold is set for the mean differences in packet lengths and inter-arrival times.
    """
    baseline_packet, baseline_inter_arrival = baseline_features
    test_packet, test_inter_arrival = test_features

    # Compute means for each feature
    mean_len_baseline = np.mean(baseline_packet)
    mean_len_test = np.mean(test_packet)
    mean_time_baseline = np.mean(baseline_inter_arrival) if len(baseline_inter_arrival) > 0 else 0
    mean_time_test = np.mean(test_inter_arrival) if len(test_inter_arrival) > 0 else 0
    
    # Optionally, you could compute standard deviations and use z-scores for better sensitivity.
    std_len_baseline = np.std(baseline_packet)
    std_time_baseline = np.std(baseline_inter_arrival) if len(baseline_inter_arrival) > 0 else 1

    z_len = abs(mean_len_test - mean_len_baseline) / std_len_baseline
    z_time = abs(mean_time_test - mean_time_baseline) / std_time_baseline

    print(f"Baseline Packet Length Mean: {mean_len_baseline:.2f}, Test Packet Length Mean: {mean_len_test:.2f} (z-score: {z_len:.2f})")
    print(f"Baseline Inter-Arrival Mean: {mean_time_baseline:.4f}, Test Inter-Arrival Mean: {mean_time_test:.4f} (z-score: {z_time:.2f})")
    
    anomaly_detected = False
    # Here, you can set thresholds based on z-scores or absolute differences.
    if z_len > length_threshold:
        print("Anomaly detected in packet lengths!")
        anomaly_detected = True
    if z_time > time_threshold:
        print("Anomaly detected in inter-arrival times!")
        anomaly_detected = True

    return anomaly_detected


if __name__ == "__main__":
    attacker_pcap = "../misc/attacker.pcap"
    target_pcap = "../misc/target.pcap"
    baseline_pcap = "../misc/baseline.pcap"
    
    # For sequence-based methods:
    attacker_seq = pcap_to_sequence(attacker_pcap)
    target_seq = pcap_to_sequence(target_pcap)
    baseline_seq = pcap_to_sequence(baseline_pcap)
    
    # For statistical feature analysis:
    baseline_features = extract_features(baseline_pcap)
    target_features = extract_features(target_pcap)
    
    print("Attacker Sequence:", attacker_seq)
    print("Defender Sequence:", target_seq)
    
    # Run sequence alignment methods
    print("\n--- Global Alignment (Needleman-Wunsch) ---")
    needleman_wunsch_align(attacker_seq, target_seq)
    
    print("\n--- Local Alignment (Smith-Waterman) ---")
    smith_waterman_align(attacker_seq, target_seq)
    
    # Compute DTW distance, which is another way to capture sequence differences.
    dtw_dist = dtw_distance(attacker_seq, target_seq, visualize=True)
    print("\n--- DTW Distance ---")
    print("DTW Distance:", dtw_dist)
    
    # Detect anomalies using the extracted statistical features:
    if detect_anomaly_from_features(baseline_features, target_features):
        print("Network anomaly detected based on statistical feature analysis!")
    else:
        print("No significant anomalies detected in statistical feature analysis.")
