import pyshark
import numpy as np
import matplotlib.pyplot as plt
from Bio.Align import PairwiseAligner
from dtw import dtw
from itertools import islice
from scapy.all import rdpcap

# -------------------------------------------------
# Feature Extraction and Sequence Construction
# -------------------------------------------------

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
# Numeric Alignment Functions
# -------------------------------------------------

def numeric_global_alignment(seq1, seq2, gap_cost=0):
    """
    Performs global alignment on two numeric sequences using dynamic programming.
    The alignment cost is the sum of absolute differences with a gap penalty.
    Lower cost indicates a better alignment.
    """
    seq1 = np.array(seq1)
    seq2 = np.array(seq2)
    n = len(seq1)
    m = len(seq2)
    dp = np.zeros((n + 1, m + 1))
    
    for i in range(1, n + 1):
        dp[i, 0] = dp[i - 1, 0] + gap_cost
    for j in range(1, m + 1):
        dp[0, j] = dp[0, j - 1] + gap_cost
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(seq1[i - 1] - seq2[j - 1])
            dp[i, j] = min(
                dp[i - 1, j - 1] + cost,
                dp[i - 1, j] + gap_cost,
                dp[i, j - 1] + gap_cost
            )
    return dp[n, m]

def numeric_local_alignment(seq1, seq2, gap_cost=0):
    """
    Performs local alignment on two numeric sequences.
    We first convert differences into a similarity measure by taking their negative.
    Higher (less negative) scores mean better local alignment.
    """
    seq1 = np.array(seq1, dtype=float)
    seq2 = np.array(seq2, dtype=float)
    n = len(seq1)
    m = len(seq2)
    dp = np.zeros((n + 1, m + 1))
    max_score = 0.0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Using negative absolute difference as similarity
            score = -abs(seq1[i - 1] - seq2[j - 1])
            match = dp[i - 1, j - 1] + score
            delete = dp[i - 1, j] - gap_cost
            insert = dp[i, j - 1] - gap_cost
            dp[i, j] = max(0, match, delete, insert)
            if dp[i, j] > max_score:
                max_score = dp[i, j]
    return max_score

# -------------------------------------------------
# Symbolic Alignment Functions
# -------------------------------------------------

def needleman_wunsch_align(seq1, seq2):
    """
    Uses Bio.Align.PairwiseAligner for global (Needleman-Wunsch) alignment of symbolic sequences.
    """
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = 0
    aligner.extend_gap_score = 0

    alignments = aligner.align(seq1, seq2)
    for alignment in islice(alignments, 5):
        print(alignment)
    
    return alignments

def smith_waterman_align(seq1, seq2):
    """
    Uses Bio.Align.PairwiseAligner for local (Smith–Waterman) alignment of symbolic sequences.
    """
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = -1
    aligner.extend_gap_score = 0

    alignments = aligner.align(seq1, seq2)
    for alignment in islice(alignments, 5):
        print(alignment)
        
    return alignments

# -------------------------------------------------
# DTW Distance (Supports Both Numeric and Symbolic)
# -------------------------------------------------

def dtw_distance(seq1, seq2, visualize=False):
    """
    Computes DTW distance between two sequences.
    If inputs are numeric, they are used directly.
    Otherwise, a mapping is applied (for symbolic sequences).
    """
    seq1_arr = np.array(seq1)
    seq2_arr = np.array(seq2)
    
    if np.issubdtype(seq1_arr.dtype, np.number) and np.issubdtype(seq2_arr.dtype, np.number):
        vec1 = seq1_arr
        vec2 = seq2_arr
    else:
        mapping = {'T': 1, 'U': 2, 'I': 3, 'X': 4}
        vec1 = np.array([mapping.get(ch, 0) for ch in seq1])
        vec2 = np.array([mapping.get(ch, 0) for ch in seq2])
    
    dtw_result = dtw(vec1, vec2, keep_internals=True)
    
    global_distance = dtw_result.distance  # DTW cost
    normalized_distance = getattr(dtw_result, 'normalizedDistance', None)
    print("Global DTW distance:", global_distance)
    if normalized_distance is not None:
        print("Normalized DTW distance:", normalized_distance)
    
    if visualize:
        try:
            cost_matrix = dtw_result.costMatrix
            path = list(zip(dtw_result.index1, dtw_result.index2))
        except AttributeError:
            print("Visualization attributes not available.")
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
            print("Visualization saved as dtw_plot.png")
    
    return global_distance

# -------------------------------------------------
# Euclidean Distance (Numeric Only)
# -------------------------------------------------

def euclidean_distance_matching(vec1, vec2):
    """
    Computes the Euclidean distance between two numeric vectors.
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must be the same length for direct Euclidean matching.")
    return np.linalg.norm(np.array(vec1) - np.array(vec2))

def plot_features(data1, data2, title, ylabel, filename=None):
    plt.figure(figsize=(10, 4))
    plt.plot(data1, label="Data1", marker='o')
    plt.plot(data2, label="Data2", marker='x')
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()

# -------------------------------------------------
# Anomaly Detection
# -------------------------------------------------

def detect_anomaly_from_features(baseline_features, test_features, length_threshold=5, time_threshold=0.5):
    """
    Detects anomalies by comparing the baseline feature vectors with the test feature vectors.
    """
    baseline_packet, baseline_inter_arrival = baseline_features
    test_packet, test_inter_arrival = test_features

    mean_len_baseline = np.mean(baseline_packet)
    mean_len_test = np.mean(test_packet)
    mean_time_baseline = np.mean(baseline_inter_arrival) if len(baseline_inter_arrival) > 0 else 0
    mean_time_test = np.mean(test_inter_arrival) if len(test_inter_arrival) > 0 else 0
    
    std_len_baseline = np.std(baseline_packet)
    std_time_baseline = np.std(baseline_inter_arrival) if len(baseline_inter_arrival) > 0 else 1

    z_len = abs(mean_len_test - mean_len_baseline) / std_len_baseline
    z_time = abs(mean_time_test - mean_time_baseline) / std_time_baseline

    print(f"Baseline Packet Length Mean: {mean_len_baseline:.2f}, Test Packet Length Mean: {mean_len_test:.2f} (z-score: {z_len:.2f})")
    print(f"Baseline Inter-Arrival Mean: {mean_time_baseline:.4f}, Test Inter-Arrival Mean: {mean_time_test:.4f} (z-score: {z_time:.2f})")
    
    anomaly_detected = False
    if z_len > length_threshold:
        print("Anomaly detected in packet lengths!")
        anomaly_detected = True
    if z_time > time_threshold:
        print("Anomaly detected in inter-arrival times!")
        anomaly_detected = True

    return anomaly_detected

# -------------------------------------------------
# Unified Sequence Analysis Function
# -------------------------------------------------

def sequence_analysis(seq1, seq2, data_type='symbolic'):
    """
    Analyzes two sequences using alignment methods.
    If data_type is 'numeric', numeric alignment functions are used;
    otherwise, symbolic (string-based) alignment is performed.
    """
    if data_type == 'numeric':
        print("\n--- Global Alignment (Numeric Needleman-Wunsch) ---")
        global_cost = numeric_global_alignment(seq1, seq2)
        print("Numeric Global Alignment Cost:", global_cost)
        print("\n--- Local Alignment (Numeric Smith-Waterman) ---")
        local_score = numeric_local_alignment(seq1, seq2)
        print("Numeric Local Alignment Score:", local_score)
        print("\n--- DTW Distance (Numeric) ---")
        dtw_dist = dtw_distance(seq1, seq2, visualize=True)
        print("Numeric DTW Distance:", dtw_dist)
    else:
        print("\n--- Global Alignment (Needleman-Wunsch) ---")
        needleman_wunsch_align(seq1, seq2)
        print("\n--- Local Alignment (Smith-Waterman) ---")
        smith_waterman_align(seq1, seq2)
        print("\n--- DTW Distance ---")
        dtw_dist = dtw_distance(seq1, seq2, visualize=True)
        print("DTW Distance:", dtw_dist)


if __name__ == "__main__":
    attacker_pcap = "../misc/attacker.pcap"
    target_pcap = "../misc/target.pcap"
    baseline_pcap = "../misc/baseline.pcap"
    
    # Retrieve symbolic sequences (e.g., protocol mappings)
    attacker_seq = pcap_to_sequence(attacker_pcap)
    target_seq = pcap_to_sequence(target_pcap)
    baseline_seq = pcap_to_sequence(baseline_pcap)
    
    print("Attacker Sequence:", attacker_seq)
    print("Target Sequence:", target_seq)
    
    # Retrieve numeric features (packet lengths and inter-arrival times)
    baseline_features = extract_features(baseline_pcap)
    attacker_features = extract_features(attacker_pcap)
    target_features = extract_features(target_pcap)
    
    print("\nSymbolic Sequence Analysis (Target vs. Baseline):")
    sequence_analysis(target_seq, baseline_seq, data_type='symbolic')
    
    # For numeric analyses, deconstruct features
    baseline_packet, baseline_inter_arrival = baseline_features
    attacker_packet, attacker_inter_arrival = attacker_features
    target_packet, target_inter_arrival = target_features

    print("\nNumeric Feature Analysis: Packet Lengths (Attacker vs. Baseline)")
    sequence_analysis(attacker_packet, baseline_packet, data_type='numeric')

    print("\nNumeric Feature Analysis: Inter-Arrival Times (Attacker vs. Baseline)")
    sequence_analysis(attacker_inter_arrival, baseline_inter_arrival, data_type='numeric')

    # Anomaly detection between baseline and target features
    if detect_anomaly_from_features(baseline_features, target_features):
        print("Network anomaly detected based on statistical feature analysis!")
    else:
        print("No significant anomalies detected in statistical feature analysis.")

    # Plot numeric features for visual comparison
    plot_features(attacker_packet, target_packet,
                  "Packet Lengths Comparison", "Packet Length", "packet_lengths.png")
    plot_features(attacker_inter_arrival, target_inter_arrival,
                  "Inter-Arrival Times Comparison", "Inter-Arrival Time (s)", "inter_arrival_times.png")
