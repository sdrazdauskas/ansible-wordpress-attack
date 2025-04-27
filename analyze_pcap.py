import numpy as np
from scapy.all import rdpcap
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from Bio import pairwise2
from Bio.pairwise2 import format_alignment

def extract_features(pcap_file):
    """
    Extracts features from a PCAP file.
    Features include packet lengths and inter-arrival times.
    """
    packets = rdpcap(pcap_file)
    packet_lengths = []
    inter_arrival_times = []

    for i, packet in enumerate(packets):
        packet_lengths.append(len(packet))
        if i > 0:
            inter_arrival_times.append(packet.time - packets[i - 1].time)

    return np.array(packet_lengths), np.array(inter_arrival_times)

def needleman_wunsch(seq1, seq2):
    """
    Perform Needleman-Wunsch alignment on two sequences.
    Aligns numerical sequences by normalizing and converting to discrete values.
    """
    # Convert to float first to avoid EDecimal issues
    seq1 = np.array(seq1, dtype=float)
    seq2 = np.array(seq2, dtype=float)
    seq1 = np.round(seq1, decimals=3)
    seq2 = np.round(seq2, decimals=3)

    # Convert to strings for alignment
    seq1 = ''.join(map(str, seq1))
    seq2 = ''.join(map(str, seq2))

    alignments = pairwise2.align.globalxx(seq1, seq2)
    print("Needleman-Wunsch Alignment:")
    for alignment in alignments[:5]:
        print(format_alignment(*alignment))

def smith_waterman(seq1, seq2):
    """
    Perform Smith-Waterman alignment on two sequences.
    """
    # Convert to float first to avoid EDecimal issues
    seq1 = np.array(seq1, dtype=float)
    seq2 = np.array(seq2, dtype=float)
    seq1 = np.round(seq1, decimals=3)
    seq2 = np.round(seq2, decimals=3)

    # Convert to strings for alignment
    seq1 = ''.join(map(str, seq1))
    seq2 = ''.join(map(str, seq2))

    alignments = pairwise2.align.localxx(seq1, seq2)
    print("Smith-Waterman Alignment:")
    for alignment in alignments[:5]:
        print(format_alignment(*alignment))

def dynamic_time_warping(seq1, seq2):
    """
    Perform Dynamic Time Warping (DTW) on two sequences.
    """
    seq1 = np.asarray(seq1, dtype=float).flatten()
    seq2 = np.asarray(seq2, dtype=float).flatten()
    print(f"DTW seq1 shape: {seq1.shape}, type: {type(seq1[0])}")
    print(f"DTW seq2 shape: {seq2.shape}, type: {type(seq2[0])}")
    # Use absolute difference for scalar sequences
    distance, path = fastdtw(seq1.tolist(), seq2.tolist(), dist=lambda a, b: abs(a - b))
    print(f"DTW Distance: {distance}")
    print(f"DTW Path: {path}")

def euclidean_matching(seq1, seq2):
    """
    Perform Euclidean matching on two sequences.
    """
    if len(seq1) != len(seq2):
        print("Sequences must be of equal length for Euclidean matching.")
        return
    distance = np.linalg.norm(seq1 - seq2)
    print(f"Euclidean Distance: {distance}")

if __name__ == "__main__":
    # Load PCAP files
    attacker_pcap = "captured_data/attacker.pcap"
    target_pcap = "captured_data/target.pcap"

    # Extract features
    attacker_lengths, attacker_times = extract_features(attacker_pcap)
    target_lengths, target_times = extract_features(target_pcap)

    # Perform analysis
    print("Analyzing Packet Lengths:")
    needleman_wunsch(attacker_lengths, target_lengths)
    smith_waterman(attacker_lengths, target_lengths)
    dynamic_time_warping(attacker_lengths, target_lengths)
    euclidean_matching(attacker_lengths, target_lengths)

    print("\nAnalyzing Inter-Arrival Times:")
    needleman_wunsch(attacker_times, target_times)
    smith_waterman(attacker_times, target_times)
    dynamic_time_warping(attacker_times, target_times)
    euclidean_matching(attacker_times, target_times)