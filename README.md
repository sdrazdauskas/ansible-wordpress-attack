# WordPress Attack & Detection Automation

This project automates the deployment of a vulnerable WordPress instance, simulates attacks (CVE-2020-25213), captures network traffic, and analyzes incidents using various sequence alignment algorithms.

## Features
- Automated, one-command setup using `main.sh` (provisions VMs, configures Ansible, installs dependencies, and runs analysis)
- Automated deployment of attacker and vulnerable WordPress VMs using Ansible
- Exploitation of WP File Manager CVE-2020-25213
- Upload and execution of arbitrary PHP files (e.g., `access.php`)
- Simultaneous packet capture on attacker and target
- Automated retrieval and analysis of PCAP files
- Incident detection and reconstruction using Needleman-Wunsch, Smith-Waterman, DTW, and Euclidean matching
- Visualization of traffic features

## Usage

1. **Modify and run `./main.sh` with your OpenNebula information (endpoint, username, etc.)**. This script will:
   - Install all required system and Python dependencies
   - Provision attacker and web server VMs on OpenNebula
   - Set up SSH keys and Ansible inventory/vault files automatically
   - Run the main Ansible playbook to deploy the environment
   - Prepare a Python virtual environment for PCAP analysis
   - Install all Python packages needed for traffic analysis and visualization
   - Run the PCAP analysis script
2. **(Optional) Re-run Ansible playbooks manually**:
   ```sh
   ansible-playbook -i ansible/inventory/hosts ansible/main.yml --ask-vault-pass --ask-become-pass
   ansible-playbook -i ansible/inventory/hosts ansible/attack.yml --ask-vault-pass --ask-become-pass
   ```
3. **(Optional) Analyze the PCAP files manually**:
   ```sh
   python3 analyze_pcap.py
   ```

## Requirements

### To run `main.sh` (full automation):
- Ubuntu 22.04 or 24.04 (or compatible Linux)
- OpenNebula account and endpoint information

### To run only the Ansible scripts and analysis manually:
- Ansible
- Python 3.8+
- Python packages: scapy, biopython, fastdtw, numpy, scipy, matplotlib

## Notes
- The `access.php` file is uploaded as the payload for exploitation.
- Visualizations are saved as PNG files for review on your local/ansible machine.

## File Structure
- `ansible/` - Ansible playbooks and configuration
- `src/access.php` - PHP payload/backdoor to be uploaded
- `analyze_pcap.py` - PCAP analysis and visualization script
- `captured_data/` - Directory for retrieved PCAP and result files

## Disclaimer
For educational use only.
