import argparse
import pandas as pd
from scapy.all import PcapReader, ARP
from tqdm import tqdm

def extract_features(pcap_file):
    rows = []
    last_seen = {}
    total_packets = 0

    # Use PcapReader for large files
    with PcapReader(pcap_file) as pcap:
        for pkt in tqdm(pcap, desc="Processing packets"):
            total_packets += 1
            if pkt.haslayer(ARP):
                ts = pkt.time
                op = pkt[ARP].op  # 1=request, 2=reply
                sender_ip = pkt[ARP].psrc
                target_ip = pkt[ARP].pdst
                sender_mac = pkt[ARP].hwsrc
                target_mac = pkt[ARP].hwdst

                # Convert IP and MAC to integers
                sender_ip_int = int.from_bytes(bytes(map(int, sender_ip.split("."))), "big")
                target_ip_int = int.from_bytes(bytes(map(int, target_ip.split("."))), "big")
                sender_mac_int = int(sender_mac.replace(":", ""), 16)
                target_mac_int = int(target_mac.replace(":", ""), 16)
                op_is_request = int(op == 1)
                op_is_reply = int(op == 2)

                # Timing feature: inter-arrival time per sender IP
                inter_arrival = ts - last_seen.get(sender_ip_int, ts)
                last_seen[sender_ip_int] = ts

                # Feature dictionary
                row = {
                    "ts": ts,
                    "sender_ip_int": sender_ip_int,
                    "target_ip_int": target_ip_int,
                    "sender_mac_int": sender_mac_int,
                    "target_mac_int": target_mac_int,
                    "op_is_request": op_is_request,
                    "op_is_reply": op_is_reply,
                    "ip_change_flag": 0,
                    "mac_impersonation_flag": 0,
                    "dup_ip_distinct_macs_5s": 0,
                    "rate_from_sender_ip_5s": 0,
                    "inter_arrival_sender_ip": inter_arrival,
                    "label": 0  # default, can be updated for attack files
                }
                rows.append(row)

    print(f"[+] Total packets processed: {total_packets}")
    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcap", required=True, help="Input PCAP file")
    parser.add_argument("--out", required=True, help="Output CSV file")
    args = parser.parse_args()

    df = extract_features(args.pcap)
    df.to_csv(args.out, index=False)
    print(f"[+] Features saved to {args.out}")
