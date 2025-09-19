# src/live_detect.py
import argparse
import scapy.all as scapy
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from collections import defaultdict
import time

# Load model and scaler
scaler = joblib.load("models/scaler.joblib")
model = load_model("models/lstm_model.h5")

# Global state for live sniffing
last_seen = {}
ts_window = []
window_sec = 5
ip_mac_map = defaultdict(set)  # For tracking duplicate IPs and MACs
mac_sender_count = defaultdict(list)  # For rate_from_sender_ip_5s

def mac_to_int(mac):
    return int(mac.replace(":", ""), 16)

def ip_to_int(ip):
    parts = ip.split(".")
    return sum(int(parts[i]) << (8*(3-i)) for i in range(4))

def process_row(sender_ip, target_ip, sender_mac, target_mac, op, ts):
    sender_ip_int = ip_to_int(sender_ip)
    target_ip_int = ip_to_int(target_ip)
    sender_mac_int = mac_to_int(sender_mac)
    target_mac_int = mac_to_int(target_mac)
    op_is_request = int(op == 1)
    op_is_reply = int(op == 2)

    # Inter-arrival time
    inter_arrival = ts - last_seen.get(sender_ip_int, ts)
    last_seen[sender_ip_int] = ts

    # IP change / MAC impersonation detection
    ip_mac_map[sender_ip_int].add(sender_mac_int)
    ip_change_flag = int(len(ip_mac_map[sender_ip_int]) > 1)
    mac_impersonation_flag = ip_change_flag

    # Rate / duplicates in last window_sec
    ts_window.append((ts, sender_ip_int, sender_mac_int))
    ts_window[:] = [(t, ip, mac) for t, ip, mac in ts_window if ts - t <= window_sec]
    dup_ip_distinct_macs_5s = int(len({mac for t, ip, mac in ts_window if ip == sender_ip_int}) > 1)
    mac_sender_count[sender_ip_int].append(ts)
    mac_sender_count[sender_ip_int] = [t for t in mac_sender_count[sender_ip_int] if ts - t <= window_sec]
    rate_from_sender_ip_5s = len(mac_sender_count[sender_ip_int])

    # Build feature row
    row = np.array([[sender_ip_int, target_ip_int, sender_mac_int, target_mac_int,
                     op_is_request, op_is_reply, ip_change_flag, mac_impersonation_flag,
                     dup_ip_distinct_macs_5s, rate_from_sender_ip_5s, inter_arrival]])

    # Scale and reshape for LSTM
    row_scaled = scaler.transform(row).reshape((1, 1, row.shape[1]))
    pred = model.predict(row_scaled, verbose=0)
    return pred[0,0]

def process_packet(pkt, threshold):
    if not pkt.haslayer(scapy.ARP):
        return
    ts = pkt.time
    sender_ip = pkt[scapy.ARP].psrc
    target_ip = pkt[scapy.ARP].pdst
    sender_mac = pkt[scapy.ARP].hwsrc
    target_mac = pkt[scapy.ARP].hwdst
    op = pkt[scapy.ARP].op

    prob = process_row(sender_ip, target_ip, sender_mac, target_mac, op, ts)
    if prob >= threshold:
        print(f"[!] Suspicious MAC: {sender_mac} ({sender_ip}) Prob: {prob:.2f}")

def process_csv(file_path, threshold):
    df = pd.read_csv(file_path)
    if "label" in df.columns:
        df = df.drop(columns=["label"])

    print(f"[+] Processing CSV: {file_path}")
    results = defaultdict(list)
    for _, row in df.iterrows():
        prob = process_row(str(row["sender_ip_int"]), str(row["target_ip_int"]), 
                           format(int(row["sender_mac_int"]), 'x'), 
                           format(int(row["target_mac_int"]), 'x'), 
                           2 if row["op_is_reply"] else 1, row.get("ts", time.time()))
        results[row["sender_ip_int"]].append(prob)

    # Average probability per IP
    avg_probs = {ip: np.mean(probs) for ip, probs in results.items()}
    print("\n[+] Average attack probability per sender IP:")
    for ip, prob in sorted(avg_probs.items(), key=lambda x: x[1], reverse=True):
        print(f"{ip}: {prob:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iface", help="Interface for live sniffing")
    parser.add_argument("--csv", help="CSV file with extracted features")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if args.csv:
        process_csv(args.csv, args.threshold)
    elif args.iface:
        scapy.sniff(iface=args.iface, prn=lambda pkt: process_packet(pkt, args.threshold))
    else:
        print("[!] Please specify --iface for live sniffing or --csv for file processing")
