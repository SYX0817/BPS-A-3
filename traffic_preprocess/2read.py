import os
import glob
import binascii
import scapy.all as scapy
from tqdm import tqdm
import pandas as pd
from scapy.layers.inet import IP, UDP
from scapy.packet import Raw
import argparse

def ip_to_int(ip):
    return sum([int(x) << (24 - i * 8) for i, x in enumerate(ip.split('.'))])

def get_direction(sip, dip):
    return 1 if ip_to_int(sip) < ip_to_int(dip) else -1

def extract_header_payload(packet, head_len=160, payload_len=480):
    packet_hex = binascii.hexlify(bytes(packet)).decode()
    raw_layer = packet.getlayer(Raw)
    payload_hex = binascii.hexlify(raw_layer.load).decode() if raw_layer else ''
    header_hex = packet_hex.replace(payload_hex, '', 1)
    header_hex = (header_hex + '0' * head_len)[:head_len]
    payload_hex = (payload_hex + '0' * payload_len)[:payload_len]
    return header_hex, payload_hex

def is_valid_packet(packet):
    skip_layers = ["IPv6", "ARP", "DNS", "STUN", "DHCP", "DHCPv6", "ICMP", "ICMPv6", "NTP", "IGMP"]
    for layer in skip_layers:
        if packet.haslayer(layer):
            return False
    if packet.haslayer(UDP):
        if packet[UDP].sport in [5355, 5353, 137] or packet[UDP].dport in [5355, 5353, 137]:
            return False
    return True

def build_singleflow(flow_packets, label, pack_num=5, head_num=160, payload_num=480):
    sip = flow_packets[0].getlayer(IP).src
    dip = flow_packets[0].getlayer(IP).dst
    sport = flow_packets[0].sport
    dport = flow_packets[0].dport

    byte_len = (head_num + payload_num) // 2  # 每个包的字节长度
    singleflow = {
        "sip": sip,
        "dip": dip,
        "sport": sport,
        "dport": dport,
        "fingerprint": [0] * pack_num,
        "bytes": [[0] * byte_len for _ in range(pack_num)],  # ✅ 每行独立 + 长度正确
        "direction_list": [0] * pack_num,
        "label": label
    }

    for i, pkt in enumerate(flow_packets):
        header, payload = extract_header_payload(pkt, head_num, payload_num)
        combined = header + payload
        singleflow['bytes'][i] = [int(combined[k:k + 2], 16) for k in range(0, len(combined), 2)]
        singleflow['fingerprint'][i] = pkt.len
        singleflow['direction_list'][i] = get_direction(pkt[IP].src, pkt[IP].dst)

    return singleflow

def pac2dict(pcap_path, label, pack_num=5, head_num=160, payload_num=480,
             max_flows_per_pcap=None, filter_packets=True, min_packets_per_flow=None):
    packets = scapy.rdpcap(pcap_path)
    dict_flows = []
    flow_count = 0
    buffer = []

    for packet in packets:
        if filter_packets and not is_valid_packet(packet):
            continue
        buffer.append(packet)

        while len(buffer) >= pack_num:
            flow_packets = buffer[:pack_num]
            buffer = buffer[pack_num:]
            if min_packets_per_flow is not None and len(flow_packets) < min_packets_per_flow:
                continue
            singleflow = build_singleflow(flow_packets, label, pack_num, head_num, payload_num)
            dict_flows.append(singleflow)
            flow_count += 1
            if max_flows_per_pcap and flow_count >= max_flows_per_pcap:
                return dict_flows

    if 0 < len(buffer) < pack_num:
        if min_packets_per_flow is None or len(buffer) >= min_packets_per_flow:
            singleflow = build_singleflow(buffer, label, pack_num, head_num, payload_num)
            dict_flows.append(singleflow)

    return dict_flows

def process_and_save(file_path, label, output_dir, max_flows_per_pcap, pack_num,
                     head_num, payload_num, filter_packets, min_packets_per_flow):
    try:
        flows = pac2dict(file_path, label, pack_num, head_num, payload_num,
                         max_flows_per_pcap, filter_packets, min_packets_per_flow)
        if not flows:
            return 0
        out_csv = os.path.join(output_dir, os.path.basename(file_path) + ".csv")
        pd.DataFrame(flows).to_csv(out_csv, index=False)
        return len(flows)
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract flows from PCAP files.")
    # parser.add_argument('--flows_pcap_path', type=str, default='2session')
    # parser.add_argument('--output_path', type=str, default='ciciot', help='Directory to store output CSVs')
    parser.add_argument('--flows_pcap_path', type=str, default='a')
    parser.add_argument('--output_path', type=str, default='aa', help='Directory to store output CSVs')
    parser.add_argument('--max_flows_per_class', type=int, default=1000, help='Max number of flows per class')
    parser.add_argument('--max_flows_per_pcap', type=int, default=1, help='Max number of flows per PCAP file')
    parser.add_argument('--pack_num', type=int, default=5, help='Number of packets per flow')
    parser.add_argument('--head_num', type=int, default=160, help='Header length per packet')
    parser.add_argument('--payload_num', type=int, default=480, help='Payload length per packet')
    parser.add_argument('--filter_packets', default=True, help='Disable protocol filtering')
    parser.add_argument('--min_packets_per_flow', type=int, default=None,
                        help='Minimum number of packets per flow to keep (set to None to disable)')
    args = parser.parse_args()

    flows_pcap_path = args.flows_pcap_path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    max_flows_per_class = args.max_flows_per_class
    max_flows_per_pcap = args.max_flows_per_pcap
    pack_num = args.pack_num
    head_num = args.head_num
    payload_num = args.payload_num
    filter_packets = args.filter_packets
    min_packets_per_flow = args.min_packets_per_flow


    processed_record_file = os.path.join(output_path, "processed.txt")
    processed_files = set()
    if os.path.exists(processed_record_file):
        with open(processed_record_file, 'r') as f:
            processed_files = set(f.read().splitlines())

    all_data = []
    classes = glob.glob(os.path.join(flows_pcap_path, "*"))

    for label, cla in enumerate(tqdm(classes, desc="Classes")):
        filenames = glob.glob(os.path.join(cla, "*.pcap"))
        out_class_dir = os.path.join(output_path, os.path.basename(cla))
        os.makedirs(out_class_dir, exist_ok=True)

        total_flow_count = 0
        for file_path in tqdm(filenames, desc=f"Processing {os.path.basename(cla)}"):
            if file_path in processed_files:
                continue
            if os.path.getsize(file_path) > 1 * 1024 * 1024 * 1024:
                print(f"⚠️ Skipped {file_path} due to large size >1GB")
                continue
            if max_flows_per_class is not None and total_flow_count >= max_flows_per_class:
                break

            result = process_and_save(
                file_path, label, out_class_dir,
                max_flows_per_pcap, pack_num, head_num, payload_num,
                filter_packets, min_packets_per_flow
            )

            if result:
                total_flow_count += result
                with open(processed_record_file, "a") as f:
                    f.write(file_path + "\n")
            if max_flows_per_class is not None and total_flow_count >= max_flows_per_class:
                break

        class_csvs = glob.glob(os.path.join(out_class_dir, "*.csv"))
        dfs = [pd.read_csv(c) for c in class_csvs]
        if dfs:
            class_df = pd.concat(dfs, ignore_index=True)
            class_df_to_save = class_df if max_flows_per_class is None else class_df[:max_flows_per_class]
            all_data.append(class_df_to_save)
            out_path = os.path.join(output_path, os.path.basename(cla) + ".csv")
            class_df_to_save.to_csv(out_path, index=False)

    if all_data:
        all_df = pd.concat(all_data, ignore_index=True)
        all_df.to_csv(output_path + ".csv", index=False)
