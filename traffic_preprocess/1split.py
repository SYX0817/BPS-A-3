import os
import glob
from tqdm import tqdm

raw_pcap_path = "1raw_traffic"
flow_pcap_path = "2session"
processed_record_path = "processed_dirs2.txt"
os.makedirs(flow_pcap_path, exist_ok=True)

# 读取已处理的子目录列表
if os.path.exists(processed_record_path):
    with open(processed_record_path, "r") as f:
        processed_dirs = set(line.strip() for line in f if line.strip())
else:
    processed_dirs = set()

def run_splitcap_for_dir(cla):
    """对一个目录下的所有pcap文件执行SplitCap"""
    print(f"\nProcessing directory: {cla}")  # 输出正在处理的子目录名称

    flow_dir = cla.replace(raw_pcap_path, flow_pcap_path)
    os.makedirs(flow_dir, exist_ok=True)

    pcaps = glob.glob(os.path.join(cla, "*.pcap"))
    for pcap in pcaps:
        command = f'SplitCap.exe -r "{pcap}" -s session -o "{flow_dir}"'
        os.system(command)

    # 处理完写入记录文件
    with open(processed_record_path, "a") as f:
        f.write(cla + "\n")

# 所有类目录
all_class_dirs = glob.glob(os.path.join(raw_pcap_path, "*"))
# 过滤掉已处理的目录
unprocessed_dirs = [d for d in all_class_dirs if d not in processed_dirs]

# 串行处理每个目录
for cla in tqdm(unprocessed_dirs, desc="Splitting PCAP dirs"):
    run_splitcap_for_dir(cla)
