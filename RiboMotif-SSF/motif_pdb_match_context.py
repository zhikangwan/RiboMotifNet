import re
import os
import csv
import sys
import time
from collections import defaultdict
from multiprocessing import Pool, cpu_count

# ================= 配置区域 =================
# 输入文件路径
MOTIF_FILE = 'all_motifs.tsv'
PDB_DBN_FILE = 'pdb_rna_withsecond.fasta'

# 输出文件路径
OUTPUT_PDB_FILE = 'taskA_motif_vs_pdb_with_context.tsv'

# 上下文窗口大小 (bp)
# 例如设置 50，表示提取 Motif 前 50bp + Motif 本身 + 后 50bp
CONTEXT_WINDOW_SIZE = 50 

# 并行进程数
NUM_PROCESSES = cpu_count()
# ============================================

# IUPAC 简并碱基到 Regex 的转换字典
IUPAC_TO_REGEX = {
    'A': 'A', 'C': 'C', 'G': 'G', 'U': 'U', 'T': 'U', 
    'R': '[AG]', 'Y': '[CU]', 'S': '[GC]', 'W': '[AU]', 
    'K': '[GU]', 'M': '[AC]', 'B': '[CGU]', 'D': '[AGU]', 
    'H': '[ACU]', 'V': '[ACG]', 'N': '[ACGU]',
}

# --- 1. 辅助函数 ---

def iupac_to_regex(expand_seq):
    """将包含 IUPAC 简并碱基和自定义 [AC] 格式的序列转换为 Python 正则表达式。"""
    seq = expand_seq.upper().replace('T', 'U') # 统一 T/U
    pattern = ""
    i = 0
    while i < len(seq):
        if seq[i] == '[':
            j = seq.find(']', i)
            if j != -1:
                internal = seq[i:j+1].replace('T', 'U')
                pattern += internal
                i = j + 1
            else:
                pattern += re.escape(seq[i])
                i += 1
        else:
            char = seq[i]
            pattern += IUPAC_TO_REGEX.get(char, re.escape(char))
            i += 1
            
    return pattern

# --- 2. 文件解析函数 ---

def parse_file1(filepath):
    """解析 Motif 定义文件 (File 1)。"""
    motifs = []
    print(f"[{time.strftime('%H:%M:%S')}] 正在解析 File 1...")
    try:
        with open(filepath, 'r', encoding='latin-1') as f:
            reader = csv.reader(f, delimiter='\t')
            try:
                next(reader) # 跳过表头
            except StopIteration:
                return []
            
            # 假设列顺序是 motif_id, motif_seq, expand_seq (索引 0, 1, 2)
            id_col, seq_col, expand_col = 0, 1, 2 
            
            for row in reader:
                if not row or len(row) < 3: continue
                
                motif_id = row[id_col].strip()
                expand_seq = row[expand_col].strip()
                
                regex_pattern = iupac_to_regex(expand_seq)
                
                motifs.append({
                    'id': motif_id,
                    'regex': re.compile(regex_pattern)
                })
        print(f"[{time.strftime('%H:%M:%S')}] File 1 解析完成：{len(motifs)} 个 Motif。")
        return motifs
    except Exception as e:
        print(f"❌ 解析文件 {filepath} 失败：{e}")
        return []

def parse_file2(filepath):
    """
    解析 PDB 序列/DBN 结构文件 (File 2)。
    包含修正后的链名解析逻辑。
    """
    pdb_data = []
    print(f"[{time.strftime('%H:%M:%S')}] 正在解析 File 2...")
    try:
        with open(filepath, 'r', encoding='latin-1') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"❌ 解析文件 {filepath} 失败：{e}")
        return []

    for i in range(0, len(lines), 3):
        if i + 2 >= len(lines): continue
            
        header = lines[i].strip()
        sequence = lines[i+1].strip().upper().replace('T', 'U')
        dbn = lines[i+2].strip()
        
        if header.startswith('>'):
            header_parts = header[1:].split('|')
            pdb_id = header_parts[0] if header_parts else 'UNKNOWN_PDB'
            
            chain_id = 'UNKNOWN' 
            if len(header_parts) > 1:
                chain_info = header_parts[1].strip()
                chain_parts = chain_info.split()
                if len(chain_parts) > 1 and chain_parts[0].lower() == 'chain':
                    chain_id = chain_parts[-1]
                elif chain_info:
                    chain_id = chain_info
            
            pdb_data.append({
                'pdb_id': pdb_id,
                'chain_id': chain_id,
                'sequence': sequence,
                'dbn': dbn,
                'full_seq_len': len(sequence),
                'full_dbn_len': len(dbn)
            })
    print(f"[{time.strftime('%H:%M:%S')}] File 2 解析完成：{len(pdb_data)} 条 PDB 序列。")
    return pdb_data

# --- 3. 核心匹配逻辑 (并行 + 上下文提取) ---

def worker_matcher(args):
    """
    并行工作函数：
    1. 匹配 Motif
    2. 提取匹配的 DBN 片段
    3. 提取匹配位置周围的上下文序列 (Upstream + Motif + Downstream)
    """
    motifs_subset, pdb_data = args
    results = []
    
    for motif in motifs_subset:
        regex = motif['regex']
        motif_id = motif['id']
        
        for entry in pdb_data:
            sequence = entry['sequence']
            dbn = entry['dbn']
            seq_len = entry['full_seq_len']
            
            # 使用 re.finditer 找到所有匹配结果
            for match in regex.finditer(sequence):
                start, end = match.span()
                
                # 1. 提取 DBN 片段 (仅对应 Motif 部分)
                matched_dbn = dbn[start:end] 
                
                # 2. 提取上下文序列
                # 计算切片范围：防止越界 (索引 < 0 或 > seq_len)
                ctx_start = max(0, start - CONTEXT_WINDOW_SIZE)
                ctx_end = min(seq_len, end + CONTEXT_WINDOW_SIZE)
                
                # context_seq 包含了: 上游序列 + Motif本身 + 下游序列
                context_seq = sequence[ctx_start:ctx_end]
                
                results.append({
                    'motif_id': motif_id,
                    'pdb_id': entry['pdb_id'],
                    'chain_id': entry['chain_id'],
                    'match_start_0based': start,
                    'match_end_0based': end - 1, 
                    'matched_seq': match.group(),
                    'matched_dbn': matched_dbn,
                    'context_seq': context_seq, # <--- 新增字段
                    'full_seq_len': seq_len,
                    'full_dbn_len': entry['full_dbn_len']
                })
                
    return results

def run_parallel_matching(motifs, pdb_data):
    """主控函数：设置并行池并运行匹配任务"""
    print(f"[{time.strftime('%H:%M:%S')}] 开始并行匹配任务 (使用 {NUM_PROCESSES} 核心)...")
    
    chunk_size = len(motifs) // NUM_PROCESSES + 1
    motif_chunks = [motifs[i:i + chunk_size] for i in range(0, len(motifs), chunk_size)]
    
    worker_args = [(chunk, pdb_data) for chunk in motif_chunks]
    
    all_results = []
    
    with Pool(processes=NUM_PROCESSES) as pool:
        list_of_result_lists = pool.map(worker_matcher, worker_args)
        for result_list in list_of_result_lists:
            all_results.extend(result_list)
            
    print(f"[{time.strftime('%H:%M:%S')}] 匹配计算完成。")
    return all_results

# --- 4. 文件输出函数 ---

def write_results_to_tsv(filepath, results):
    """将匹配结果写入 TSV 文件。"""
    if not results:
        print(f"✅ 无匹配结果，跳过文件写入 {filepath}。")
        return

    # 定义头部（Header），新增 context_seq
    header = [
        'motif_id', 'pdb_id', 'chain_id', 
        'match_start_0based', 'match_end_0based', 
        'matched_seq', 'matched_dbn', 'context_seq', # <--- 新增列
        'full_seq_len', 'full_dbn_len'
    ]

    try:
        with open(filepath, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header, delimiter='\t')
            
            writer.writeheader()
            writer.writerows(results)
            
        print(f"✅ 结果已写入文件：{filepath} (共 {len(results)} 条记录)")
        
    except Exception as e:
        print(f"❌ 写入文件 {filepath} 失败：{e}")


# --- 5. 主程序 ---

def main():
    print("🚀 启动 File 1 (Motif) 与 File 2 (PDB) 匹配任务 (含上下文提取)...")
    
    if not os.path.exists(MOTIF_FILE) or not os.path.exists(PDB_DBN_FILE):
        print(f"❌ 错误：请确保以下文件存在于当前目录：{MOTIF_FILE} 和 {PDB_DBN_FILE}")
        sys.exit(1)
        
    # 1. 解析输入文件
    motifs = parse_file1(MOTIF_FILE)
    pdb_data = parse_file2(PDB_DBN_FILE)
    
    if not motifs or not pdb_data:
        print("终止：输入数据为空或解析失败。")
        return
    
    # 2. 执行匹配
    pdb_results = run_parallel_matching(motifs, pdb_data)
    
    # 3. 输出结果
    write_results_to_tsv(OUTPUT_PDB_FILE, pdb_results)
    
    print("\n任务全部完成！")

if __name__ == '__main__':
    main()