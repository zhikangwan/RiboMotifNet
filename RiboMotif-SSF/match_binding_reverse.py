import csv
import os
import sys
import time
from multiprocessing import Pool, cpu_count

# ================= 配置区域 =================
# 输入文件
FILE1_MOTIF = 'all_motifs.tsv'         # File 1: Motif 定义
FILE5_BINDING = 'Bindingsites_database.tsv'  # File 5: Binding Sites 数据库

# 输出文件
OUTPUT_FILE = 'motif_have_function.tsv'

# 限制每个 File 1 Motif 的最大匹配行数
MAX_MATCHES_PER_MOTIF = 1000

# File 5 序列的最大长度 (根据你的描述是 20bp)
# 任何超过这个长度的 File 1 Motif 都不可能匹配上，直接跳过计算
MAX_FILE5_SEQ_LEN = 25  # 设置稍微宽裕一点，比如 25

# 并行进程数 (默认使用全部 CPU)
NUM_PROCESSES = cpu_count()
# ===========================================

def load_file5_grouped(filepath):
    """
    读取 File 5，按序列内容进行分组。
    返回: 
    1. unique_seqs: 不重复的序列列表 (用于遍历)
    2. seq_db: 字典 {sequence: [raw_line_string, ...]}
    """
    print(f"[{time.strftime('%H:%M:%S')}] 正在加载 File 5 (Binding Sites Database)...")
    
    seq_db = {} 
    total_lines = 0
    
    try:
        # 使用 latin-1 避免编码报错
        with open(filepath, 'r', encoding='latin-1') as f:
            reader = csv.reader(f, delimiter='\t')
            try:
                header = next(reader)
            except StopIteration:
                print("❌ File 5 为空")
                sys.exit(1)
            
            # 自动寻找 motif_sequence 列
            try:
                # 这里假设列名是 'motif_sequence'，如果是其他名字请修改
                # 常见的变体: 'Motif_Sequence', 'sequence', 'consensus'
                # 这里做一个简单的模糊匹配查找
                seq_col_idx = -1
                for i, col_name in enumerate(header):
                    if 'sequence' in col_name.lower() or 'consensus' in col_name.lower():
                        seq_col_idx = i
                        break
                
                if seq_col_idx == -1:
                    # 如果找不到，默认尝试第3列 (索引2)，根据你提供的样例
                    seq_col_idx = 2 
                    print(f"⚠️ 警告: 未在表头找到 'motif_sequence'，尝试使用第 {seq_col_idx+1} 列作为序列列。")

            except ValueError:
                print("❌ 错误：无法确定序列列")
                sys.exit(1)
                
            for row in reader:
                if not row: continue
                total_lines += 1
                
                # 获取序列并标准化 (大写, T->U)
                if len(row) > seq_col_idx:
                    seq = row[seq_col_idx].strip().upper().replace('T', 'U')
                    
                    if seq not in seq_db:
                        seq_db[seq] = []
                    
                    # 内存优化: 仅当该序列存储的行数少于 MAX_MATCHES_PER_MOTIF * 1.5 时才继续存
                    # 避免对于某些极度常见的序列存储了百万行数据，占用过多内存
                    if len(seq_db[seq]) < MAX_MATCHES_PER_MOTIF + 500:
                        seq_db[seq].append('\t'.join(row))
                
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 {filepath}")
        sys.exit(1)

    unique_seqs = list(seq_db.keys())
    print(f"[{time.strftime('%H:%M:%S')}] File 5 加载完毕。")
    print(f"  -> 总行数: {total_lines}")
    print(f"  -> 不重复序列数: {len(unique_seqs)}")
    
    return header, unique_seqs, seq_db

def load_file1(filepath):
    """读取 File 1 的 Motif 数据"""
    motifs = []
    try:
        with open(filepath, 'r', encoding='latin-1') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader)
            
            # 假设前三列是 motif_id, motif_seq, expand_seq
            # 根据你之前的描述
            
            for row in reader:
                if len(row) < 2: continue
                
                motif_seq = row[1].strip().upper().replace('T', 'U')
                
                motifs.append({
                    'seq': motif_seq,
                    'raw_row': '\t'.join(row)
                })
    except Exception as e:
        print(f"❌ 读取 File 1 失败: {e}")
        sys.exit(1)
        
    print(f"[{time.strftime('%H:%M:%S')}] File 1 加载完毕: {len(motifs)} 个 Motif。")
    return header, motifs

def worker_matcher(args):
    """
    并行 Worker: 检查 file1_seq 是否存在于 unique_file5_seqs 中
    """
    subset_motifs, unique_file5_seqs = args
    results = [] # 格式: [(motif_raw_line, [matched_file5_seqs])]
    
    for motif in subset_motifs:
        m_seq = motif['seq']
        m_len = len(m_seq)
        
        # --- 核心优化: 长度过滤 ---
        # 如果 File 1 的 motif 比 File 5 可能的最长序列还长，那绝不可能匹配上
        # 你的 File 5 只有 4-20bp，File 1 如果是 30bp，直接跳过
        if m_len > MAX_FILE5_SEQ_LEN:
            continue
            
        matched_seqs_for_this_motif = []
        count_estimate = 0
        
        for db_seq in unique_file5_seqs:
            # --- 核心逻辑: File 1 in File 5 ---
            # 仅当 File 1 长度 <= File 5 长度时才有可能
            if m_len <= len(db_seq):
                if m_seq in db_seq:
                    matched_seqs_for_this_motif.append(db_seq)
                    # 估算匹配数量，用于截断 (假设平均每个 seq 对应 2 行)
                    count_estimate += 1
                    if count_estimate > MAX_MATCHES_PER_MOTIF:
                        break
        
        if matched_seqs_for_this_motif:
            results.append((motif['raw_row'], matched_seqs_for_this_motif))
            
    return results

def main():
    print(f"🚀 开始反向匹配任务 (File 1 in File 5) - 使用 {NUM_PROCESSES} 核心...")
    
    # 1. 加载数据
    f5_header, unique_seqs, seq_db = load_file5_grouped(FILE5_BINDING)
    f1_header, motifs = load_file1(FILE1_MOTIF)
    
    # 2. 准备并行数据
    print(f"[{time.strftime('%H:%M:%S')}] 开始并行计算...")
    
    # 将 Motifs 切块
    chunk_size = len(motifs) // NUM_PROCESSES + 1
    motif_chunks = [motifs[i:i + chunk_size] for i in range(0, len(motifs), chunk_size)]
    
    worker_args = [(chunk, unique_seqs) for chunk in motif_chunks]
    
    total_matches_written = 0
    
    # 3. 执行并行并写入
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        # 写入合并表头
        combined_header = '\t'.join(f1_header + f5_header) + '\n'
        f_out.write(combined_header)
        
        with Pool(processes=NUM_PROCESSES) as pool:
            # 使用 imap_unordered 实时获取结果
            iterator = pool.imap_unordered(worker_matcher, worker_args)
            
            for result_chunk in iterator:
                # 处理每个 Worker 返回的一批结果
                for motif_raw, matched_seqs in result_chunk:
                    
                    match_count = 0
                    
                    for seq in matched_seqs:
                        # 从主进程的大字典中获取 File 5 的原始行
                        file5_lines = seq_db.get(seq, [])
                        
                        # 再次执行精确截断
                        remaining = MAX_MATCHES_PER_MOTIF - match_count
                        if remaining <= 0: break
                        
                        to_write = file5_lines[:remaining]
                        
                        for line in to_write:
                            f_out.write(f"{motif_raw}\t{line}\n")
                            
                        match_count += len(to_write)
                        
                    if match_count > 0:
                        total_matches_written += 1

    print(f"[{time.strftime('%H:%M:%S')}] 任务完成！")
    print(f"✅ 共有 {total_matches_written} 个 File 1 Motif 找到了匹配。")
    print(f"✅ 结果已保存至: {OUTPUT_FILE}")

if __name__ == '__main__':
    # 简单的文件检查
    if not os.path.exists(FILE1_MOTIF) or not os.path.exists(FILE5_BINDING):
        print(f"❌ 错误：请确保当前目录下存在 {FILE1_MOTIF} 和 {FILE5_BINDING}")
    else:
        main()