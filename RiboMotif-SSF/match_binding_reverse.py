import csv
import os
import sys
import time
from multiprocessing import Pool, cpu_count


FILE1_MOTIF = 'all_motifs.tsv'         
FILE5_BINDING = 'Bindingsites_database.tsv'  
OUTPUT_FILE = 'motif_have_function.tsv'

MAX_MATCHES_PER_MOTIF = 1000


MAX_FILE5_SEQ_LEN = 25  


NUM_PROCESSES = cpu_count()

def load_file5_grouped(filepath):

    print(f"[{time.strftime('%H:%M:%S')}] loading File 5 (Binding Sites Database)...")
    
    seq_db = {} 
    total_lines = 0
    
    try:
        with open(filepath, 'r', encoding='latin-1') as f:
            reader = csv.reader(f, delimiter='\t')
            try:
                header = next(reader)
            except StopIteration:
                print("❌ File 5 is none")
                sys.exit(1)
            
            try:
                seq_col_idx = -1
                for i, col_name in enumerate(header):
                    if 'sequence' in col_name.lower() or 'consensus' in col_name.lower():
                        seq_col_idx = i
                        break
                
                if seq_col_idx == -1:
                    seq_col_idx = 2 
                    print(f"Error:have do not find 'motif_sequence'，try use  {seq_col_idx+1} 。")

            except ValueError:
                print("Error: Unable to determine sequence number")
                sys.exit(1)
                
            for row in reader:
                if not row: continue
                total_lines += 1
                
                if len(row) > seq_col_idx:
                    seq = row[seq_col_idx].strip().upper().replace('T', 'U')
                    
                    if seq not in seq_db:
                        seq_db[seq] = []
                    
                    if len(seq_db[seq]) < MAX_MATCHES_PER_MOTIF + 500:
                        seq_db[seq].append('\t'.join(row))
                
    except FileNotFoundError:
        print(f"Error: File {filepath} not found")
        sys.exit(1)

    unique_seqs = list(seq_db.keys())
    print(f"[{time.strftime('%H:%M:%S')}] File 5 has finished loading.")
    print(f"  -> Total number of rows: {total_lines}")
    print(f"  -> Number of unique sequences: {len(unique_seqs)}")
    
    return header, unique_seqs, seq_db

def load_file1(filepath):
    motifs = []
    try:
        with open(filepath, 'r', encoding='latin-1') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader)
            
            for row in reader:
                if len(row) < 2: continue
                
                motif_seq = row[1].strip().upper().replace('T', 'U')
                
                motifs.append({
                    'seq': motif_seq,
                    'raw_row': '\t'.join(row)
                })
    except Exception as e:
        print(f"Failed to read File 1: {e}")
        sys.exit(1)
        
    print(f"[{time.strftime('%H:%M:%S')}] File 1 loaded: {len(motifs)} motifs.")
    return header, motifs

def worker_matcher(args):
    subset_motifs, unique_file5_seqs = args
    results = [] 
    
    for motif in subset_motifs:
        m_seq = motif['seq']
        m_len = len(m_seq)
        
        if m_len > MAX_FILE5_SEQ_LEN:
            continue
            
        matched_seqs_for_this_motif = []
        count_estimate = 0
        
        for db_seq in unique_file5_seqs:
            if m_len <= len(db_seq):
                if m_seq in db_seq:
                    matched_seqs_for_this_motif.append(db_seq)
                    count_estimate += 1
                    if count_estimate > MAX_MATCHES_PER_MOTIF:
                        break
        
        if matched_seqs_for_this_motif:
            results.append((motif['raw_row'], matched_seqs_for_this_motif))
            
    return results

def main():
    print(f"Start reverse matching task (File 1 in File 5) - using the {NUM_PROCESSES} core...")
    
    f5_header, unique_seqs, seq_db = load_file5_grouped(FILE5_BINDING)
    f1_header, motifs = load_file1(FILE1_MOTIF)
    
    print(f"[{time.strftime('%H:%M:%S')}]Parallel computation begins...")
    
    chunk_size = len(motifs) // NUM_PROCESSES + 1
    motif_chunks = [motifs[i:i + chunk_size] for i in range(0, len(motifs), chunk_size)]
    
    worker_args = [(chunk, unique_seqs) for chunk in motif_chunks]
    
    total_matches_written = 0
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        combined_header = '\t'.join(f1_header + f5_header) + '\n'
        f_out.write(combined_header)
        
        with Pool(processes=NUM_PROCESSES) as pool:
            iterator = pool.imap_unordered(worker_matcher, worker_args)
            
            for result_chunk in iterator:
                for motif_raw, matched_seqs in result_chunk:
                    
                    match_count = 0
                    
                    for seq in matched_seqs:
                        file5_lines = seq_db.get(seq, [])
                        
                        remaining = MAX_MATCHES_PER_MOTIF - match_count
                        if remaining <= 0: break
                        
                        to_write = file5_lines[:remaining]
                        
                        for line in to_write:
                            f_out.write(f"{motif_raw}\t{line}\n")
                            
                        match_count += len(to_write)
                        
                    if match_count > 0:
                        total_matches_written += 1

    print(f"[{time.strftime('%H:%M:%S')}] Mission accomplished!")
    print(f"A total of {total_matches_written} File 1 Motif matches were found.")
    print(f" result have saved to: {OUTPUT_FILE}")

if __name__ == '__main__':
    # 简单的文件检查
    if not os.path.exists(FILE1_MOTIF) or not os.path.exists(FILE5_BINDING):
        print(f"Error: Please ensure that {FILE1_MOTIF} and {FILE5_BINDING} exist in the current directory.")
    else:
        main()