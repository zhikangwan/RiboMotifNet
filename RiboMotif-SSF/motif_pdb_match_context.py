import re
import os
import csv
import sys
import time
from collections import defaultdict
from multiprocessing import Pool, cpu_count

MOTIF_FILE = 'all_motifs.tsv'
PDB_DBN_FILE = 'pdb_rna_withsecond.fasta'

OUTPUT_PDB_FILE = 'taskA_motif_vs_pdb_with_context.tsv'

CONTEXT_WINDOW_SIZE = 50 

NUM_PROCESSES = cpu_count()
IUPAC_TO_REGEX = {
    'A': 'A', 'C': 'C', 'G': 'G', 'U': 'U', 'T': 'U', 
    'R': '[AG]', 'Y': '[CU]', 'S': '[GC]', 'W': '[AU]', 
    'K': '[GU]', 'M': '[AC]', 'B': '[CGU]', 'D': '[AGU]', 
    'H': '[ACU]', 'V': '[ACG]', 'N': '[ACGU]',
}



def iupac_to_regex(expand_seq):
    seq = expand_seq.upper().replace('T', 'U') 
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


def parse_file1(filepath):
    motifs = []
    print(f"[{time.strftime('%H:%M:%S')}] Parsing File 1...")
    try:
        with open(filepath, 'r', encoding='latin-1') as f:
            reader = csv.reader(f, delimiter='\t')
            try:
                next(reader) 
            except StopIteration:
                return []
            
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
        print(f"[{time.strftime('%H:%M:%S')}] File 1 parsed: {len(motifs)} motifs.")
        return motifs
    except Exception as e:
        print(f" Failed to parse file {filepath}: {e}")
        return []

def parse_file2(filepath):

    pdb_data = []
    print(f"[{time.strftime('%H:%M:%S')}] Parsing File 2...")
    try:
        with open(filepath, 'r', encoding='latin-1') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Failed to parse file {filepath} ：{e}")
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
    print(f"[{time.strftime('%H:%M:%S')}] File 2 parsing complete: {len(pdb_data)} PDB sequences")
    return pdb_data


def worker_matcher(args):

    motifs_subset, pdb_data = args
    results = []
    
    for motif in motifs_subset:
        regex = motif['regex']
        motif_id = motif['id']
        
        for entry in pdb_data:
            sequence = entry['sequence']
            dbn = entry['dbn']
            seq_len = entry['full_seq_len']
            

            for match in regex.finditer(sequence):
                start, end = match.span()
                

                matched_dbn = dbn[start:end] 
                

                ctx_start = max(0, start - CONTEXT_WINDOW_SIZE)
                ctx_end = min(seq_len, end + CONTEXT_WINDOW_SIZE)
                

                context_seq = sequence[ctx_start:ctx_end]
                
                results.append({
                    'motif_id': motif_id,
                    'pdb_id': entry['pdb_id'],
                    'chain_id': entry['chain_id'],
                    'match_start_0based': start,
                    'match_end_0based': end - 1, 
                    'matched_seq': match.group(),
                    'matched_dbn': matched_dbn,
                    'context_seq': context_seq, 
                    'full_seq_len': seq_len,
                    'full_dbn_len': entry['full_dbn_len']
                })
                
    return results

def run_parallel_matching(motifs, pdb_data):

    print(f"[{time.strftime('%H:%M:%S')}] Start parallel matching task (using the {NUM_PROCESSES} core)...")
    
    chunk_size = len(motifs) // NUM_PROCESSES + 1
    motif_chunks = [motifs[i:i + chunk_size] for i in range(0, len(motifs), chunk_size)]
    
    worker_args = [(chunk, pdb_data) for chunk in motif_chunks]
    
    all_results = []
    
    with Pool(processes=NUM_PROCESSES) as pool:
        list_of_result_lists = pool.map(worker_matcher, worker_args)
        for result_list in list_of_result_lists:
            all_results.extend(result_list)
            
    print(f"[{time.strftime('%H:%M:%S')}] Matching calculation complete.")
    return all_results


def write_results_to_tsv(filepath, results):
    if not results:
        print(f"No matching results found. Skip writing to {filepath}.")
        return

    header = [
        'motif_id', 'pdb_id', 'chain_id', 
        'match_start_0based', 'match_end_0based', 
        'matched_seq', 'matched_dbn', 'context_seq', 
        'full_seq_len', 'full_dbn_len'
    ]

    try:
        with open(filepath, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header, delimiter='\t')
            
            writer.writeheader()
            writer.writerows(results)
            
        print(f" The results have been written to the file: {filepath} (containing {len(results)} records).")
        
    except Exception as e:
        print(f" Writing to file {filepath} failed: {e}")


# --- 5. 主程序 ---

def main():
    print("Start the File 1 (Motif) and File 2 (PDB) matching task (including context extraction)...")
    
    if not os.path.exists(MOTIF_FILE) or not os.path.exists(PDB_DBN_FILE):
        print(f" Error: Please ensure the following files exist in the current directory: {MOTIF_FILE} and {PDB_DBN_FILE}")
        sys.exit(1)
        

    motifs = parse_file1(MOTIF_FILE)
    pdb_data = parse_file2(PDB_DBN_FILE)
    
    if not motifs or not pdb_data:
        print("Termination: Input data is empty or parsing failed.")
        return

    pdb_results = run_parallel_matching(motifs, pdb_data)
    

    write_results_to_tsv(OUTPUT_PDB_FILE, pdb_results)
    
    print("\nAll tasks completed!")

if __name__ == '__main__':
    main()