import sqlite3
import pandas as pd
import os
import datetime
import sys

# ================= 配置区域 =================
DB_OUTPUT_PATH = "/share/home/u2315173006/RNAMotif/RiboMotif-SF/database/RiboMotif-SF.db"

INPUT_FILES = {
    "motifs":  "/share/home/u2315173006/RNAMotif/RiboMotif-SF/all_motifs.tsv",
    "pdb":     "/share/home/u2315173006/RNAMotif/RiboMotif-SF/match_result/taskA_pdb_matches.tsv",
    "func":    "/share/home/u2315173006/RNAMotif/RiboMotif-SF/match_result/motif_have_function.tsv",
    "context": "/share/home/u2315173006/RNAMotif/RiboMotif-SF/match_result/rfam_motif_context_summary.tsv"
}
# ===========================================

def get_connection(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def create_schema(conn):
    cursor = conn.cursor()
    print("[1/4] Initializing Schema...")
    cursor.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)')
    cursor.execute('''CREATE TABLE IF NOT EXISTS motifs (
                        motif_id TEXT PRIMARY KEY, 
                        motif_seq TEXT NOT NULL, 
                        expand_seq TEXT)''')
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS pdb_structures (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        motif_id TEXT, pdb_id TEXT, chain_id TEXT,
                        match_start INTEGER, match_end INTEGER,
                        matched_seq TEXT, matched_dbn TEXT,
                        context_seq TEXT, context_dbn TEXT, full_len INTEGER,
                        FOREIGN KEY(motif_id) REFERENCES motifs(motif_id) ON DELETE CASCADE)''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS functional_annotations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        motif_id TEXT, gene_name TEXT, gene_id TEXT,
                        database_source TEXT, experiment_type TEXT, organism TEXT,
                        instance_seq TEXT, motif_len INTEGER,
                        FOREIGN KEY(motif_id) REFERENCES motifs(motif_id) ON DELETE CASCADE)''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS sequence_instances (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        motif_id TEXT, family_id TEXT, context_seq TEXT,
                        seq_accession TEXT, start INTEGER, end INTEGER,
                        p_value REAL, source_method TEXT,
                        FOREIGN KEY(motif_id) REFERENCES motifs(motif_id) ON DELETE CASCADE)''')
    conn.commit()

def build_id_map(motifs_file):
    """
    核心逻辑：构建 (家族, 序号) -> 全量ID 的映射
    例如: ('RF00014', '1') -> 'RF00014-MEME-1'
    """
    print("[Info] Building ID reconciliation map from motifs file...")
    df = pd.read_csv(motifs_file, sep='\t')
    id_map = {}
    for mid in df['motif_id']:
        # 假设格式是 RFxxxxx-TOOL-INDEX
        parts = mid.split('-')
        if len(parts) >= 3:
            family = parts[0]
            index = parts[-1]
            id_map[(family, index)] = mid
    return id_map

def import_tsv_to_db(file_key, table_name, conn, id_map, col_mapping=None):
    file_path = INPUT_FILES[file_key]
    if not os.path.exists(file_path):
        return

    print(f"Processing {file_key} -> {table_name}...")
    db_cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table_name})")]

    for chunk in pd.read_csv(file_path, sep='\t', chunksize=50000, low_memory=False):
        if col_mapping:
            chunk = chunk.rename(columns=col_mapping)
        
        # --- 特殊处理 context 文件的 ID 对齐 ---
        if file_key == "context":
            def reconstruct_id(row):
                # 提取子表 motif_id 中的序号 (例如从 '1-AGU...' 提取 '1')
                sub_index = str(row['motif_id']).split('-')[0]
                # 从字典中查找匹配的全量母表 ID
                return id_map.get((row['family_id'], sub_index), None)
            
            chunk['motif_id'] = chunk.apply(reconstruct_id, axis=1)
            # 过滤掉无法匹配 ID 的行，确保不违反外键约束
            chunk = chunk.dropna(subset=['motif_id'])
        # ------------------------------------

        valid_cols = [c for c in chunk.columns if c in db_cols]
        chunk_final = chunk[valid_cols]
        
        if table_name == 'motifs':
            chunk_final.to_sql('temp_motifs', conn, if_exists='replace', index=False)
            conn.execute(f"INSERT OR IGNORE INTO {table_name} SELECT * FROM temp_motifs")
            conn.execute("DROP TABLE temp_motifs")
        else:
            chunk_final.to_sql(table_name, conn, if_exists='append', index=False)
            
    conn.commit()

def main():
    print(f"=== RiboMotif-SF Database Builder (Fixed ID Mapping) ===")
    conn = get_connection(DB_OUTPUT_PATH)
    create_schema(conn)

    # 1. 首先构建 ID 映射表
    id_map = build_id_map(INPUT_FILES["motifs"])

    # 2. 顺序导入
    import_tsv_to_db("motifs", "motifs", conn, id_map)
    
    # 导入 pdb
    pdb_map = {'match_start_0based': 'match_start', 'match_end_0based': 'match_end'}
    import_tsv_to_db("pdb", "pdb_structures", conn, id_map, col_mapping=pdb_map)
    
    # 导入功能
    func_map = {'database': 'database_source', 'experiment': 'experiment_type', 'motif_sequence': 'instance_seq', 'motif_length': 'motif_len'}
    import_tsv_to_db("func", "functional_annotations", conn, id_map, col_mapping=func_map)
    
    # 导入分布 (context)
    ctx_map = {'seq_id': 'seq_accession', 'source': 'source_method'}
    import_tsv_to_db("context", "sequence_instances", conn, id_map, col_mapping=ctx_map)

    print("\n[Final] Creating Indices and Vacuuming...")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_inst_motif ON sequence_instances(motif_id)")
    conn.execute("VACUUM")
    conn.close()
    print("Database built successfully.")

if __name__ == "__main__":
    main()