import pandas as pd
import re
import os

# 파일 경로 설정
input_file = '/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/modify_code/clincqa_moe_second_intent_selectedCE/task_dataset_metrics.xlsx'
output_file = '/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/modify_code/clincqa_moe_second_intent_selectedCE/reshaped_metrics.xlsx'

# 엑셀 파일 읽기 (sheet_name='task_dataset_table'은 plot_results.py에서 생성하는 기본 시트명)
try:
    df = pd.read_excel(input_file, sheet_name='task_dataset_table')
except Exception as e:
    print(f"Error reading excel: {e}")
    # 시트명이 다를 경우 첫 번째 시트 사용 시도
    df = pd.read_excel(input_file)

# 대상 데이터셋 필터링
target_datasets = ['forget', 'retain', 'utility1', 'utility2']
df_filtered = df[df['dataset'].isin(target_datasets)].copy()

# task에서 t 값 추출 (예: t0, t1 -> 0, 1)
def extract_t(task_str):
    if pd.isna(task_str): return None
    match = re.search(r't(\d+)', str(task_str))
    if match:
        return int(match.group(1))
    return task_str

df_filtered['t'] = df_filtered['task'].apply(extract_t)

# 피벗팅할 메트릭들 (원하는 메트릭을 리스트에 추가하세요)
metrics = ['choice_accuracy', 'gt_loss', 'perplexity', 'mrr']

# 결과 저장을 위한 ExcelWriter
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    for metric in metrics:
        if metric in df_filtered.columns:
            # 피벗: index는 dataset, columns는 t, 값은 metric
            pivot_df = df_filtered.pivot_table(index='dataset', columns='t', values=metric)
            
            # 행 순서 보장: forget -> retain -> utility1 -> utility2
            available_rows = [r for r in target_datasets if r in pivot_df.index]
            pivot_df = pivot_df.reindex(available_rows)
            
            # 소수점 6자리까지 저장 (데이터 자체 반올림)
            pivot_df = pivot_df.round(6)
            
            # 시트 저장
            pivot_df.to_excel(writer, sheet_name=metric)
            
            # 셀 포맷 설정 (표시 형식도 소수점 6자리로 강제)
            ws = writer.sheets[metric]
            for row in ws.iter_rows(min_row=2, min_col=2): # 헤더와 인덱스 제외 필드
                for cell in row:
                    if isinstance(cell.value, (int, float)):
                        cell.number_format = '0.000000'
            
            print(f"Saved metric sheet: {metric} (rounded to 6 decimal places)")

print(f"\nDone! Reshaped file saved at: {output_file}")
