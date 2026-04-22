import json
import pandas as pd
import os
import glob

def main():
    base_dir = "/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment"
    output_excel = "o3_domain_accuracy_trend.xlsx"
    
    seeds = [0, 1, 2]
    task_domains = [
        "work", "travel", "home", "banking", "credit_cards",
        "kitchen_and_dining", "auto_and_commute", "utility", "small_talk"
    ]
    
    all_data = []

    for seed in seeds:
        seed_dir_name = f"SCALE_0.1_seed_{seed}_o_unlearn_lora_force_checkpoints_clinc9_origin"
        seed_dir = os.path.join(base_dir, seed_dir_name)
        
        if not os.path.exists(seed_dir):
            print(f"Warning: Seed directory not found: {seed_dir}")
            continue
            
        for t_idx, task_domain in enumerate(task_domains):
            # Filename pattern: test_seed{seed}_{task_domain}_testSD.json
            json_filename = f"test_seed{seed}_{task_domain}_testSD.json"
            json_path = os.path.join(seed_dir, json_filename)
            
            if not os.path.exists(json_path):
                print(f"Warning: JSON not found: {json_path}")
                continue
            
            # Process JSON
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'results' not in data:
                continue
                
            results = data['results']
            
            for entry in results:
                eval_domain = entry.get('domain')
                # We can use 'exact_correct' or 'contains_correct'. 
                # Let's use 'exact_correct' for main metric as it's more specific.
                is_correct = entry.get('exact_correct', False)
                contains_correct = entry.get('contains_correct', False)
                
                all_data.append({
                    'seed': seed,
                    'task_idx': t_idx,
                    'task_domain': task_domain,
                    'eval_domain': eval_domain,
                    'is_correct': is_correct,
                    'contains_correct': contains_correct
                })

    if not all_data:
        print("No data extracted. Please check directory paths and filenames.")
        return

    df = pd.DataFrame(all_data)
    
    # Calculate accuracy per (seed, task_idx, eval_domain)
    # Using 'is_correct' (exact_correct) for trend
    accuracy_per_run = df.groupby(['seed', 'task_idx', 'eval_domain'])['is_correct'].mean().reset_index()
    accuracy_per_run.rename(columns={'is_correct': 'accuracy'}, inplace=True)
    
    # Average across seeds
    trend_df = accuracy_per_run.groupby(['task_idx', 'eval_domain'])['accuracy'].mean().reset_index()
    
    # Pivot for Excel
    trend_df['task_label'] = trend_df['task_idx'].apply(lambda x: f"t{x}")
    pivot_df = trend_df.pivot(index='eval_domain', columns='task_label', values='accuracy')
    pivot_df = pivot_df.round(6)
    pd.set_option('display.float_format', '{:.6f}'.format)
    
    # Reorder columns (t0 to t8)
    cols = [f"t{i}" for i in range(len(task_domains))]
    existing_cols = [c for c in cols if c in pivot_df.columns]
    pivot_df = pivot_df[existing_cols]
    
    # Sort index by task_domains order
    domain_order = {d: i for i, d in enumerate(task_domains)}
    pivot_df = pivot_df.reindex(sorted(pivot_df.index, key=lambda x: domain_order.get(x, 999)))

    print("\nO3 Accuracy Trend (Average across seeds, Exact Acc):")
    print(pivot_df)

    # Save to Excel
    try:
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            pivot_df.to_excel(writer, sheet_name='Exact Acc Trend')
            
            # Also add Contains Acc Trend for completeness
            cont_accuracy_per_run = df.groupby(['seed', 'task_idx', 'eval_domain'])['contains_correct'].mean().reset_index()
            cont_accuracy_per_run.rename(columns={'contains_correct': 'accuracy'}, inplace=True)
            cont_trend_df = cont_accuracy_per_run.groupby(['task_idx', 'eval_domain'])['accuracy'].mean().reset_index()
            cont_trend_df['task_label'] = cont_trend_df['task_idx'].apply(lambda x: f"t{x}")
            cont_pivot_df = cont_trend_df.pivot(index='eval_domain', columns='task_label', values='accuracy')
            cont_pivot_df = cont_pivot_df[existing_cols]
            cont_pivot_df = cont_pivot_df.reindex(sorted(cont_pivot_df.index, key=lambda x: domain_order.get(x, 999)))
            cont_pivot_df = cont_pivot_df.round(6)
            cont_pivot_df.to_excel(writer, sheet_name='Contains Acc Trend')
            
            trend_df.round(6).to_excel(writer, sheet_name='Detailed Trend', index=False)
            accuracy_per_run.round(6).to_excel(writer, sheet_name='Per Seed Exact Accuracy', index=False)

            # Apply number format (6 decimal places) to all sheets
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                for row in worksheet.iter_rows(min_row=2, max_row=worksheet.max_row, min_col=2, max_col=worksheet.max_column):
                    for cell in row:
                        if isinstance(cell.value, (int, float)):
                            cell.number_format = '0.000000'
            
        print(f"\nTrend results saved to {output_excel}")
    except Exception as e:
        print(f"Error saving to Excel: {e}")

if __name__ == "__main__":
    main()
