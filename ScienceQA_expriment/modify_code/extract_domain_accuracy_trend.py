import json
import pandas as pd
import os
import glob

def main():
    base_dir = "/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/modify_code/run_results/eval_basemoe_error/llama3.1-8b-instruct/clinc_moe_fixed"
    output_excel = "domain_accuracy_trend.xlsx"
    
    seeds = [0, 1, 2]
    task_domains = [
        "work", "travel", "home", "banking", "credit_cards",
        "kitchen_and_dining", "auto_and_commute", "utility", "small_talk"
    ]
    
    all_data = []

    for seed in seeds:
        seed_dir_suffix = f"clincqa_moe_fixed_layer16_{seed}"
        seed_dir = os.path.join(base_dir, seed_dir_suffix)
        
        if not os.path.exists(seed_dir):
            print(f"Warning: Seed directory not found: {seed_dir}")
            continue
            
        for t_idx, domain_name in enumerate(task_domains):
            task_dir_pattern = f"lunar_llama3.1-8b-instruct_eval_base_clincqa_t{t_idx}_{domain_name}"
            task_dir = os.path.join(seed_dir, task_dir_pattern)
            
            if not os.path.exists(task_dir):
                # Try to find it with glob if the name is slightly different
                matches = glob.glob(os.path.join(seed_dir, f"*_t{t_idx}_*"))
                if matches:
                    task_dir = matches[0]
                else:
                    print(f"Warning: Task directory not found: {task_dir}")
                    continue
            
            json_path = os.path.join(task_dir, f"forget_clincqa_t{t_idx}_{domain_name}.json")
            if not os.path.exists(json_path):
                # Try to find any json starting with forget_ in that dir
                jsons = glob.glob(os.path.join(task_dir, "forget_*.json"))
                if jsons:
                    json_path = jsons[0]
                else:
                    print(f"Warning: JSON not found in {task_dir}")
                    continue
            
            # Process JSON
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'forget' not in data or 'results' not in data['forget'] or 'generated_text' not in data['forget']['results']:
                continue
                
            generated_text_list = data['forget']['results']['generated_text']
            
            for entry in generated_text_list:
                if len(entry) < 4:
                    continue
                gen_text = entry[1]
                gt_label = entry[2]
                eval_domain = entry[3]
                
                is_correct = gt_label.lower() in gen_text.lower()
                
                all_data.append({
                    'seed': seed,
                    'task_idx': t_idx,
                    'task_domain': domain_name,
                    'eval_domain': eval_domain,
                    'is_correct': is_correct
                })

    if not all_data:
        print("No data extracted. Please check directory paths.")
        return

    df = pd.DataFrame(all_data)
    
    # Calculate accuracy per (seed, task_idx, eval_domain)
    accuracy_per_run = df.groupby(['seed', 'task_idx', 'eval_domain'])['is_correct'].mean().reset_index()
    accuracy_per_run.rename(columns={'is_correct': 'accuracy'}, inplace=True)
    
    # Average across seeds
    trend_df = accuracy_per_run.groupby(['task_idx', 'eval_domain'])['accuracy'].mean().reset_index()
    
    # Pivot for Excel
    # Rows: eval_domain, Cols: task_idx (represented as t0, t1, ...)
    trend_df['task_label'] = trend_df['task_idx'].apply(lambda x: f"t{x}")
    pivot_df = trend_df.pivot(index='eval_domain', columns='task_label', values='accuracy')
    pivot_df = pivot_df.round(9)
    pd.set_option('display.float_format', '{:.9f}'.format)
    
    # Reorder columns to ensure t0, t1, ... t8
    cols = [f"t{i}" for i in range(len(task_domains))]
    existing_cols = [c for c in cols if c in pivot_df.columns]
    pivot_df = pivot_df[existing_cols]
    
    # Sort index by task_domains order if possible
    domain_order = {d: i for i, d in enumerate(task_domains)}
    pivot_df = pivot_df.reindex(sorted(pivot_df.index, key=lambda x: domain_order.get(x, 999)))

    print("\nAccuracy Trend (Average across seeds):")
    print(pivot_df)

    # Save to Excel
    try:
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            pivot_df.to_excel(writer, sheet_name='Accuracy Trend Mapping')
            trend_df.to_excel(writer, sheet_name='Detailed Trend', index=False)
            accuracy_per_run.to_excel(writer, sheet_name='Per Seed Accuracy', index=False)
        print(f"\nTrend results saved to {output_excel}")
    except Exception as e:
        print(f"Error saving to Excel: {e}")

if __name__ == "__main__":
    main()
