import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def parse_log(file_path):
    tasks = {}
    current_forget_idx = None
    current_epoch = None
    
    # Data storage: tasks[forget_idx] = {'epochs': [], 'gate_ce': [], 'expert_mse': []}
    
    with open(file_path, 'r') as f:
        for line in f:
            # Check for new task
            task_match = re.search(r'\[MoE Train\] current_forget_idx=(\d+)', line)
            if task_match:
                current_forget_idx = int(task_match.group(1))
                if current_forget_idx not in tasks:
                    tasks[current_forget_idx] = {'gate_ce': [], 'expert_mse': [], 'steps': []}
                continue
            
            # Check for epoch (to track progress or label)
            epoch_match = re.search(r'Epoch \[(\d+)/\d+\]', line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                # Note: We don't necessarily need to store epoch per point if we use steps,
                # but we'll use it for fractional x-axis if needed.
            
            # Extract Gate Norms
            # [DEBUG Grad] Gate   Norm from MSE: 0.000000 | Gate   Norm from CE: 0.122591
            gate_match = re.search(r'Gate\s+Norm from MSE: [\d\.]+\s+\|\s+Gate\s+Norm from CE: ([\d\.]+)', line)
            if gate_match and current_forget_idx is not None:
                tasks[current_forget_idx]['gate_ce'].append(float(gate_match.group(1)))
                tasks[current_forget_idx]['steps'].append(current_epoch if current_epoch else 0)
                continue
            
            # Extract Expert Norms
            # [DEBUG Grad] Expert Norm from MSE: 0.001305 | Expert Norm from CE: 0.000000
            expert_match = re.search(r'Expert Norm from MSE: ([\d\.]+)\s+\|\s+Expert Norm from CE: [\d\.]+', line)
            if expert_match and current_forget_idx is not None:
                # We assume Expert Norm line follows Gate Norm line or is close.
                # If they are on separate lines, we need to make sure they align.
                # In the log provided, they are consecutive.
                # Let's check if the last added point needs this value.
                if len(tasks[current_forget_idx]['expert_mse']) < len(tasks[current_forget_idx]['gate_ce']):
                    tasks[current_forget_idx]['expert_mse'].append(float(expert_match.group(1)))
    
    # Cleanup tasks that didn't get balanced data
    for idx in list(tasks.keys()):
        min_len = min(len(tasks[idx]['gate_ce']), len(tasks[idx]['expert_mse']), len(tasks[idx]['steps']))
        tasks[idx]['gate_ce'] = tasks[idx]['gate_ce'][:min_len]
        tasks[idx]['expert_mse'] = tasks[idx]['expert_mse'][:min_len]
        tasks[idx]['steps'] = tasks[idx]['steps'][:min_len]
        if min_len == 0:
            del tasks[idx]
            
    return tasks

def plot_tasks(tasks, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for idx, data in tasks.items():
        plt.figure(figsize=(10, 6))
        
        # Plot Gate Norm from CE
        plt.plot(data['gate_ce'], label='Gate Norm from CE', color='blue', marker='o', markersize=3)
        # Plot Expert Norm from MSE
        plt.plot(data['expert_mse'], label='Expert Norm from MSE', color='red', marker='x', markersize=3)
        
        plt.title(f'Gradient Norms - Forget Index {idx}')
        plt.xlabel('Logged Step (points during epochs)')
        plt.ylabel('Norm Value')
        plt.yscale('log') # Useful since norms can vary in magnitude
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        
        save_path = os.path.join(output_dir, f'grad_norms_task_{idx}.png')
        plt.savefig(save_path)
        plt.close()
        print(f'Saved plot to {save_path}')

if __name__ == "__main__":
    log_file = "/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/modify_code/run_logs/test_clinc_unverified.log"
    output_directory = "/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/modify_code/run_logs/plots"
    
    grad_data = parse_log(log_file)
    if not grad_data:
        print("No data found in log.")
    else:
        plot_tasks(grad_data, output_directory)

