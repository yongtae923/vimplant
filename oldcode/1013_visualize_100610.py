"""
100610_newtarget Data Visualization Code
Analyzes and visualizes experimental results for various targets across LH/RH hemispheres.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
import warnings
warnings.filterwarnings('ignore')

# Font settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_data(data_dir="data/output/100610_newtarget_1013"):
    """Load result data from all txt files."""
    print("Loading data...")
    
    data_path = Path(data_dir)
    results = []
    
    for txt_file in data_path.glob("*.txt"):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract information from filename
            filename = txt_file.stem
            parts = filename.split('_')
            
            subject = parts[0]
            hemisphere = parts[1]
            target = parts[-1].replace('targ-', '')  # Get the last part after splitting by '_'
            
            # Parse results
            result = {
                'subject': subject,
                'hemisphere': hemisphere,
                'target': target,
                'filename': filename
            }
            
            # Extract numeric values
            dice_match = re.search(r'Dice coefficient: ([\d.]+)', content)
            hd_match = re.search(r'Hellinger distance: ([\d.]+)', content)
            yield_match = re.search(r'Grid yield: ([\d.]+)', content)
            max_phosphene_match = re.search(r'Max phospheneMap: ([\d.]+)', content)
            contacts_match = re.search(r'Number of contact points: (\d+)', content)
            
            if dice_match:
                result['dice_coefficient'] = float(dice_match.group(1))
            if hd_match:
                result['hellinger_distance'] = float(hd_match.group(1))
            if yield_match:
                result['grid_yield'] = float(yield_match.group(1))
            if max_phosphene_match:
                result['max_phosphene'] = float(max_phosphene_match.group(1))
            if contacts_match:
                result['contact_points'] = int(contacts_match.group(1))
            
            # Extract parameters
            alpha_match = re.search(r'Alpha: ([\d-]+)', content)
            beta_match = re.search(r'Beta: ([\d-]+)', content)
            offset_match = re.search(r'Offset from base: (\d+)', content)
            shank_match = re.search(r'Shank length: (\d+)', content)
            
            if alpha_match:
                result['alpha'] = int(alpha_match.group(1))
            if beta_match:
                result['beta'] = int(beta_match.group(1))
            if offset_match:
                result['offset'] = int(offset_match.group(1))
            if shank_match:
                result['shank_length'] = int(shank_match.group(1))
            
            results.append(result)
            
        except Exception as e:
            print(f"Error loading file {txt_file}: {e}")
    
    df = pd.DataFrame(results)
    print(f"Total {len(results)} results loaded.")
    return df

def calculate_loss_value(df):
    """Calculate loss value based on dice coefficient, yield, and Hellinger distance."""
    # Based on loss_comb = ([(1, 0.1, 1)]) weights for loss terms
    # Weight 1: dice coefficient (higher is better, so use 1 - dice_coefficient)
    # Weight 0.1: grid yield (higher is better, so use 1 - grid_yield)  
    # Weight 1: Hellinger distance (lower is better, so use as is)
    
    # Calculate loss value
    df['loss_value'] = (1 * (1 - df['dice_coefficient']) + 
                       0.1 * (1 - df['grid_yield']) + 
                       1 * df['hellinger_distance'])
    
    return df

def create_performance_comparison_graphs(df, output_dir="data/output/100610_analysis"):
    """Create performance comparison graphs for each target map across LH/RH hemispheres."""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Calculate loss values
    df = calculate_loss_value(df)
    
    # Get unique targets in specific order
    target_order = ['full', 'inner', 'upper', 'lower', 'a', 'b', 'c', 'abc', 'all', 'square', 'triangle']
    targets = [t for t in target_order if t in df['target'].unique()]
    print(f"Found {len(targets)} targets: {targets}")
    
    # Create subplots for each target
    n_targets = len(targets)
    n_cols = 3
    n_rows = (n_targets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    fig.suptitle('Performance Comparison: LH vs RH by Target Map', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()
    
    for i, target in enumerate(targets):
        ax = axes_flat[i]
        
        # Filter data for current target
        target_data = df[df['target'] == target]
        print(f"Target {target}: {len(target_data)} records")
        print(f"LH records: {len(target_data[target_data['hemisphere'] == 'LH'])}")
        print(f"RH records: {len(target_data[target_data['hemisphere'] == 'RH'])}")
        
        # Create grouped bar plot
        x_pos = np.arange(len(['Dice Coef', 'Hellinger Dist', 'Grid Yield', 'Loss Value']))
        width = 0.35
        
        # Get mean values for LH and RH
        lh_data = target_data[target_data['hemisphere'] == 'LH']
        rh_data = target_data[target_data['hemisphere'] == 'RH']
        
        lh_values = [
            lh_data['dice_coefficient'].mean(),
            lh_data['hellinger_distance'].mean(),
            lh_data['grid_yield'].mean(),
            lh_data['loss_value'].mean()
        ]
        
        rh_values = [
            rh_data['dice_coefficient'].mean(),
            rh_data['hellinger_distance'].mean(),
            rh_data['grid_yield'].mean(),
            rh_data['loss_value'].mean()
        ]
        
        # Create bars
        bars1 = ax.bar(x_pos - width/2, lh_values, width, label='LH', alpha=0.8, color='skyblue')
        bars2 = ax.bar(x_pos + width/2, rh_values, width, label='RH', alpha=0.8, color='lightcoral')
        
        # Customize plot
        ax.set_title(f'Target: {target}', fontweight='bold')
        ax.set_xlabel('Performance Metrics')
        ax.set_ylabel('Values')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['Dice Coef', 'Hellinger Dist', 'Grid Yield', 'Loss Value'], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Hide unused subplots
    for i in range(n_targets, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path / 'performance_comparison_by_target.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return df

def create_detailed_comparison_graphs(df, output_dir="data/output/100610_analysis"):
    """Create detailed comparison graphs for each performance metric."""
    
    output_path = Path(output_dir)
    
    # Create subplots for all metrics
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Performance Metrics Comparison: LH vs RH by Target Map', fontsize=16, fontweight='bold')
    
    # Define target order for consistent sorting
    target_order = ['full', 'inner', 'upper', 'lower', 'a', 'b', 'c', 'abc', 'all', 'square', 'triangle']
    
    # 1. Dice Coefficient Comparison
    pivot_dice = df.pivot_table(
        index='target', 
        columns='hemisphere', 
        values='dice_coefficient', 
        aggfunc='mean'
    )
    
    # Reorder targets
    pivot_dice = pivot_dice.reindex([t for t in target_order if t in pivot_dice.index])
    
    pivot_dice.plot(kind='bar', ax=axes[0,0], color=['skyblue', 'lightcoral'], alpha=0.8)
    axes[0,0].set_title('Dice Coefficient Comparison', fontsize=12, fontweight='bold')
    axes[0,0].set_xlabel('Target Map')
    axes[0,0].set_ylabel('Dice Coefficient')
    axes[0,0].legend(title='Hemisphere')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].grid(True, alpha=0.3)
    
    # Add value labels
    for i, (target, row) in enumerate(pivot_dice.iterrows()):
        for j, (hemisphere, value) in enumerate(row.items()):
            if not pd.isna(value):
                axes[0,0].text(i + (j-0.5)*0.4, value + 0.01, f'{value:.3f}', 
                              ha='center', va='bottom', fontsize=8)
    
    # 2. Hellinger Distance Comparison
    pivot_hd = df.pivot_table(
        index='target', 
        columns='hemisphere', 
        values='hellinger_distance', 
        aggfunc='mean'
    )
    
    # Reorder targets
    pivot_hd = pivot_hd.reindex([t for t in target_order if t in pivot_hd.index])
    
    pivot_hd.plot(kind='bar', ax=axes[0,1], color=['skyblue', 'lightcoral'], alpha=0.8)
    axes[0,1].set_title('Hellinger Distance Comparison', fontsize=12, fontweight='bold')
    axes[0,1].set_xlabel('Target Map')
    axes[0,1].set_ylabel('Hellinger Distance')
    axes[0,1].legend(title='Hemisphere')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].grid(True, alpha=0.3)
    
    # Add value labels
    for i, (target, row) in enumerate(pivot_hd.iterrows()):
        for j, (hemisphere, value) in enumerate(row.items()):
            if not pd.isna(value):
                axes[0,1].text(i + (j-0.5)*0.4, value + 0.01, f'{value:.3f}', 
                              ha='center', va='bottom', fontsize=8)
    
    # 3. Grid Yield Comparison
    pivot_yield = df.pivot_table(
        index='target', 
        columns='hemisphere', 
        values='grid_yield', 
        aggfunc='mean'
    )
    
    # Reorder targets
    pivot_yield = pivot_yield.reindex([t for t in target_order if t in pivot_yield.index])
    
    pivot_yield.plot(kind='bar', ax=axes[1,0], color=['skyblue', 'lightcoral'], alpha=0.8)
    axes[1,0].set_title('Grid Yield Comparison', fontsize=12, fontweight='bold')
    axes[1,0].set_xlabel('Target Map')
    axes[1,0].set_ylabel('Grid Yield')
    axes[1,0].legend(title='Hemisphere')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].grid(True, alpha=0.3)
    
    # Add value labels
    for i, (target, row) in enumerate(pivot_yield.iterrows()):
        for j, (hemisphere, value) in enumerate(row.items()):
            if not pd.isna(value):
                axes[1,0].text(i + (j-0.5)*0.4, value + 0.01, f'{value:.3f}', 
                              ha='center', va='bottom', fontsize=8)
    
    # 4. Loss Value Comparison
    pivot_loss = df.pivot_table(
        index='target', 
        columns='hemisphere', 
        values='loss_value', 
        aggfunc='mean'
    )
    
    # Reorder targets
    pivot_loss = pivot_loss.reindex([t for t in target_order if t in pivot_loss.index])
    
    pivot_loss.plot(kind='bar', ax=axes[1,1], color=['skyblue', 'lightcoral'], alpha=0.8)
    axes[1,1].set_title('Loss Value Comparison', fontsize=12, fontweight='bold')
    axes[1,1].set_xlabel('Target Map')
    axes[1,1].set_ylabel('Loss Value')
    axes[1,1].legend(title='Hemisphere')
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].grid(True, alpha=0.3)
    
    # Add value labels
    for i, (target, row) in enumerate(pivot_loss.iterrows()):
        for j, (hemisphere, value) in enumerate(row.items()):
            if not pd.isna(value):
                axes[1,1].text(i + (j-0.5)*0.4, value + 0.01, f'{value:.3f}', 
                              ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path / 'detailed_comparison_all_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_parameter_comparison_graphs(df, output_dir="data/output/100610_analysis"):
    """Create parameter comparison graphs for each target map across LH/RH hemispheres."""
    
    output_path = Path(output_dir)
    
    # Define target order for consistent sorting
    target_order = ['full', 'inner', 'upper', 'lower', 'a', 'b', 'c', 'abc', 'all', 'square', 'triangle']
    
    # Create subplots for all parameters
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Parameter Comparison: LH vs RH by Target Map', fontsize=16, fontweight='bold')
    
    # 1. Alpha Parameter Comparison
    pivot_alpha = df.pivot_table(
        index='target', 
        columns='hemisphere', 
        values='alpha', 
        aggfunc='mean'
    )
    
    # Reorder targets
    pivot_alpha = pivot_alpha.reindex([t for t in target_order if t in pivot_alpha.index])
    
    pivot_alpha.plot(kind='bar', ax=axes[0,0], color=['skyblue', 'lightcoral'], alpha=0.8)
    axes[0,0].set_title('Alpha Parameter Comparison', fontsize=12, fontweight='bold')
    axes[0,0].set_xlabel('Target Map')
    axes[0,0].set_ylabel('Alpha Value')
    axes[0,0].legend(title='Hemisphere')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].grid(True, alpha=0.3)
    
    # Add value labels
    for i, (target, row) in enumerate(pivot_alpha.iterrows()):
        for j, (hemisphere, value) in enumerate(row.items()):
            if not pd.isna(value):
                axes[0,0].text(i + (j-0.5)*0.4, value + 0.5, f'{value:.0f}', 
                              ha='center', va='bottom', fontsize=8)
    
    # 2. Beta Parameter Comparison
    pivot_beta = df.pivot_table(
        index='target', 
        columns='hemisphere', 
        values='beta', 
        aggfunc='mean'
    )
    
    # Reorder targets
    pivot_beta = pivot_beta.reindex([t for t in target_order if t in pivot_beta.index])
    
    pivot_beta.plot(kind='bar', ax=axes[0,1], color=['skyblue', 'lightcoral'], alpha=0.8)
    axes[0,1].set_title('Beta Parameter Comparison', fontsize=12, fontweight='bold')
    axes[0,1].set_xlabel('Target Map')
    axes[0,1].set_ylabel('Beta Value')
    axes[0,1].legend(title='Hemisphere')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].grid(True, alpha=0.3)
    
    # Add value labels
    for i, (target, row) in enumerate(pivot_beta.iterrows()):
        for j, (hemisphere, value) in enumerate(row.items()):
            if not pd.isna(value):
                axes[0,1].text(i + (j-0.5)*0.4, value + 0.5, f'{value:.0f}', 
                              ha='center', va='bottom', fontsize=8)
    
    # 3. Offset Parameter Comparison
    pivot_offset = df.pivot_table(
        index='target', 
        columns='hemisphere', 
        values='offset', 
        aggfunc='mean'
    )
    
    # Reorder targets
    pivot_offset = pivot_offset.reindex([t for t in target_order if t in pivot_offset.index])
    
    pivot_offset.plot(kind='bar', ax=axes[1,0], color=['skyblue', 'lightcoral'], alpha=0.8)
    axes[1,0].set_title('Offset Parameter Comparison', fontsize=12, fontweight='bold')
    axes[1,0].set_xlabel('Target Map')
    axes[1,0].set_ylabel('Offset Value')
    axes[1,0].legend(title='Hemisphere')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].grid(True, alpha=0.3)
    
    # Add value labels
    for i, (target, row) in enumerate(pivot_offset.iterrows()):
        for j, (hemisphere, value) in enumerate(row.items()):
            if not pd.isna(value):
                axes[1,0].text(i + (j-0.5)*0.4, value + 0.5, f'{value:.0f}', 
                              ha='center', va='bottom', fontsize=8)
    
    # 4. Shank Length Parameter Comparison
    pivot_shank = df.pivot_table(
        index='target', 
        columns='hemisphere', 
        values='shank_length', 
        aggfunc='mean'
    )
    
    # Reorder targets
    pivot_shank = pivot_shank.reindex([t for t in target_order if t in pivot_shank.index])
    
    pivot_shank.plot(kind='bar', ax=axes[1,1], color=['skyblue', 'lightcoral'], alpha=0.8)
    axes[1,1].set_title('Shank Length Parameter Comparison', fontsize=12, fontweight='bold')
    axes[1,1].set_xlabel('Target Map')
    axes[1,1].set_ylabel('Shank Length Value')
    axes[1,1].legend(title='Hemisphere')
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].grid(True, alpha=0.3)
    
    # Add value labels
    for i, (target, row) in enumerate(pivot_shank.iterrows()):
        for j, (hemisphere, value) in enumerate(row.items()):
            if not pd.isna(value):
                axes[1,1].text(i + (j-0.5)*0.4, value + 0.5, f'{value:.0f}', 
                              ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path / 'parameter_comparison_all.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_table(df, output_dir="data/output/100610_analysis"):
    """Create and save summary table."""
    output_path = Path(output_dir)
    
    # Create summary table
    summary_table = df.groupby(['target', 'hemisphere']).agg({
        'dice_coefficient': 'mean',
        'hellinger_distance': 'mean', 
        'grid_yield': 'mean',
        'loss_value': 'mean',
        'contact_points': 'mean',
        'alpha': 'mean',
        'beta': 'mean',
        'offset': 'mean',
        'shank_length': 'mean'
    }).round(4)
    
    # Save to CSV
    summary_table.to_csv(output_path / 'summary_table.csv')
    
    print("\n=== Summary Table ===")
    print(summary_table.to_string())
    
    return summary_table

def main():
    """Main execution function."""
    print("Starting 100610_newtarget data visualization...")
    
    # Load data
    df = load_data()
    
    # Create performance comparison graphs
    df = create_performance_comparison_graphs(df)
    
    # Create detailed comparison graphs
    create_detailed_comparison_graphs(df)
    
    # Create parameter comparison graphs
    create_parameter_comparison_graphs(df)
    
    # Create summary table
    summary_table = create_summary_table(df)
    
    print("Visualization completed! All plots saved to data/output/100610_analysis/")
    return df

if __name__ == "__main__":
    df = main()
