import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import defaultdict

def get_min_energy(configurations):
    if not configurations:
        return None
    return min(config["formation_energy"] for config in configurations)

def extract_surface_info(surface_name):
    """Extract metal, dopant, miller index, and promoter information from surface name."""
    info = {
        'base_metal': None,
        'dopant': None,
        'dopant_fraction': None,
        'miller': None,
        'promoter': None,
        'promoter_coverage': None,
        'structure_id': None
    }
    
    # Extract base metal (mp-id)
    mp_match = re.search(r'(mp-\d+)', surface_name)
    if mp_match:
        info['base_metal'] = mp_match.group(1)
    
    # Extract miller indices
    miller_match = re.search(r'\((\d+),\s*(\d+),\s*(\d+)\)', surface_name)
    if miller_match:
        h, k, l = miller_match.groups()
        info['miller'] = f"({h},{k},{l})"
    
    # Extract dopant and fraction
    dopant_match = re.search(r'([A-Z][a-z]?)(\d+\.\d+)', surface_name)
    if dopant_match:
        info['dopant'] = dopant_match.group(1)
        info['dopant_fraction'] = float(dopant_match.group(2))
    
    # Extract structure ID (for alloys)
    struct_id_match = re.search(r'_(\d+)($|_)', surface_name)
    if struct_id_match:
        info['structure_id'] = struct_id_match.group(1)
    
    # Check for promoter info in the data
    promoter_info = None
    
    return info, promoter_info

def group_surfaces_by_type(results):
    """Group surfaces by type (pure, alloy, promoted)."""
    surface_groups = {
        'pure': {},
        'alloy': defaultdict(list),
        'promoted': defaultdict(list)
    }
    
    for surface_name, data in results.items():
        # Check if this is a promoted surface from data
        promoter_info = None
        for rxn_name, rxn_data in data.items():
            if "promoter_info" in rxn_data and rxn_data["promoter_info"]:
                promoter_info = rxn_data["promoter_info"]
                break
        
        info, _ = extract_surface_info(surface_name)
        
        if promoter_info:
            # This is a promoted surface
            base_key = f"{info['base_metal']}_{info['miller']}"
            if info['dopant']:
                base_key += f"_{info['dopant']}{info['dopant_fraction']}"
            
            promoter_key = f"{promoter_info['element']}{promoter_info['coverage']}"
            surface_groups['promoted'][base_key].append({
                'name': surface_name,
                'promoter': promoter_info['element'],
                'coverage': promoter_info['coverage'],
                'data': data
            })
        elif info['dopant']:
            # This is an alloy surface
            base_key = f"{info['base_metal']}_{info['miller']}_{info['dopant']}{info['dopant_fraction']}"
            surface_groups['alloy'][base_key].append({
                'name': surface_name,
                'structure_id': info['structure_id'],
                'data': data
            })
        else:
            # This is a pure surface
            base_key = f"{info['base_metal']}_{info['miller']}"
            surface_groups['pure'][base_key] = {
                'name': surface_name,
                'data': data
            }
    
    return surface_groups

def plot_reaction_network_by_metal(results, output_dir="./plots"):
    """Plot reaction networks grouped by metal and surface type."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    surface_groups = group_surfaces_by_type(results)
    
    # 1. Plot pure surfaces
    for base_key, surface_info in surface_groups['pure'].items():
        surface_name = surface_info['name']
        metal_id, miller = base_key.split('_', 1)
        output_filename = f"{output_dir}/{metal_id}_{miller.replace(',', '')}_pure.png"
        plot_single_surface_reaction(surface_info['data'], metal_id, miller, output_filename)
    
    # 2. Plot alloy surfaces
    for base_key, surfaces in surface_groups['alloy'].items():
        if not surfaces:
            continue
        
        # Group structure IDs for the same alloy type
        parts = base_key.split('_')
        metal_id = parts[0]
        miller = parts[1]
        alloy_info = '_'.join(parts[2:])
        
        output_filename = f"{output_dir}/{metal_id}_{miller.replace(',', '')}_{alloy_info}_alloy_comparison.png"
        plot_alloy_comparison(surfaces, metal_id, miller, alloy_info, output_filename)
    
    # 3. Plot promoted surfaces
    for base_key, surfaces in surface_groups['promoted'].items():
        if not surfaces:
            continue
        
        parts = base_key.split('_')
        metal_id = parts[0]
        miller = parts[1]
        
        alloy_info = ""
        if len(parts) > 2:
            alloy_info = '_'.join(parts[2:])
            base_desc = f"{metal_id} {miller} {alloy_info}"
        else:
            base_desc = f"{metal_id} {miller}"
        
        output_filename = f"{output_dir}/{metal_id}_{miller.replace(',', '')}"
        if alloy_info:
            output_filename += f"_{alloy_info}"
        output_filename += "_promoter_comparison.png"
        
        plot_promoter_comparison(surfaces, base_desc, output_filename)
    
    # 4. Summary comparison plot for each base metal
    for metal_id in set(key.split('_')[0] for key in surface_groups['pure']):
        output_filename = f"{output_dir}/{metal_id}_summary_comparison.png"
        plot_summary_comparison(metal_id, surface_groups, output_filename)

def plot_single_surface_reaction(surface_data, metal_id, miller, output_filename):
    """Plot reaction network for a single surface with improved formatting."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {
        'bare_surface': '#e74c3c',
        'ethylene + O': '#3498db',
        'ethylene oxide metallocycle + O': '#9b59b6', 
        'ethylene epoxide + O': '#2ecc71'
    }
    
    x_positions = {
        'bare_surface': 0,
        'ethylene + O': 1,
        'ethylene oxide metallocycle + O': 2,
        'ethylene epoxide + O': 3
    }
    
    # Create legend elements
    legend_elements = [
        plt.Line2D([0], [0], color=colors[key], lw=3, label=key)
        for key in colors
    ]
    
    energies = {'bare_surface': 0.0}  # Set reference state to 0
    
    # Get minimum energies for each state and check if data exists
    missing_data = []
    for state in colors.keys():
        if state == 'bare_surface':
            continue
        key = f"ethylene epoxidation_{state}"
        if key in surface_data and surface_data[key]["configurations"]:
            min_energy = get_min_energy(surface_data[key]["configurations"])
            if min_energy is not None:
                energies[state] = min_energy
            else:
                missing_data.append(state)
        else:
            missing_data.append(state)
    
    # Add a warning if data is missing
    if missing_data:
        ax.text(0.5, 0.95, f"Missing data for: {', '.join(missing_data)}", 
                ha='center', va='top', transform=ax.transAxes, 
                color='red', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    line_length = 0.3
    
    # Determine y-axis limits based on available data
    y_values = list(energies.values())
    if len(y_values) > 1:  # We have at least one reaction state besides bare_surface
        y_min = min(y_values) - 0.5
        y_max = max(y_values) + 0.5
        # Ensure reasonable range even with similar values
        if y_max - y_min < 1.0:
            y_mean = (y_max + y_min) / 2
            y_min = y_mean - 0.5
            y_max = y_mean + 0.5
    else:
        # Default range if we only have reference state
        y_min = -0.5
        y_max = 0.5
    
    # Plot states and connections
    prev_state = None
    for state in colors.keys():
        if state in energies:
            x = x_positions[state]
            y = energies[state]
            
            # Plot horizontal line for state
            ax.plot([x - line_length/2, x + line_length/2], [y, y], 
                   color=colors[state], linewidth=3)
            
            # Add energy label
            ax.text(x, y - 0.1, f"{y:.2f}", ha='center', va='top', 
                   fontsize=10, color=colors[state], fontweight='bold')
            
            # Connect to previous state
            if prev_state and prev_state in energies:
                prev_x = x_positions[prev_state]
                prev_y = energies[prev_state]
                ax.plot([prev_x + line_length/2, x - line_length/2],
                       [prev_y, y], 'gray', linestyle='--', alpha=0.7, linewidth=2)
            
            prev_state = state
    
    # Customize plot
    miller_str = str(miller).replace(',', '')
    if isinstance(miller, tuple):
        miller_str = f"({miller[0]},{miller[1]},{miller[2]})"
    ax.set_title(f"{metal_id} {miller_str}", fontsize=14, fontweight='bold', pad=10)
    
    # Set axis limits with less wasted space
    ax.set_xlim(-0.5, max(x_positions.values()) + 0.5)
    ax.set_ylim(y_min, y_max)
    
    # Add state labels on x-axis
    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels([state.split(' + ')[0] if ' + ' in state else state 
                       for state in x_positions.keys()], fontsize=10, rotation=30, ha='right')
    
    ax.set_ylabel("Formation Energy (eV)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add legend with better placement
    ax.legend(handles=legend_elements, loc='best', fontsize=10, framealpha=0.9)
    
    # Add reaction arrows and labels for clarity
    arrow_props = dict(arrowstyle='->', color='gray', lw=1.5, alpha=0.7)
    for i in range(len(list(x_positions.keys()))-1):
        state1 = list(x_positions.keys())[i]
        state2 = list(x_positions.keys())[i+1]
        x1 = x_positions[state1]
        x2 = x_positions[state2]
        ax.annotate('', xy=(x2-0.3, y_max-0.2), xytext=(x1+0.3, y_max-0.2), 
                    arrowprops=arrow_props)
    
    # Improve overall appearance
    fig.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_alloy_comparison(surfaces, metal_id, miller, alloy_info, output_filename):
    """Plot comparison of different alloy structures for the same composition."""
    fig = plt.figure(figsize=(12, 8))
    
    n_surfaces = len(surfaces)
    n_cols = min(3, n_surfaces)
    n_rows = (n_surfaces + n_cols - 1) // n_cols
    
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.4, wspace=0.3)
    axs = gs.subplots()
    
    if n_surfaces == 1:
        axs = np.array([axs])
    
    axs = axs.flatten()
    
    colors = {
        'bare_surface': '#e74c3c',
        'ethylene + O': '#3498db',
        'ethylene oxide metallocycle + O': '#9b59b6', 
        'ethylene epoxide + O': '#2ecc71'
    }
    
    x_positions = {
        'bare_surface': 0,
        'ethylene + O': 1,
        'ethylene oxide metallocycle + O': 2,
        'ethylene epoxide + O': 3
    }
    
    # Create legend elements
    legend_elements = [
        plt.Line2D([0], [0], color=colors[key], label=key, lw=2)
        for key in colors
    ]
    
    # Find global min/max energies for consistent y-axis
    global_min = float('inf')
    global_max = float('-inf')
    
    for i, surface_info in enumerate(surfaces):
        surface_data = surface_info['data']
        energies = {'bare_surface': 0.0}
        
        for state in colors.keys():
            if state == 'bare_surface':
                continue
            key = f"ethylene epoxidation_{state}"
            if key in surface_data:
                min_energy = get_min_energy(surface_data[key]["configurations"])
                if min_energy is not None:
                    energies[state] = min_energy
                    global_min = min(global_min, min_energy)
                    global_max = max(global_max, min_energy)
    
    # Calculate y-axis limits
    y_range = global_max - global_min
    padding = max(0.5, 0.25 * y_range)
    y_min = global_min - padding
    y_max = 0.5#global_max + padding
    
    # Plot each surface
    for i, surface_info in enumerate(surfaces):
        surface_data = surface_info['data']
        structure_id = surface_info.get('structure_id', 'unknown')
        
        ax = axs[i]
        energies = {'bare_surface': 0.0}
        
        for state in colors.keys():
            if state == 'bare_surface':
                continue
            key = f"ethylene epoxidation_{state}"
            if key in surface_data:
                min_energy = get_min_energy(surface_data[key]["configurations"])
                if min_energy is not None:
                    energies[state] = min_energy
        
        line_length = 0.3
        
        # Plot states and connections
        prev_state = None
        for state in colors.keys():
            if state in energies:
                x = x_positions[state]
                y = energies[state]
                
                # Plot horizontal line for state
                ax.plot([x - line_length/2, x + line_length/2], [y, y], 
                       color=colors[state], linewidth=2)
                
                # Add energy label
                ax.text(x, y - 0.1, f"{y:.2f}", ha='center', va='top', 
                       fontsize=8, color=colors[state])
                
                # Connect to previous state
                if prev_state and prev_state in energies:
                    prev_x = x_positions[prev_state]
                    prev_y = energies[prev_state]
                    ax.plot([prev_x + line_length/2, x - line_length/2],
                           [prev_y, y], 'gray', linestyle='--', alpha=0.5)
                
                prev_state = state
        
        # Customize plot
        miller_str = miller.replace(',', '')
        ax.set_title(f"Structure {structure_id}", pad=10)
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(y_min, y_max)
        
        ax.set_xlabel("Reaction Coordinate")
        ax.set_ylabel("Formation Energy (eV)")
        ax.grid(True, linestyle='--', alpha=0.3)
    
    # Hide any unused subplots
    for i in range(len(surfaces), len(axs)):
        axs[i].set_visible(False)
    
    # Add legend to figure
    fig.legend(handles=legend_elements, loc='center right', 
              bbox_to_anchor=(0.98, 0.98), ncol=1)
    
    plt.suptitle(f"{metal_id} {miller.replace(',', '')} with {alloy_info}", y=1.02)
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_promoter_comparison(surfaces, base_desc, output_filename):
    """Plot comparison of surfaces with different promoters."""
    fig = plt.figure(figsize=(12, 8))
    
    n_surfaces = len(surfaces)
    n_cols = min(3, n_surfaces)
    n_rows = (n_surfaces + n_cols - 1) // n_cols
    
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.4, wspace=0.3)
    axs = gs.subplots()
    
    if n_surfaces == 1:
        axs = np.array([axs])
    
    axs = axs.flatten()
    
    colors = {
        'bare_surface': '#e74c3c',
        'ethylene + O': '#3498db',
        'ethylene oxide metallocycle + O': '#9b59b6', 
        'ethylene epoxide + O': '#2ecc71'
    }
    
    x_positions = {
        'bare_surface': 0,
        'ethylene + O': 1,
        'ethylene oxide metallocycle + O': 2,
        'ethylene epoxide + O': 3
    }
    
    # Create legend elements
    legend_elements = [
        plt.Line2D([0], [0], color=colors[key], label=key, lw=2)
        for key in colors
    ]
    
    # Find global min/max energies for consistent y-axis
    global_min = float('inf')
    global_max = float('-inf')
    
    for i, surface_info in enumerate(surfaces):
        surface_data = surface_info['data']
        energies = {'bare_surface': 0.0}
        
        for state in colors.keys():
            if state == 'bare_surface':
                continue
            key = f"ethylene epoxidation_{state}"
            if key in surface_data:
                min_energy = get_min_energy(surface_data[key]["configurations"])
                if min_energy is not None:
                    energies[state] = min_energy
                    global_min = min(global_min, min_energy)
                    global_max = max(global_max, min_energy)
    
    # Calculate y-axis limits
    y_range = global_max - global_min
    padding = max(0.5, 0.25 * y_range)
    y_min = global_min - padding
    y_max = global_max + padding
    
    # Plot each surface
    for i, surface_info in enumerate(surfaces):
        surface_data = surface_info['data']
        promoter = surface_info['promoter']
        coverage = surface_info['coverage']
        
        ax = axs[i]
        energies = {'bare_surface': 0.0}
        
        for state in colors.keys():
            if state == 'bare_surface':
                continue
            key = f"ethylene epoxidation_{state}"
            if key in surface_data:
                min_energy = get_min_energy(surface_data[key]["configurations"])
                if min_energy is not None:
                    energies[state] = min_energy
        
        line_length = 0.3
        
        # Plot states and connections
        prev_state = None
        for state in colors.keys():
            if state in energies:
                x = x_positions[state]
                y = energies[state]
                
                # Plot horizontal line for state
                ax.plot([x - line_length/2, x + line_length/2], [y, y], 
                       color=colors[state], linewidth=2)
                
                # Add energy label
                ax.text(x, y - 0.1, f"{y:.2f}", ha='center', va='top', 
                       fontsize=8, color=colors[state])
                
                # Connect to previous state
                if prev_state and prev_state in energies:
                    prev_x = x_positions[prev_state]
                    prev_y = energies[prev_state]
                    ax.plot([prev_x + line_length/2, x - line_length/2],
                           [prev_y, y], 'gray', linestyle='--', alpha=0.5)
                
                prev_state = state
        
        # Customize plot
        ax.set_title(f"{promoter} ({coverage:.3f})", pad=10)
        ax.set_xlim(-1.5, 2.5)
        ax.set_ylim(y_min, y_max)
        
        ax.set_xlabel("Reaction Coordinate")
        ax.set_ylabel("Formation Energy (eV)")
        ax.grid(True, linestyle='--', alpha=0.3)
    
    # Hide any unused subplots
    for i in range(len(surfaces), len(axs)):
        axs[i].set_visible(False)
    
    # Add legend to figure
    fig.legend(handles=legend_elements, loc='center right', 
              bbox_to_anchor=(0.98, 0.5), ncol=1)
    
    plt.suptitle(f"{base_desc} with Promoters", y=1.02)
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_summary_comparison(metal_id, surface_groups, output_filename):
    """Plot a summary comparison of best pure, alloy, and promoted surfaces."""
    # Collect surfaces as before...
    pure_surfaces = []
    for base_key, surface_info in surface_groups['pure'].items():
        if base_key.startswith(f"{metal_id}_"):
            surface_data = surface_info['data']
            epoxide_key = "ethylene epoxidation_ethylene epoxide + O"
            if epoxide_key in surface_data:
                min_energy = get_min_energy(surface_data[epoxide_key]["configurations"])
                if min_energy is not None:
                    pure_surfaces.append({
                        'label': f"Pure {base_key.split('_')[1]}",
                        'data': surface_data,
                        'epoxide_energy': min_energy
                    })
    
    # Similarly collect alloy and promoted surfaces...
    alloy_surfaces = []
    promoted_surfaces = []
    # ... (include your existing collection code)
    
    # Now plot with improved formatting
    all_surfaces = pure_surfaces + alloy_surfaces + promoted_surfaces
    all_surfaces.sort(key=lambda x: x.get('epoxide_energy', float('inf')))
    
    if not all_surfaces:
        logging.warning(f"No data to plot for {metal_id}")
        return
    
    # Take top 6 surfaces
    top_surfaces = all_surfaces[:min(6, len(all_surfaces))]
    
    fig = plt.figure(figsize=(18, 10))
    
    n_surfaces = len(top_surfaces)
    n_cols = min(3, n_surfaces)
    n_rows = (n_surfaces + n_cols - 1) // n_cols
    
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.4, wspace=0.3)
    axs = gs.subplots()
    
    if n_surfaces == 1:
        axs = np.array([axs])
    
    axs = axs.flatten()
    
    colors = {
        'bare_surface': '#e74c3c',
        'ethylene + O': '#3498db',
        'ethylene oxide metallocycle + O': '#9b59b6', 
        'ethylene epoxide + O': '#2ecc71'
    }
    
    x_positions = {
        'bare_surface': 0,
        'ethylene + O': 1,
        'ethylene oxide metallocycle + O': 2,
        'ethylene epoxide + O': 3
    }
    
    # Create legend elements
    legend_elements = [
        plt.Line2D([0], [0], color=colors[key], label=key, lw=2)
        for key in colors
    ]
    
    # Find global min/max for all data with padding
    global_min = float('inf')
    global_max = float('-inf')
    
    for surf_idx, surface_info in enumerate(top_surfaces):
        surface_data = surface_info['data']
        energies = {'bare_surface': 0.0}
        
        for state in colors.keys():
            if state == 'bare_surface':
                continue
            key = f"ethylene epoxidation_{state}"
            if key in surface_data:
                min_energy = get_min_energy(surface_data[key]["configurations"])
                if min_energy is not None:
                    energies[state] = min_energy
                    global_min = min(global_min, min_energy)
                    global_max = max(global_max, min_energy)
    
    # Set reasonable y-range with padding
    if global_min != float('inf') and global_max != float('-inf'):
        range_pad = max(0.5, 0.2 * (global_max - global_min))
        y_min = global_min - range_pad
        y_max = global_max + range_pad
    else:
        # Default if we have no data
        y_min, y_max = -1, 1
    
    # Plot each surface with improved formatting
    for i, surface_info in enumerate(top_surfaces):
        surface_data = surface_info['data']
        label = surface_info['label']
        
        ax = axs[i]
        energies = {'bare_surface': 0.0}
        
        # Collect data for each reaction state
        missing_data = []
        for state in colors.keys():
            if state == 'bare_surface':
                continue
            key = f"ethylene epoxidation_{state}"
            if key in surface_data and surface_data[key]["configurations"]:
                min_energy = get_min_energy(surface_data[key]["configurations"])
                if min_energy is not None:
                    energies[state] = min_energy
                else:
                    missing_data.append(state)
            else:
                missing_data.append(state)
        
        # Add a warning if data is missing
        if missing_data:
            ax.text(0.5, 0.95, f"Missing: {', '.join(missing_data)}", 
                   ha='center', va='top', transform=ax.transAxes, 
                   color='red', fontsize=8, bbox=dict(facecolor='white', alpha=0.8))
        
        line_length = 0.3
        
        # Plot reaction pathway
        prev_state = None
        for state in colors.keys():
            if state in energies:
                x = x_positions[state]
                y = energies[state]
                
                # Plot horizontal line for state
                ax.plot([x - line_length/2, x + line_length/2], [y, y], 
                       color=colors[state], linewidth=3)
                
                # Add energy label
                ax.text(x, y - 0.1, f"{y:.2f}", ha='center', va='top', 
                       fontsize=9, color=colors[state], fontweight='bold')
                
                # Connect to previous state
                if prev_state and prev_state in energies:
                    prev_x = x_positions[prev_state]
                    prev_y = energies[prev_state]
                    ax.plot([prev_x + line_length/2, x - line_length/2],
                           [prev_y, y], 'gray', linestyle='--', alpha=0.7, linewidth=1.5)
                
                prev_state = state
        
        # Customize each subplot
        ax.set_title(label, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlim(-0.5, max(x_positions.values()) + 0.5)
        ax.set_ylim(y_min, y_max)
        
        # Add state labels on x-axis
        ax.set_xticks(list(x_positions.values()))
        short_labels = [state.split(' + ')[0] if ' + ' in state else state 
                      for state in x_positions.keys()]
        ax.set_xticklabels(short_labels, fontsize=8, rotation=30, ha='right')
        
        ax.set_ylabel("Formation Energy (eV)" if i % n_cols == 0 else "", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(top_surfaces), len(axs)):
        axs[i].set_visible(False)
    
    # Add legend and title with better formatting
    fig.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.5, 0.06), ncol=len(colors), fontsize=11, framealpha=0.9)
    
    plt.suptitle(f"Top Catalysts for Ethylene Epoxidation - {metal_id}", 
                y=0.98, fontsize=16, fontweight='bold')
    
    # Add metadata about criteria
    fig.text(0.5, 0.94, 
             f"Surfaces ranked by epoxide formation energy. Lower values indicate better catalytic performance.", 
             ha='center', fontsize=10, fontstyle='italic')
    
    # Improve overall appearance
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    with open("EO_mono_results.json", 'r') as f:
        results = json.load(f)
    
    plot_reaction_network_by_metal(results, output_dir="./plots")

if __name__ == "__main__":
    main()
