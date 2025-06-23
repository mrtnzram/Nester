# Core libraries
import os
from pathlib import Path

# Data handling
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_hex
from matplotlib import colormaps

# Widgets for UI
import ipywidgets as widgets
from IPython.display import display

# Clustering & embedding
import umap
import hdbscan

def Nester(syllable_df,output_dir):
        # --- Internal state ---
        edited_labels = {}
        active_df = None
        current_axes = None
        edit_history = []

        # --- Ensure new column exists ---
        if 'hdbscan_syntax_label' not in syllable_df.columns:
            syllable_df['hdbscan_syntax_label'] = None

        if 'final_syntax' not in syllable_df.columns:
            syllable_df['final_syntax'] = np.nan
            
        # --- Save Edits Back to syllable_df ---
        def save_edits_to_syllable_df():
            global active_df, bird_dropdown, syllable_df

            if active_df is None or active_df.empty:
                print("❌ No data to save: active_df is None or empty.")
                return
            
            if 'final_syntax' not in active_df.columns:
                print("❌ 'final_syntax' column missing in active_df.")
                return

            if 'final_syntax' not in syllable_df.columns:
                syllable_df['final_syntax'] = np.nan

            # Get current bird_id from dropdown widget
            current_bird = bird_dropdown.value
            if current_bird is None:
                print("❌ No bird selected.")
                return

            # Filter syllable_df to rows of current bird
            bird_rows = syllable_df['bird_id'] == current_bird

            # Valid indices where both active_df and syllable_df overlap for current bird only
            valid_index = active_df.index.intersection(syllable_df[bird_rows].index)

            if valid_index.empty:
                print("❌ No overlapping indices for current bird to save.")
                return

            # Save edits only for the current bird rows
            syllable_df.loc[valid_index, 'final_syntax'] = active_df.loc[valid_index, 'final_syntax'].values

            print(f"✅ Saved {len(valid_index)} editable labels into syllable_df['final_syntax'] for bird '{current_bird}'.")
                
        # --- UI Controls ---
        bird_dropdown = widgets.Dropdown(
            options=sorted(syllable_df['bird_id'].unique()),
            description='Bird:'
        )
        key_dropdown = widgets.Dropdown(description='Bout:')
        umap_slider = widgets.FloatText(
            description='UMAP distance', value=0.25,
            step = 0.01,
            layout=widgets.Layout(width="250px"),
            style={'description_width': 'initial'}
        )
        mcs_slider = widgets.IntText(description='Cluster Size', value=5)
        direction_toggle = widgets.ToggleButtons(
            options=[('↑ Increment', 1), ('↓ Decrement', -1)],
            description='Click mode:',
            style={'description_width': 'initial'}
        )
        plot_all_bouts_button = widgets.Button(
            description="Show All Bouts for Bird",
            layout=widgets.Layout(width="300px")
        )
        save_figures_button = widgets.Button(
            description="Save All Figures",
            layout=widgets.Layout(width="300px")
        )
        plot_output = widgets.Output()

        # --- Embedding & Clustering Memory ---
        bird_embeddings = {}  # bird_id → {embedded, labels, params}
        per_bird_params = {}

        # --- Making UMAP & HDBSCAN columns ---
        if 'mcs_used' not in syllable_df.columns:
            syllable_df['hdbscan_mcs'] = 5
        if 'umap_min_dist' not in syllable_df.columns:
            syllable_df['umap_min_dist'] = 0.25
        seed = 1234

        # --- Initializing UMAP & HDBSCAN columns ---
        # Ensure the syntax label column exists
        syllable_df['hdbscan_syntax_label'] = np.nan

        for bird_id in syllable_df['bird_id'].unique():
            bird_df = syllable_df[syllable_df['bird_id'] == bird_id].copy()

            if bird_df.empty or 'spectrogram' not in bird_df.columns:
                print(f"⚠️ Skipping '{bird_id}' — no usable data.")
                continue

            try:
                mcs = int(bird_df['hdbscan_mcs'].iloc[0])
                min_dist = float(bird_df['umap_min_dist'].iloc[0])
            except Exception as e:
                print(f"❌ Param extraction failed for '{bird_id}': {e}")
                continue

            specs = np.array([s / np.max(s) if np.max(s) > 0 else s for s in bird_df['spectrogram'].values])
            specs[specs < 0] = 0
            flat = specs.reshape(specs.shape[0], -1)

            try:
                embedded = umap.UMAP(min_dist=min_dist, random_state=seed).fit_transform(flat)
                labels = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=1).fit_predict(embedded)
                syllable_df.loc[bird_df.index, 'hdbscan_syntax_label'] = labels
            except Exception as e:
                print(f"❌ Clustering failed for '{bird_id}': {e}")

        # --- Utility ---
        def generate_label_colors(unique_labels, cmap_name="tab20"):
            cmap = colormaps.get_cmap(cmap_name).resampled(len(unique_labels))
            return {label: to_hex(cmap(i)) for i, label in enumerate(unique_labels)}

        def save_current_params(bird_id):
            if bird_id:
                per_bird_params[bird_id] = {
                    'min_dist': umap_slider.value,
                    'mcs': mcs_slider.value
                }

        import os

        def save_all_figures_callback(_):
            bird_id = bird_dropdown.value
            if bird_id is None:
                print("No bird selected.")
                return

            # Get bird data
            bird_df = syllable_df[syllable_df['bird_id'] == bird_id].copy()
            if bird_df.empty:
                print(f"No data for bird '{bird_id}'.")
                return

            species = bird_df['species'].iloc[0]
            min_dist = umap_slider.value
            mcs = mcs_slider.value

            # Ensure global clustering exists
            if bird_id not in bird_embeddings:
                print("Clustering not available. Run 'Show All Bouts' first.")
                return

            labels = bird_embeddings[bird_id]['labels']
            bird_df['final_syntax'] = labels
            for i, val in edited_labels.items():
                if i in bird_df.index:
                    bird_df.loc[i, 'final_syntax'] = val

            # Create folder
            os.makedirs(output_dir / 'corrected', exist_ok=True)
            os.makedirs(output_dir / 'corrected'/'by_bird', exist_ok=True)

            # --- Save full bird view ---
            bout_keys = sorted(bird_df['key'].unique())
            fig, axes = plt.subplots(nrows=len(bout_keys), figsize=(12, 3.5 * len(bout_keys)), sharex=False)

            if len(bout_keys) == 1:
                axes = [axes]

            for ax, key in zip(axes, bout_keys):
                bout_df = bird_df[bird_df['key'] == key]
                specs = np.array([s / np.max(s) if np.max(s) > 0 else s for s in bout_df['spectrogram'].values])
                specs[specs < 0] = 0
                label_colors = generate_label_colors(np.unique(bout_df['final_syntax']))

                for i, spec in enumerate(specs):
                    label = bout_df.iloc[i]['final_syntax']
                    color = label_colors.get(label, '#aaa')
                    ax.imshow(spec, aspect='auto', extent=[i, i+1, 0, 1], cmap='magma')
                    ax.add_patch(mpatches.Rectangle((i, -0.05), 1, 0.1,
                                                    facecolor=color, edgecolor='black', alpha=0.7, transform=ax.transData))
                    ax.text(i + 0.5, -0.1, str(label), ha='center', va='top', fontsize=8)

                ax.axis('off')
                ax.set_xlim(0, len(specs))
                ax.set_ylim(-0.3, 1.1)
                ax.set_title(f"{species} {bird_id} | {key} | UMAP: {min_dist}, HDBSCAN: {mcs}")

            fig.tight_layout()
            fig.savefig(output_dir / 'corrected' / 'by_bird' / f"{species}_{bird_id}_ALL_BOUTS.png", dpi=150)
            plt.close(fig)

            # --- Save individual bout views ---
            for key in bout_keys:
                bout_df = bird_df[bird_df['key'] == key]
                specs = np.array([s / np.max(s) if np.max(s) > 0 else s for s in bout_df['spectrogram'].values])
                specs[specs < 0] = 0
                label_colors = generate_label_colors(np.unique(bout_df['final_syntax']))

                fig, ax = plt.subplots(figsize=(12, 2))
                for i, spec in enumerate(specs):
                    label = bout_df.iloc[i]['final_syntax']
                    color = label_colors.get(label, '#aaa')
                    ax.imshow(spec, aspect='auto', extent=[i, i+1, 0, 1], cmap='magma')
                    ax.add_patch(mpatches.Rectangle((i, -0.05), 1, 0.1,
                                                    facecolor=color, edgecolor='black', alpha=0.7, transform=ax.transData))
                    ax.text(i + 0.5, -0.1, str(label), ha='center', va='top', fontsize=8)
                ax.axis('off')
                ax.set_xlim(0, len(specs))
                ax.set_ylim(-0.3, 1.1)
                ax.set_title(f"{species} {bird_id} | {key} | UMAP: {min_dist}, HDBSCAN: {mcs}")
                fig.tight_layout()
                fig.savefig(output_dir/ 'corrected' / f"{species}_{bird_id}_{key}.png", dpi=150)
                plt.close(fig)

            print(f"All figures saved for bird {bird_id} in output_figures/{bird_id}_{key}/")

        # --- Main Plot Logic ---
        def plot_all_bouts_for_bird():
            bird_id = bird_dropdown.value
            if bird_id is None:
                return

            bird_df = syllable_df[syllable_df['bird_id'] == bird_id].copy()
            if bird_df.empty:
                with plot_output:
                    plot_output.clear_output(wait=True)
                    print(f"No data for bird '{bird_id}'.")
                return

            # --- UMAP + HDBSCAN for full bird ---
            specs = np.array([s / np.max(s) if np.max(s) > 0 else s for s in bird_df['spectrogram'].values])
            specs[specs < 0] = 0
            flat = specs.reshape(specs.shape[0], -1)

            min_dist = umap_slider.value
            mcs = mcs_slider.value
            params = {'min_dist': min_dist, 'mcs': mcs}

            cache = bird_embeddings.get(bird_id)
            if cache and cache['params'] == params:
                labels = cache['labels']
            else:
                embedded = umap.UMAP(min_dist=min_dist, random_state=seed).fit_transform(flat)
                labels = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=1).fit_predict(embedded)
                bird_embeddings[bird_id] = {'labels': labels, 'params': params}

            bird_df['final_syntax'] = labels
            for i, val in edited_labels.items():
                if i in bird_df.index:
                    bird_df.loc[i, 'final_syntax'] = val

            syllable_df.loc[bird_df.index, 'hdbscan_mcs'] = mcs
            syllable_df.loc[bird_df.index, 'umap_min_dist'] = min_dist

            # --- Plot each bout row ---
            bout_keys = sorted(bird_df['key'].unique())
            if len(bout_keys) > 10:
                with plot_output:
                    plot_output.clear_output(wait=True)
                    print(f"⚠️ Bird '{bird_id}' has {len(bout_keys)} bouts — showing individual view instead.")
                update_plot()
                return bird_df.index, labels

            with plot_output:
                plot_output.clear_output(wait=True)
                plt.close('all')
                fig, axes = plt.subplots(nrows=len(bout_keys), figsize=(12, 3.5 * len(bout_keys)), sharex=False)
                if len(bout_keys) == 1:
                    axes = [axes]

                for ax, key in zip(axes, bout_keys):
                    bout_df = bird_df[bird_df['key'] == key].copy()
                    specs = np.array([s / np.max(s) if np.max(s) > 0 else s for s in bout_df['spectrogram'].values])
                    specs[specs < 0] = 0
                    label_colors = generate_label_colors(np.unique(bout_df['final_syntax']))
                    for i, spec in enumerate(specs):
                        label = bout_df.iloc[i]['final_syntax']
                        color = label_colors.get(label, '#aaa')
                        ax.imshow(spec, aspect='auto', extent=[i, i + 1, 0, 1], cmap='magma')
                        ax.add_patch(mpatches.Rectangle((i, -0.05), 1, 0.1, facecolor=color,
                                                        edgecolor='black', alpha=0.7, transform=ax.transData))
                        ax.text(i + 0.5, -0.1, str(label), ha='center', va='top', fontsize=8, color='black')
                    ax.set_xlim(0, len(specs))
                    ax.set_ylim(-0.3, 1.1)
                    ax.axis('off')
                    species = bout_df['species'].iloc[0]
                    ax.set_title(f"{species} {bird_id} | {key} | UMAP: {min_dist}, HDBSCAN: {mcs}")
                plt.tight_layout()
                plt.show()

            label_counts = bird_df['final_syntax'].value_counts().to_dict()
            with plot_output:
                print(f"\nLabels for {species} {bird_id}:")
                for label, count in sorted(label_counts.items()):
                    print(f" {label}: {count}")
                print(f"\nTotal syllables: {len(bird_df)}")

            return bird_df.index, labels


        # --- When calling the function ---
        results = plot_all_bouts_for_bird()
        if results:
            indices, labels = results
            syllable_df.loc[indices, 'final_syntax'] = labels

        # --- Bout Zoom View ---
        def update_plot(change=None):
            global current_axes, active_df
            bird_id = bird_dropdown.value
            key = key_dropdown.value
            if key is None:
                return

            bird_df = syllable_df[syllable_df['bird_id'] == bird_id].copy()
            if bird_id not in bird_embeddings:
                return

            labels = bird_embeddings[bird_id]['labels']
            bird_df['final_syntax'] = labels
            for i, val in edited_labels.items():
                if i in bird_df.index:
                    bird_df.loc[i, 'final_syntax'] = val

            bird_df = bird_df[bird_df['key'] == key]
            if bird_df.empty:
                return

            specs = np.array([s / np.max(s) if np.max(s) > 0 else s for s in bird_df['spectrogram'].values])
            specs[specs < 0] = 0

            label_colors = generate_label_colors(np.unique(bird_df['final_syntax']))
            species = bird_df['species'].iloc[0]
            mcs = mcs_slider.value
            min_dist = umap_slider.value

            with plot_output:
                plot_output.clear_output(wait=True)
                plt.close('all')
                fig, ax = plt.subplots(figsize=(12, 2))
                fig.canvas.manager.set_window_title(f"{species}_{bird_id}_{key}")
                current_axes = ax

                for i, spec in enumerate(specs):
                    label = bird_df.iloc[i]['final_syntax']
                    color = label_colors.get(label, '#aaa')
                    ax.imshow(spec, aspect='auto', extent=[i, i + 1, 0, 1], cmap='magma')
                    ax.add_patch(mpatches.Rectangle((i, -0.05), 1, 0.1, facecolor=color,
                                                    edgecolor='black', alpha=0.7, transform=ax.transData))
                    ax.text(i + 0.5, -0.1, str(label), ha='center', va='top', fontsize=8, color='black')

                ax.set_xlim(0, len(specs))
                ax.set_ylim(-0.1, 1.05)
                ax.axis('off')
                ax.set_title(f"{species} | Bird {bird_id} | {key} | UMAP: {min_dist}, HDBSCAN: {mcs}")
                fig.canvas.mpl_connect("button_press_event", on_click)
                plt.show()

            # --- Summary printout ---
            label_counts = bird_df['final_syntax'].value_counts().to_dict()
            with plot_output:
                print(f"\nLabels for {species} {bird_id} {key}: {len(bird_df)} syllables")
                for label, count in sorted(label_counts.items()):
                    print(f"{label}: {count}")
            

            global active_df
            active_df = bird_df

        # --- Click Editing ---
        def on_click(event):
            global active_df
            if current_axes is None or event.inaxes != current_axes or active_df is None:
                return

            i = int(event.xdata)
            if 0 <= i < len(active_df):
                actual_idx = active_df.index[i]
                old = active_df.loc[actual_idx, 'final_syntax']
                step = direction_toggle.value
                new = old + step if old + step >= 0 else 0

                edit_history.append((actual_idx, old))
                active_df.loc[actual_idx, 'final_syntax'] = new
                edited_labels[actual_idx] = new
                update_plot()

        save_labels_button = widgets.Button(
            description="Save Labels to DataFrame",
            layout=widgets.Layout(width="300px")
        )

        # --- Observer Logic ---
        def update_keys_for_bird(change):
            old_bird = getattr(update_keys_for_bird, 'last_bird', None)
            bird_id = change['new']
            update_keys_for_bird.last_bird = bird_id

            save_current_params(old_bird)

            if bird_id in per_bird_params:
                umap_slider.unobserve_all()
                mcs_slider.unobserve_all()
                umap_slider.value = per_bird_params[bird_id]['min_dist']
                mcs_slider.value = per_bird_params[bird_id]['mcs']
                rebind_slider_observers()

            keys = syllable_df[syllable_df['bird_id'] == bird_id]['key'].unique()
            key_dropdown.options = sorted(keys)
            key_dropdown.value = None

            plot_all_bouts_for_bird()

        # --- Hook everything up ---
        bird_dropdown.observe(update_keys_for_bird, names='value')
        key_dropdown.observe(update_plot, names='value')
        plot_all_bouts_button.on_click(lambda _: plot_all_bouts_for_bird())
        save_figures_button.on_click(save_all_figures_callback)
        save_labels_button.on_click(lambda _: save_edits_to_syllable_df())
        def on_param_change(change):
            save_current_params(bird_dropdown.value)
            plot_all_bouts_for_bird()
            
        def rebind_slider_observers():
            umap_slider.observe(on_param_change, names='value')
            mcs_slider.observe(on_param_change, names='value')
            
        # Initial bind
        rebind_slider_observers()

        # Initialize key options and layout
        update_keys_for_bird({'new': bird_dropdown.value})
        display(widgets.VBox([
            widgets.HBox([bird_dropdown, key_dropdown, umap_slider, mcs_slider]),
            direction_toggle,
            plot_all_bouts_button,
            save_labels_button,
            widgets.HBox([save_figures_button]),
            plot_output
        ]))

        update_plot()
        if active_df is not None:
            syllable_df.loc[active_df.index, 'hdbscan_syntax_label'] = active_df['final_syntax']