#!/usr/bin/env python3
"""
Append experiment results to comparison table with two-level headers.
"""
import sys
import os
from pathlib import Path

def parse_table(content):
    """Parse existing markdown table."""
    lines = content.split('\n')

    # Find table start
    table_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith('| Sample'):
            table_start = i
            break

    if table_start is None:
        return None, None, lines

    # Extract header, separator, and rows
    header = lines[table_start]
    separator = lines[table_start + 1] if table_start + 1 < len(lines) else None

    rows = []
    i = table_start + 2
    while i < len(lines) and lines[i].strip().startswith('|'):
        rows.append(lines[i])
        i += 1

    before_table = lines[:table_start]
    after_table = lines[i:] if i < len(lines) else []

    return (header, separator, rows), (before_table, after_table), lines

def create_new_table(experiment_name, results):
    """Create new table with first experiment."""
    header = f"| Sample | {experiment_name} ||||"
    subheader = "|        | 1 | 2 | 3 | 4 |"
    separator = "|--------|---|---|---|---|"

    rows = []
    for sample, methods in sorted(results.items()):
        row = f"| {sample} | {methods[0]} | {methods[1]} | {methods[2]} | {methods[3]} |"
        rows.append(row)

    return '\n'.join([header, subheader, separator] + rows)

def append_to_table(existing_table, experiment_name, results):
    """Append experiment columns to existing table."""
    header, separator, rows = existing_table

    # Parse existing header to count experiments
    # Update header
    new_header = header.rstrip('|').rstrip() + f" {experiment_name} ||||"

    # Update subheader (extract existing subheader from second row if it exists)
    # For now, assume standard format
    new_subheader = separator.replace('|--------|', '|        |').rstrip('|').rstrip() + " 1 | 2 | 3 | 4 |"

    # Update separator
    new_separator = separator.rstrip('|').rstrip() + "---|---|---|---|"

    # Update data rows
    new_rows = []
    for row in rows:
        sample_name = row.split('|')[1].strip()
        if sample_name in results:
            methods = results[sample_name]
            new_row = row.rstrip('|').rstrip() + f" {methods[0]} | {methods[1]} | {methods[2]} | {methods[3]} |"
            new_rows.append(new_row)
        else:
            # Sample not in new results, add empty cells
            new_row = row.rstrip('|').rstrip() + " - | - | - | - |"
            new_rows.append(new_row)

    # Add any new samples
    existing_samples = {row.split('|')[1].strip() for row in rows}
    for sample in sorted(results.keys()):
        if sample not in existing_samples:
            # Need to fill previous experiment columns with "-"
            num_prev_experiments = (len(separator.split('|')) - 2) // 4  # -2 for Sample column and trailing
            prev_cols = " - | - | - | - |" * num_prev_experiments
            methods = results[sample]
            new_row = f"| {sample} |{prev_cols} {methods[0]} | {methods[1]} | {methods[2]} | {methods[3]} |"
            new_rows.append(new_row)

    return '\n'.join([new_header, new_subheader, new_separator] + new_rows)

def main():
    if len(sys.argv) < 4:
        print("Usage: append_experiment.py <results_md> <output_file> <experiment_name>")
        sys.exit(1)

    results_file = sys.argv[1]
    output_file = sys.argv[2]
    experiment_name = sys.argv[3]

    # Parse new results from compare_methods output
    with open(results_file) as f:
        content = f.read()

    results = {}
    for line in content.split('\n'):
        if line.strip().startswith('|') and '|' in line[1:]:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 6 and parts[1] not in ('Sample', ''):
                sample = parts[1]
                if sample and not sample.startswith('-'):
                    results[sample] = [parts[2], parts[3], parts[4], parts[5]]

    # Check if output file exists
    if os.path.exists(output_file):
        with open(output_file) as f:
            existing_content = f.read()

        table_data, positions, _ = parse_table(existing_content)

        if table_data:
            # Append to existing table
            new_table = append_to_table(table_data, experiment_name, results)
            before, after = positions

            # Rebuild content
            output_content = '\n'.join(before) + '\n' + new_table + '\n' + '\n'.join(after)
        else:
            # No table found, create new
            title = f"# Method Comparison Results\n\nMethods: **1**=MedianMean | **2**=Darkest | **3**=MostSaturated | **4**=Mode\n\n"
            new_table = create_new_table(experiment_name, results)
            output_content = title + new_table + "\n\n## Experiment Parameters\n"
    else:
        # Create new file
        title = f"# Method Comparison Results\n\nMethods: **1**=MedianMean | **2**=Darkest | **3**=MostSaturated | **4**=Mode\n\n"
        new_table = create_new_table(experiment_name, results)
        output_content = title + new_table + "\n\n## Experiment Parameters\n"

    with open(output_file, 'w') as f:
        f.write(output_content)

    print(f"Updated {output_file} with {experiment_name}")

if __name__ == '__main__':
    main()
