data = {
    'Ours (k=7)': 82.1,
    'Reformer': 81.7,
    'BigBird': 80.0,
    'Scatterbrain': 79.5,
    'Longformer': 78.8,
    'Sinkhorn': 76.0,
    'Synthesizr': 74.0,
    'Performer': 74.0,
    'Cosformer': 62.2,
}
my_key = 'Ours (k=7)'

col_keys = list(data.keys())

cell_data = []
cell_data.append(['MNLI (Acc.)'] + [data[c] for c in col_keys])
cell_data.append(['MNLI (Rel. Acc.)'] + [data[c] / data[my_key] * 100 for c in col_keys])
just_width = 13
just_width_header = 20
def format(lst):
    return [f'{i:.1f}'.rjust(just_width) for i in lst]
table_data = ""
for row in cell_data:
    if isinstance(row, list):
        name, row_data = row[0], row[1:]
        table_data += '&'.join([name.rjust(just_width_header)]+format(row_data))
        table_data += '\\\\\n'
    elif isinstance(row, str):
        table_data += row
    else:
        raise Exception()


table_cells = "|".join(['c'] * len(cell_data[0]))
table_header = "&".join(["".rjust(just_width_header)] + [k.rjust(just_width) for k in col_keys])
table = \
f"\\begin{{table}}[h]\n"\
f"\\caption{{glue benchmark}}\n"\
f"\\label{{table.baseline.glue}}\n"\
f"\\begin{{center}}\n"\
f"\\begin{{tabular}}{{{table_cells}}}\n"\
f"{table_header}\\\\\n"\
f"\\hline\n"\
f"{table_data}"\
f"\\end{{tabular}}\n"\
f"\\end{{center}}\n"\
f"\\end{{table}}"

print(table)