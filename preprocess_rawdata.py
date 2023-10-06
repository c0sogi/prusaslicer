from pathlib import Path
from nn.dataloader import _read_ss_curves, _read_x_and_y_from_table

# Load raw data (ss curves and table)
raw_data_dir = "./raw_data"
table_filename = "table.csv"
x_lstm = _read_ss_curves(Path(raw_data_dir))
x_ann, y_ann = _read_x_and_y_from_table(Path(raw_data_dir) / table_filename)
y_ann = y_ann.loc[:, ["strength", "lengthavg"]]

# Filter out invalid data from x_ann and y_ann
keys = set(x_ann.index) & set(y_ann.index) & set(x_lstm.index)
print(f"===== Number of valid data: {len(keys)} =====")
x_ann = x_ann[x_ann.index.isin(keys)]
y_ann = y_ann[y_ann.index.isin(keys)]
x_lstm = x_lstm[x_lstm.index.isin(keys)]
_shape = x_lstm.shape


# Merge x_lstm, x_ann, and y_ann
for to_merge in (x_ann, y_ann):
    x_lstm = x_lstm.merge(to_merge, on="Name", how="left")
x_lstm.dropna(inplace=True)
assert _shape[0] == x_lstm.shape[0], f"{_shape} != {x_lstm.shape}"
