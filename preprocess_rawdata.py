from pathlib import Path
from nn.dataloader import read_ss_curves, read_x_and_y_from_table

# Load raw data (ss curves and table)
raw_data_dir = "./raw_data"
table_filename = "table.csv"
x_lstm = read_ss_curves(Path(raw_data_dir))
x_ann, y_ann = read_x_and_y_from_table(Path(raw_data_dir) / table_filename)
y_ann = y_ann.loc[:, ["strength", "lengthavg"]]

# Filter out invalid data from x_ann and y_ann
keys = (
    set(x_ann.index)
    & set(y_ann.index)
    & set(x_lstm["Name"].drop_duplicates())
)
print(f"===== Number of valid data: {len(keys)} =====")
x_ann = x_ann[x_ann.index.isin(keys)]
y_ann = y_ann[y_ann.index.isin(keys)]
x_lstm = x_lstm[x_lstm["Name"].isin(keys)]
print(f"===== x_ann: {x_ann.shape} =====")
print(f"===== y_ann: {y_ann.shape} =====")
print(f"===== x_lstm: {x_lstm.shape} =====")
print(x_lstm.head(3))
