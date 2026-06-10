import csv
import os
import statistics


def calc_stats(results):
    best = max(results)
    worst = min(results)
    avg = statistics.mean(results)
    std = statistics.stdev(results) if len(results) > 1 else 0.0
    return best, worst, avg, std


def build_row_from_schema(values, schema):
    row = {}
    for item in schema:
        value = values[item["key"]]
        if "digits" in item:
            value = round(value, item["digits"])
        row[item["column"]] = value
    return row


def validate_and_order_row(row, columns):
    missing = [column for column in columns if column not in row]
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")
    return {column: row[column] for column in columns}


def append_csv_rows(csv_filepath, columns, rows):
    file_exists = os.path.isfile(csv_filepath)
    with open(csv_filepath, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, delimiter=";")

        if not file_exists:
            writer.writeheader()

        for row in rows:
            writer.writerow(validate_and_order_row(row, columns))
            file.flush()
            print(f"Case {row['Case_ID']} saved!")
