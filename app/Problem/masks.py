import numpy as np


def repair_zero_mask(mask, generator):
    values = np.asarray(mask, dtype=int).reshape(-1).copy()
    if values.size == 0:
        return values
    if int(values.sum()) == 0:
        values[int(generator.integers(0, values.size))] = 1
    return values


def repair_zero_masks(masks, generator):
    values = np.asarray(masks, dtype=int)
    if values.ndim == 1:
        return repair_zero_mask(values, generator)

    repaired = values.copy()
    zero_rows = np.where(repaired.sum(axis=1) == 0)[0]
    if zero_rows.size:
        columns = generator.integers(0, repaired.shape[1], size=zero_rows.size)
        repaired[zero_rows, columns] = 1
    return repaired
