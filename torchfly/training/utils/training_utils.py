import torch

def check_if_cycle(update_count, cycle_count):
    return update_count % cycle_count == cycle_count - 1