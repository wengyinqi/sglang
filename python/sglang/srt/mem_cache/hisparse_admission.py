"""Lifetime backing reservations shared by CPU-DDR and MPU HiSparse runs."""


class HiSparseBackingBudget:
    def __init__(self, capacity, page_size, compress_ratio):
        if any(
            type(x) is not int or x <= 0 for x in (capacity, page_size, compress_ratio)
        ):
            raise ValueError("Backing budget requires positive integer capacities")
        self.capacity = capacity
        self.page_size = page_size
        self.compress_ratio = compress_ratio
        self.reserved = 0
        self.leases = {}

    def slots_for(self, prompt_tokens, max_new_tokens):
        if any(type(x) is not int or x < 0 for x in (prompt_tokens, max_new_tokens)):
            raise ValueError("Request token budgets must be nonnegative integers")
        full = prompt_tokens + max_new_tokens
        compressed = (full + self.compress_ratio - 1) // self.compress_ratio
        return max(
            self.page_size,
            (compressed + self.page_size - 1) // self.page_size * self.page_size,
        )

    def reserve(self, rid, slots):
        if type(slots) is not int or slots <= 0:
            raise ValueError("Backing lease must be a positive integer")
        if rid in self.leases:
            if self.leases[rid] != slots:
                raise ValueError("Live request changed its backing reservation")
            return True
        if slots > self.capacity - self.reserved:
            return False
        self.leases[rid] = slots
        self.reserved += slots
        return True

    def release(self, rid):
        self.reserved -= self.leases.pop(rid, 0)
