"""CPU control-plane reservations for full backing KV capacity, without payload."""


class MPUAdmissionBudget:
    """Conservative input + maximum output leases, held until request release.

    Full KV survives early freeing of logical indices. Reserving the entire
    lifetime also prevents multiple in-flight prefills or decode growth from
    overcommitting the backing store. Slot allocation itself remains in the
    MPU pool. Scheduler access is single-threaded.
    """

    def __init__(self, capacity: int):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("MPU admission capacity must be a positive integer")
        self.capacity = capacity
        self.reserved = 0
        self._leases: dict[str, int] = {}

    def reserve(self, rid: str, slots: int) -> bool:
        if isinstance(slots, bool) or not isinstance(slots, int) or slots <= 0:
            raise ValueError("MPU reservation must be a positive integer")
        previous = self._leases.get(rid, 0)
        if previous:
            if previous != slots:
                raise ValueError("An active MPU request changed its KV reservation")
            return True
        if slots > self.capacity - self.reserved:
            return False
        self._leases[rid] = slots
        self.reserved += slots
        return True

    def release(self, rid: str) -> None:
        self.reserved -= self._leases.pop(rid, 0)


# Both DDR and MPU experiment variants share this admission policy. Keep the
# original public name for existing MPU callers and unit tests.
BackingAdmissionBudget = MPUAdmissionBudget
