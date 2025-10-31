from lighter import System

class FixedSystem(System):
    """Fix for mode=None bug in old Lighter."""
    def training_step(self, batch, batch_idx):
        self.mode = "train"
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        self.mode = "val"
        return super().validation_step(batch, batch_idx)
