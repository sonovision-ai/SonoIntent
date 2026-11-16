"""The four SonoIntent test settings."""
SETTINGS = ("ID", "OOD_dataset", "OOD_organ", "OOD_intent")

def setting_of(row):
    return row.get("ood_type") or "ID"
