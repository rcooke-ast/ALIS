---
name: atomic-data
description: Add, validate, or convert atomic data entries for the ALIS atomic database, checking for duplicates and verifying physical units.
---

Work with the ALIS atomic data file (`alis/data/atomic.xml`) to add new entries, validate existing ones, or convert the database to a new format.

## Current format (`alis/data/atomic.xml`)

Each entry stores:
- **Ion name** — e.g. `1H_I`, `6C_IV` (format: `<atomic_number><element_symbol>_<ionisation>`)
- **Rest wavelength** — in Ångströms (Å)
- **Oscillator strength** — dimensionless f-value
- **Damping constant** — Gamma in s⁻¹
- **Atomic mass** — in atomic mass units (amu)

The file also has an `nrows` attribute that must match the total number of entries exactly — this must be updated manually whenever entries are added or removed.

## Steps

**To add a new entry:**
1. Ask the user for the transition details: ion, wavelength (Å), f-value, Gamma (s⁻¹), atomic mass (amu), and the source reference (e.g. NIST ASD).
2. Read `alis/data/atomic.xml` and check for duplicates: same ion and wavelength within 0.01 Å.
3. Verify the values are physically reasonable:
   - Wavelength: 100–10000 Å (UV to near-IR)
   - f-value: 0 < f ≤ 2
   - Gamma: typically 10⁶–10¹⁰ s⁻¹
4. Add the entry in sorted order (by wavelength within each ion group).
5. Increment `nrows` to reflect the new count.
6. Report what was added and the updated `nrows`.

**To validate existing entries:**
1. Read the full `atomic.xml` file.
2. Check: correct XML structure, `nrows` matches the actual row count, no duplicate wavelengths per ion, all values within physical ranges.
3. Report any issues found, with the ion name and wavelength of each problematic entry.

**To convert to a new format:**
1. Discuss the target format with the user (e.g. ECSV, JSON, HDF5, SQLite).
2. Read all entries from `atomic.xml` and write them to the new format, preserving all fields.
3. Verify round-trip fidelity: read back and compare every field.

## Notes

- The authoritative source for atomic transition data is the NIST Atomic Spectra Database (ASD) at https://www.nist.gov/pml/atomic-spectra-database.
- Always record the source reference when adding new entries.
- Do not modify `atomic.xml` without explicit instruction from the user.
