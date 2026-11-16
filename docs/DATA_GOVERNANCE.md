# Data governance

SonoIntent aggregates 30 existing ultrasound datasets; every source keeps its
own license, citation and access terms. This repository ships **no data** —
no images, masks, annotations or indexes.

## Release tiers

| tier | sources | policy |
|---|---|---|
| candidate | 9 | license basis found; assets published only after written confirmation |
| restricted_gated | 5 | gated hosting under the original terms (non-commercial / share-alike / research-use); a gate is access control, not a redistribution authorization |
| hold_prohibited | 1 | EchoNet-Dynamic: no redistribution of any kind; official application entry point only |
| hold_unclear | 15 | rebuild-only until a data-level license is confirmed |

## Privacy

Public artifacts use opaque per-task identifiers and portable asset paths.
Internal identifiers, local filesystem paths, the identifier mapping and its
secret key are never published. Metadata scans found no patient names or
hospital identifiers in image headers; burned-in pixel text still needs a
final manual check before any asset release, and no IRB / consent record for
secondary publication has been located yet — both are release gates.

## Release gates

- G0 license: written/per-source permission, license copies, version + SHA-256
- G1 privacy: de-identification checks completed (metadata + burned-in text)
- G2 reproducibility: counts, hashes and split invariants verifiable
- G3 integrity: package checksums pass
- G4 governance: dataset card, NOTICE, takedown contact, version policy
