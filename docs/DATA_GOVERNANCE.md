# Data governance

SonoIntent aggregates 30 existing ultrasound datasets. Each source keeps its
own license, citation and access terms. This repository redistributes no
images, masks or annotations.

## Release tiers

| tier | sources | policy |
|---|---:|---|
| candidate | 9 | license basis identified; assets released only after written confirmation |
| restricted_gated | 5 | hosted under the original terms (non-commercial, share-alike or research-use); access control does not grant redistribution rights |
| hold_prohibited | 1 | EchoNet-Dynamic — no redistribution; official application only |
| hold_unclear | 15 | no data-level license confirmed; users obtain the original data |

## Privacy

Public records use opaque per-task identifiers and relative asset paths.
Internal identifiers, local paths and the identifier mapping are not
published. Image metadata contains no patient names or hospital identifiers;
burned-in pixel text is checked before any image release. Secondary
publication of images also requires ethics approval, which is not yet in
place.

## Release gates

- **G0 license:** written per-source permission, license copies, version and
  SHA-256.
- **G1 privacy:** de-identification checks completed for metadata and
  burned-in text.
- **G2 reproducibility:** counts, hashes and split invariants verifiable.
- **G3 integrity:** package checksums pass.
- **G4 governance:** dataset card, NOTICE, takedown contact and version
  policy in place.
