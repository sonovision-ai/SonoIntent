import collections

from sonointent.build.build_split import assign, group_of, write_split


def _rows(dataset, n, prefix="s"):
    return [{"dataset": dataset, "sample_id": f"{prefix}{i}",
             "rendered_image": f"{dataset}/{prefix}{i}.png",
             "rendered_mask": f"{dataset}/{prefix}{i}__x_mask.png"} for i in range(n)]


def test_group_of_patient_level():
    assert group_of({"dataset": "CAMUS", "sample_id": "CAMUS_patient01__x"}) == "patient:patient01"
    assert group_of({"dataset": "BUSI", "sample_id": "BUSI_case1__x"}) == "img:BUSI_case1__x"


def test_ood_whole_dataset_goes_to_test():
    rows = _rows("BUS-UC", 7) + _rows("BUSI", 100)
    train, val, test = assign(rows)
    ood = [r for r in test if r["ood_type"] == "OOD_dataset"]
    assert len(ood) == 7
    assert not [r for r in train + val if r["dataset"] == "BUS-UC"]


def test_id_split_has_no_group_overlap():
    rows = []
    for p in range(20):
        rows += [{"dataset": "CAMUS", "sample_id": f"CAMUS_p{p}__f{i}",
                  "rendered_image": "a", "rendered_mask": "b"} for i in range(5)]
    train, val, test = assign(rows, seed=42)
    groups = [{(r["dataset"], group_of(r)) for r in part}
              for part in (train, val, test)]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert len(train) + len(val) + len(test) == len(rows)


def test_write_split_counts(tmp_path):
    rows = _rows("BUSI", 40)
    stats = write_split(rows, str(tmp_path), check_files=False)
    assert stats["train"] + stats["val"] + stats["test"] == 40
    assert stats["disjoint"]
    for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
        assert (tmp_path / name).exists()
