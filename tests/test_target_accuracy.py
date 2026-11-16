import pytest

from sonointent.eval.target_accuracy import classify, target_of


def test_delineate_object_wins():
    text = ("Therefore, delineate the myocardial layer between the endocardium "
            "and epicardium surrounding the left ventricular cavity.")
    r = classify(text, "left_ventricular_myocardium")
    assert r["label"] == "correct"
    assert r["slug"] == "left_ventricular_myocardium"


def test_exclusion_tail_is_cut():
    text = ("Therefore, delineate the left ventricular cavity enclosed by the "
            "myocardium, excluding the left ventricular myocardium.")
    r = classify(text, "left_ventricular_cavity")
    assert r["label"] == "correct"


def test_class_alias_uses_dataset_class():
    assert classify("Therefore, delineate the lesion itself.",
                    "breast_lesion")["label"] == "correct"
    assert classify("Therefore, delineate the lesion itself.",
                    "examined_muscle")["label"] == "incorrect"


def test_unresolved_without_target():
    assert classify("The image shows a hypoechoic region.", "left_atrium")["label"] == "unresolved"


def test_target_of_reads_mask_slug():
    row = {"rendered_mask": "rendered/BUSI/BUSI__case01__breast_lesion_mask.png",
           "dataset": "BUSI"}
    assert target_of(row) == "乳腺病灶"


def test_target_of_falls_back_to_dataset():
    row = {"rendered_mask": "rendered/CAMUS/CAMUS__p01__frame.png",
           "dataset": "CAMUS"}
    assert target_of(row) == "左心室腔"
