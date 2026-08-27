from pathlib import Path

import pandas as pd

from app.services.taxonomy import TAXONOMY

GOLD_PATH = Path("evals/gold/voc_gold_seed_v0.2.csv")


def test_gold_seed_has_200_unique_rows_and_full_taxonomy_coverage():
    gold = pd.read_csv(GOLD_PATH)
    assert len(gold) == 200
    assert gold["ticket_id"].is_unique
    expected = {subcategory for values in TAXONOMY.values() for subcategory in values}
    assert set(gold["subcategory_gold"]) == expected


def test_gold_seed_labels_match_category_taxonomy_pairs():
    gold = pd.read_csv(GOLD_PATH)
    for row in gold.itertuples(index=False):
        assert row.category_gold in TAXONOMY
        assert row.subcategory_gold in TAXONOMY[row.category_gold]


def test_synthetic_seed_is_explicitly_provisional():
    gold = pd.read_csv(GOLD_PATH)
    assert set(gold["source_type"]) == {"synthetic"}
    assert set(gold["label_status"]) == {"provisional"}
    assert not gold["label_status"].eq("reviewed").any()
