from atcg.data import nearest_accessions, normalize_gtdb_accession


def test_nearest_accessions_uses_patristic_distance_and_stable_ties() -> None:
    tree = (
        "((RS_GCF_000000001.1:1,GCF_000000002.1:2):1,(GB_GCA_000000003.1:1,GCF_000000004.1:3):4);"
    )

    neighbors = nearest_accessions(
        tree,
        anchor_accession="GCF_000000001.1",
        candidate_accessions={
            "GCF_000000002.1",
            "GCA_000000003.1",
            "GCF_000000004.1",
        },
        limit=3,
    )

    assert neighbors == (
        ("GCF_000000002.1", 3.0),
        ("GCA_000000003.1", 7.0),
        ("GCF_000000004.1", 9.0),
    )
    assert normalize_gtdb_accession("RS_GCF_1.1") == "GCF_1.1"
