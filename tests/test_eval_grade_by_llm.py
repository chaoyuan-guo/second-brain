from eval.scripts.grade_by_llm import LLMJudge, compute_traceability_metrics


def test_compute_traceability_metrics_tracks_citation_accuracy_and_honesty_trigger():
    question = {
        "expected_sources": ["动态规划.md"],
    }
    answer = "动态规划的核心是状态转移 [c01]，但当前证据有限，建议核实。"
    tool_trace = [
        {
            "source_refs": [
                {"citation_id": "c01", "score": 0.91},
                {"citation_id": "c02", "score": 0.95},
            ]
        }
    ]

    metrics = compute_traceability_metrics(answer, question, tool_trace)

    assert metrics["inline_citation_coverage"] == 1.0
    assert metrics["precise_native_citation_coverage"] == 0.0
    assert metrics["citation_accuracy"] == 0.0
    assert metrics["total_citations"] == 1
    assert metrics["valid_citation_count"] == 0
    assert metrics["path_field_coverage"] == 0.0
    assert metrics["citation_id_field_coverage"] == 1.0
    assert metrics["snippet_field_coverage"] == 0.0
    assert metrics["char_offset_field_coverage"] == 0.0
    assert metrics["precise_source_ref_count"] == 0
    assert metrics["degraded_source_ref_count"] == 2
    assert metrics["has_precise_native_path"] is False
    assert metrics["should_trigger_honesty"] is True
    assert metrics["did_trigger_honesty"] is True


def test_build_report_aggregates_traceability_metrics():
    judge = object.__new__(LLMJudge)
    judge._model = "unit-test-model"

    testset = {
        "meta": {"name": "unit", "version": "1.0"},
        "questions": [
            {"id": "Q1", "category": "traceability"},
            {"id": "Q2", "category": "traceability"},
        ],
    }
    results = [
        {
            "id": "Q1",
            "passed": True,
            "total_score": 0.9,
            "scores": {
                "personalization": 4,
                "precision": 4,
                "honesty": 4,
                "traceability": 5,
            },
            "traceability_metrics": {
                "coverage_eligible": True,
                "inline_citation_coverage": 1.0,
                "precise_native_citation_coverage": 1.0,
                "total_citations": 2,
                "valid_citation_count": 2,
                "path_field_coverage": 1.0,
                "citation_id_field_coverage": 1.0,
                "snippet_field_coverage": 1.0,
                "char_offset_field_coverage": 1.0,
                "precise_source_ref_count": 2,
                "degraded_source_ref_count": 0,
                "has_precise_native_path": True,
                "did_trigger_honesty": True,
                "should_trigger_honesty": True,
            },
        },
        {
            "id": "Q2",
            "passed": False,
            "total_score": 0.5,
            "scores": {
                "personalization": 3,
                "precision": 3,
                "honesty": 3,
                "traceability": 2,
            },
            "traceability_metrics": {
                "coverage_eligible": True,
                "inline_citation_coverage": 0.0,
                "precise_native_citation_coverage": 0.0,
                "total_citations": 1,
                "valid_citation_count": 0,
                "path_field_coverage": 0.5,
                "citation_id_field_coverage": 0.0,
                "snippet_field_coverage": 0.0,
                "char_offset_field_coverage": 0.0,
                "precise_source_ref_count": 0,
                "degraded_source_ref_count": 1,
                "has_precise_native_path": False,
                "did_trigger_honesty": False,
                "should_trigger_honesty": True,
            },
        },
    ]

    report = judge._build_report(testset, results)
    metrics = report["summary"]["metrics"]

    assert metrics["inline_citation_coverage"] == 0.5
    assert metrics["precise_native_citation_coverage"] == 0.5
    assert metrics["citation_accuracy"] == 0.6667
    assert metrics["honesty_trigger_precision"] == 1.0
    assert metrics["path_field_coverage"] == 0.75
    assert metrics["citation_id_field_coverage"] == 0.5
    assert metrics["snippet_field_coverage"] == 0.5
    assert metrics["char_offset_field_coverage"] == 0.5
    assert metrics["precise_source_ref_count"] == 2
    assert metrics["degraded_source_ref_count"] == 1
    assert metrics["precise_native_main_path_hits"] == 1
