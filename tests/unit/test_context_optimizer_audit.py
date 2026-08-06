"""Unit tests for AuditRecorder and metadata-only JSONL audit logging in ContextOptimizer."""

import json
import os
import pytest
from context_optimizer import (
    AUDIT_ALLOWLIST_FIELDS,
    AuditRecorder,
    ContextOptimizer,
    OptimizationAudit,
    OptimizationResult,
)


class TestAuditRecorder:
    """Tests for AuditRecorder functionality."""

    def test_audit_recorder_creates_directory_and_file(self, tmp_path):
        log_dir = str(tmp_path / "audit_logs")
        recorder = AuditRecorder(log_dir=log_dir)
        assert os.path.exists(log_dir)
        assert os.path.exists(os.path.join(log_dir, "audit.jsonl"))
        recorder.close()

    def test_audit_recorder_writes_metadata_only(self, tmp_path):
        log_dir = str(tmp_path / "audit_logs")
        recorder = AuditRecorder(log_dir=log_dir)

        audit = OptimizationAudit(
            strategy="safe",
            original_cost=1000,
            optimized_cost=800,
            savings_tokens=200,
            transformations_applied=["dedup", "whitespace"],
            protected_units_preserved=2,
            blocks_removed=1,
            blocks_merged=0,
            blocks_deduplicated=1,
            validation_passed=True,
            duration_ms=12.5,
        )

        recorder.record(audit, extra={"model": "gpt-4o", "sensitive_payload": "SECRET_DATA_LEAK"})
        recorder.close()

        file_path = os.path.join(log_dir, "audit.jsonl")
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 1
        record = json.loads(lines[0])

        assert record["strategy"] == "safe"
        assert record["original_cost"] == 1000
        assert record["optimized_cost"] == 800
        assert record["savings_tokens"] == 200
        assert "dedup" in record["transformations_applied"]
        assert record["model"] == "gpt-4o"

        # Verify sensitive payload was excluded by strict allowlist
        assert "sensitive_payload" not in record
        assert "SECRET_DATA_LEAK" not in lines[0]

        # All fields in record must belong to AUDIT_ALLOWLIST_FIELDS
        for key in record:
            assert key in AUDIT_ALLOWLIST_FIELDS

    def test_audit_recorder_rotation(self, tmp_path):
        log_dir = str(tmp_path / "audit_logs")
        # Use small max_bytes to force rotation
        recorder = AuditRecorder(log_dir=log_dir, max_bytes=200, backup_count=3)

        audit = OptimizationAudit(
            strategy="safe",
            original_cost=500,
            optimized_cost=400,
            savings_tokens=100,
            duration_ms=5.0,
        )

        # Write enough records to trigger multiple rotations
        for _ in range(10):
            recorder.record(audit)

        recorder.close()

        # Check rotated files exist
        assert os.path.exists(os.path.join(log_dir, "audit.jsonl"))
        assert os.path.exists(os.path.join(log_dir, "audit.jsonl.1"))

    def test_audit_recorder_query_paginated(self, tmp_path):
        log_dir = str(tmp_path / "audit_logs")
        recorder = AuditRecorder(log_dir=log_dir)

        for i in range(15):
            audit = OptimizationAudit(
                strategy="safe" if i % 2 == 0 else "fail_open",
                original_cost=100 * (i + 1),
                optimized_cost=80 * (i + 1),
                savings_tokens=20 * (i + 1),
                duration_ms=1.0 * i,
            )
            recorder.record(audit)

        recorder.close()

        # Query page 1 (per_page=5)
        query_recorder = AuditRecorder(log_dir=log_dir)
        res = query_recorder.query(page=1, per_page=5)
        assert res["page"] == 1
        assert res["per_page"] == 5
        assert res["total"] == 15
        assert res["pages"] == 3
        assert len(res["items"]) == 5

        # Test strategy filter
        res_safe = query_recorder.query(page=1, per_page=20, strategy_filter="safe")
        assert res_safe["total"] == 8
        for item in res_safe["items"]:
            assert item["strategy"] == "safe"
            # Allowlist check
            for key in item:
                assert key in AUDIT_ALLOWLIST_FIELDS

        query_recorder.close()


class TestContextOptimizerAuditIntegration:
    """Tests for ContextOptimizer integration with AuditRecorder."""

    @pytest.mark.asyncio
    async def test_context_optimizer_records_audit(self, tmp_path):
        log_dir = str(tmp_path / "audit_logs")
        recorder = AuditRecorder(log_dir=log_dir)
        optimizer = ContextOptimizer(audit_recorder=recorder)

        payload = {
            "model": "local-model",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
        }
        backend_info = {"backend_type": "local", "context_size": 8192, "parallel_slots": 1}

        result = await optimizer.optimize(payload, backend_info)

        assert isinstance(result, OptimizationResult)
        assert result.audit.strategy == "safe"

        # Verify record was written to audit.jsonl
        file_path = os.path.join(log_dir, "audit.jsonl")
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["strategy"] == "safe"
        assert data["original_cost"] > 0
        recorder.close()

    @pytest.mark.asyncio
    async def test_context_optimizer_query_audit_logs(self, tmp_path):
        log_dir = str(tmp_path / "audit_logs")
        recorder = AuditRecorder(log_dir=log_dir)
        optimizer = ContextOptimizer(audit_recorder=recorder)

        payload = {
            "messages": [
                {"role": "system", "content": "Sys prompt"},
                {"role": "user", "content": "Test query"},
            ]
        }
        backend_info = {"backend_type": "local", "context_size": 4096}

        await optimizer.optimize(payload, backend_info)

        res = optimizer.query_audit_logs(page=1, per_page=10)
        assert res["total"] == 1
        assert len(res["items"]) == 1
        assert res["items"][0]["strategy"] == "safe"

        recorder.close()
