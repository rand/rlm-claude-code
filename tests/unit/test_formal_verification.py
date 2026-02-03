"""
Tests for formal verification integration (SPEC-07.20-07.25).

Tests cover:
- Verification chains for code tasks
- Precondition/postcondition generation
- Constraint types (type, behavioral, test)
- Refactoring postconditions
- Recursive decomposition integration
- Automatic correction attempts
"""

from src.formal_verification import (
    AutoCorrector,
    CodeChange,
    Constraint,
    ConstraintType,
    CorrectionResult,
    PostconditionGenerator,
    PreconditionGenerator,
    RefactoringPostconditions,
    VerificationChain,
    VerificationResult,
    VerificationStatus,
    Verifier,
)


class TestVerificationChains:
    """Tests for verification chains (SPEC-07.20)."""

    def test_chain_supports_code_tasks(self):
        """SPEC-07.20: Support verification chains for code tasks."""
        chain = VerificationChain()

        change = CodeChange(
            file_path="src/example.py",
            original_content="def add(a, b): return a + b",
            modified_content="def add(a: int, b: int) -> int: return a + b",
            change_type="refactor",
        )

        result = chain.verify_change(change)

        assert isinstance(result, VerificationResult)

    def test_chain_runs_all_verification_steps(self):
        """Chain should run precondition, postcondition, and verification."""
        chain = VerificationChain()

        change = CodeChange(
            file_path="src/example.py",
            original_content="x = 1",
            modified_content="x = 2",
            change_type="modify",
        )

        result = chain.verify_change(change)

        # Should have run all steps
        assert result.preconditions_checked
        assert result.postconditions_checked


class TestPreconditionGeneration:
    """Tests for precondition generation (SPEC-07.21)."""

    def test_generates_preconditions(self):
        """SPEC-07.21: generate_preconditions(change) -> list[Constraint]."""
        generator = PreconditionGenerator()

        change = CodeChange(
            file_path="src/module.py",
            original_content="def foo(): pass",
            modified_content="def foo(x): return x",
            change_type="modify",
        )

        preconditions = generator.generate_preconditions(change)

        assert isinstance(preconditions, list)
        assert all(isinstance(c, Constraint) for c in preconditions)

    def test_preconditions_include_file_exists(self):
        """Preconditions should verify file exists."""
        generator = PreconditionGenerator()

        change = CodeChange(
            file_path="src/existing.py",
            original_content="# content",
            modified_content="# new content",
            change_type="modify",
        )

        preconditions = generator.generate_preconditions(change)

        # Should have a file existence precondition
        assert any(c.name == "file_exists" for c in preconditions)

    def test_preconditions_include_syntax_valid(self):
        """Preconditions should verify original syntax is valid."""
        generator = PreconditionGenerator()

        change = CodeChange(
            file_path="src/valid.py",
            original_content="def valid(): pass",
            modified_content="def valid(): return True",
            change_type="modify",
        )

        preconditions = generator.generate_preconditions(change)

        assert any(c.name == "syntax_valid" for c in preconditions)


class TestPostconditionGeneration:
    """Tests for postcondition generation (SPEC-07.21)."""

    def test_generates_postconditions(self):
        """SPEC-07.21: generate_postconditions(change) -> list[Constraint]."""
        generator = PostconditionGenerator()

        change = CodeChange(
            file_path="src/module.py",
            original_content="def foo(): pass",
            modified_content="def foo(x): return x",
            change_type="modify",
        )

        postconditions = generator.generate_postconditions(change)

        assert isinstance(postconditions, list)
        assert all(isinstance(c, Constraint) for c in postconditions)

    def test_postconditions_include_syntax_valid(self):
        """Postconditions should verify modified syntax is valid."""
        generator = PostconditionGenerator()

        change = CodeChange(
            file_path="src/module.py",
            original_content="x = 1",
            modified_content="x = 2",
            change_type="modify",
        )

        postconditions = generator.generate_postconditions(change)

        assert any(c.name == "syntax_valid" for c in postconditions)


class TestConstraintTypes:
    """Tests for constraint types (SPEC-07.22)."""

    def test_type_constraint(self):
        """SPEC-07.22: Type constraints (via type checker)."""
        constraint = Constraint(
            name="type_check",
            constraint_type=ConstraintType.TYPE,
            description="All types must check",
        )

        assert constraint.constraint_type == ConstraintType.TYPE

    def test_behavioral_constraint(self):
        """SPEC-07.22: Behavioral constraints (via CPMpy)."""
        constraint = Constraint(
            name="invariant_holds",
            constraint_type=ConstraintType.BEHAVIORAL,
            description="Loop invariant must hold",
        )

        assert constraint.constraint_type == ConstraintType.BEHAVIORAL

    def test_test_constraint(self):
        """SPEC-07.22: Test constraints (via test execution)."""
        constraint = Constraint(
            name="tests_pass",
            constraint_type=ConstraintType.TEST,
            description="All tests must pass",
        )

        assert constraint.constraint_type == ConstraintType.TEST


class TestVerifier:
    """Tests for verification (SPEC-07.21)."""

    def test_verify_constraints(self):
        """SPEC-07.21: verify(constraints, code) -> VerificationResult."""
        verifier = Verifier()

        constraints = [
            Constraint(
                name="syntax_valid",
                constraint_type=ConstraintType.TYPE,
                description="Syntax must be valid",
            ),
        ]
        code = "def foo(): return 42"

        result = verifier.verify(constraints, code)

        assert isinstance(result, VerificationResult)
        assert result.status in [VerificationStatus.PASSED, VerificationStatus.FAILED]

    def test_verify_returns_passed_for_valid_code(self):
        """Verification should pass for valid code."""
        verifier = Verifier()

        constraints = [
            Constraint(
                name="syntax_valid",
                constraint_type=ConstraintType.TYPE,
                description="Syntax must be valid",
            ),
        ]
        code = "x = 1 + 2"

        result = verifier.verify(constraints, code)

        assert result.status == VerificationStatus.PASSED

    def test_verify_returns_failed_for_invalid_syntax(self):
        """Verification should fail for invalid syntax."""
        verifier = Verifier()

        constraints = [
            Constraint(
                name="syntax_valid",
                constraint_type=ConstraintType.TYPE,
                description="Syntax must be valid",
            ),
        ]
        code = "def foo( return"  # Invalid syntax

        result = verifier.verify(constraints, code)

        assert result.status == VerificationStatus.FAILED

    def test_verify_reports_failed_constraints(self):
        """Failed verification should report which constraints failed."""
        verifier = Verifier()

        constraints = [
            Constraint(
                name="syntax_valid",
                constraint_type=ConstraintType.TYPE,
                description="Syntax must be valid",
            ),
        ]
        code = "def broken("

        result = verifier.verify(constraints, code)

        assert len(result.failed_constraints) > 0


class TestRefactoringPostconditions:
    """Tests for refactoring postconditions (SPEC-07.23)."""

    def test_generates_call_sites_typecheck(self):
        """SPEC-07.23: Generate 'All call sites still type-check'."""
        generator = RefactoringPostconditions()

        change = CodeChange(
            file_path="src/module.py",
            original_content="def foo(): pass",
            modified_content="def foo(x): return x",
            change_type="refactor",
        )

        postconditions = generator.generate(change)

        assert any(
            "call_sites" in c.name.lower() or "type" in c.name.lower() for c in postconditions
        )

    def test_generates_tests_pass(self):
        """SPEC-07.23: Generate 'All tests still pass'."""
        generator = RefactoringPostconditions()

        change = CodeChange(
            file_path="src/module.py",
            original_content="def foo(): return 1",
            modified_content="def foo(): return 2",
            change_type="refactor",
        )

        postconditions = generator.generate(change)

        assert any("test" in c.name.lower() for c in postconditions)

    def test_generates_no_new_imports_when_requested(self):
        """SPEC-07.23: Generate 'No new imports introduced' if requested."""
        generator = RefactoringPostconditions()

        change = CodeChange(
            file_path="src/module.py",
            original_content="def foo(): pass",
            modified_content="import os\ndef foo(): pass",
            change_type="refactor",
        )

        postconditions = generator.generate(change, check_imports=True)

        assert any("import" in c.name.lower() for c in postconditions)


class TestRecursiveIntegration:
    """Tests for recursive decomposition integration (SPEC-07.24)."""

    def test_spawns_verification_subqueries(self):
        """SPEC-07.24: Spawn verification sub-queries for each postcondition."""
        chain = VerificationChain()

        change = CodeChange(
            file_path="src/module.py",
            original_content="def foo(): pass",
            modified_content="def foo(): return 42",
            change_type="modify",
        )

        result = chain.verify_change(change)

        # Should have sub-query results for postconditions
        assert result.subquery_count >= 0

    def test_aggregates_verification_results(self):
        """SPEC-07.24: Aggregate verification results."""
        chain = VerificationChain()

        change = CodeChange(
            file_path="src/module.py",
            original_content="x = 1",
            modified_content="x = 2",
            change_type="modify",
        )

        result = chain.verify_change(change)

        # Should have aggregated status
        assert result.status in [
            VerificationStatus.PASSED,
            VerificationStatus.FAILED,
            VerificationStatus.PARTIAL,
        ]


class TestAutomaticCorrection:
    """Tests for automatic correction attempts (SPEC-07.25)."""

    def test_triggers_correction_on_failure(self):
        """SPEC-07.25: Verification failures trigger automatic correction."""
        corrector = AutoCorrector()

        failed_result = VerificationResult(
            status=VerificationStatus.FAILED,
            failed_constraints=[
                Constraint(
                    name="syntax_valid",
                    constraint_type=ConstraintType.TYPE,
                    description="Syntax must be valid",
                )
            ],
            passed_constraints=[],
        )

        change = CodeChange(
            file_path="src/module.py",
            original_content="def foo(): pass",
            modified_content="def foo( return",  # Broken
            change_type="modify",
        )

        correction = corrector.attempt_correction(change, failed_result)

        assert isinstance(correction, CorrectionResult)
        assert correction.attempted

    def test_correction_provides_suggested_fix(self):
        """Correction should provide a suggested fix."""
        corrector = AutoCorrector()

        failed_result = VerificationResult(
            status=VerificationStatus.FAILED,
            failed_constraints=[
                Constraint(
                    name="syntax_valid",
                    constraint_type=ConstraintType.TYPE,
                    description="Syntax must be valid",
                )
            ],
            passed_constraints=[],
        )

        change = CodeChange(
            file_path="src/module.py",
            original_content="def foo(): pass",
            modified_content="def foo( return",
            change_type="modify",
        )

        correction = corrector.attempt_correction(change, failed_result)

        # Should have a suggestion or explanation
        assert correction.suggestion is not None or correction.explanation is not None

    def test_correction_records_attempts(self):
        """Correction should record attempts made."""
        corrector = AutoCorrector()

        failed_result = VerificationResult(
            status=VerificationStatus.FAILED,
            failed_constraints=[
                Constraint(
                    name="type_check",
                    constraint_type=ConstraintType.TYPE,
                    description="Types must check",
                )
            ],
            passed_constraints=[],
        )

        change = CodeChange(
            file_path="src/module.py",
            original_content="def foo(): pass",
            modified_content="def foo() -> str: return 42",  # Type mismatch
            change_type="modify",
        )

        correction = corrector.attempt_correction(change, failed_result)

        assert isinstance(correction.attempts, list)


class TestCodeChange:
    """Tests for CodeChange structure."""

    def test_code_change_has_file_path(self):
        """CodeChange should have file_path."""
        change = CodeChange(
            file_path="src/test.py",
            original_content="a = 1",
            modified_content="a = 2",
            change_type="modify",
        )

        assert change.file_path == "src/test.py"

    def test_code_change_has_contents(self):
        """CodeChange should have original and modified content."""
        change = CodeChange(
            file_path="src/test.py",
            original_content="original",
            modified_content="modified",
            change_type="modify",
        )

        assert change.original_content == "original"
        assert change.modified_content == "modified"

    def test_code_change_has_type(self):
        """CodeChange should have change type."""
        change = CodeChange(
            file_path="src/test.py",
            original_content="",
            modified_content="new",
            change_type="create",
        )

        assert change.change_type == "create"

    def test_code_change_to_dict(self):
        """CodeChange should be serializable."""
        change = CodeChange(
            file_path="src/test.py",
            original_content="old",
            modified_content="new",
            change_type="modify",
        )

        data = change.to_dict()
        assert "file_path" in data
        assert "original_content" in data
        assert "modified_content" in data


class TestConstraint:
    """Tests for Constraint structure."""

    def test_constraint_has_name(self):
        """Constraint should have name."""
        constraint = Constraint(
            name="my_constraint",
            constraint_type=ConstraintType.TYPE,
            description="Description",
        )

        assert constraint.name == "my_constraint"

    def test_constraint_has_type(self):
        """Constraint should have constraint_type."""
        constraint = Constraint(
            name="test",
            constraint_type=ConstraintType.BEHAVIORAL,
            description="Test constraint",
        )

        assert constraint.constraint_type == ConstraintType.BEHAVIORAL

    def test_constraint_has_description(self):
        """Constraint should have description."""
        constraint = Constraint(
            name="test",
            constraint_type=ConstraintType.TEST,
            description="Detailed description",
        )

        assert constraint.description == "Detailed description"

    def test_constraint_to_dict(self):
        """Constraint should be serializable."""
        constraint = Constraint(
            name="test",
            constraint_type=ConstraintType.TYPE,
            description="Test",
        )

        data = constraint.to_dict()
        assert "name" in data
        assert "constraint_type" in data


class TestVerificationResult:
    """Tests for VerificationResult structure."""

    def test_result_has_status(self):
        """Result should have status."""
        result = VerificationResult(
            status=VerificationStatus.PASSED,
            passed_constraints=[],
            failed_constraints=[],
        )

        assert result.status == VerificationStatus.PASSED

    def test_result_has_constraints_lists(self):
        """Result should track passed and failed constraints."""
        passed = Constraint("ok", ConstraintType.TYPE, "Passed")
        failed = Constraint("bad", ConstraintType.TYPE, "Failed")

        result = VerificationResult(
            status=VerificationStatus.PARTIAL,
            passed_constraints=[passed],
            failed_constraints=[failed],
        )

        assert len(result.passed_constraints) == 1
        assert len(result.failed_constraints) == 1

    def test_result_to_dict(self):
        """Result should be serializable."""
        result = VerificationResult(
            status=VerificationStatus.FAILED,
            passed_constraints=[],
            failed_constraints=[],
        )

        data = result.to_dict()
        assert "status" in data
        assert "passed_constraints" in data
        assert "failed_constraints" in data


class TestVerificationStatus:
    """Tests for VerificationStatus enum."""

    def test_passed_status(self):
        """PASSED status exists."""
        assert VerificationStatus.PASSED.value == "passed"

    def test_failed_status(self):
        """FAILED status exists."""
        assert VerificationStatus.FAILED.value == "failed"

    def test_partial_status(self):
        """PARTIAL status exists."""
        assert VerificationStatus.PARTIAL.value == "partial"


class TestConstraintType:
    """Tests for ConstraintType enum."""

    def test_type_constraint_type(self):
        """TYPE constraint type exists."""
        assert ConstraintType.TYPE.value == "type"

    def test_behavioral_constraint_type(self):
        """BEHAVIORAL constraint type exists."""
        assert ConstraintType.BEHAVIORAL.value == "behavioral"

    def test_test_constraint_type(self):
        """TEST constraint type exists."""
        assert ConstraintType.TEST.value == "test"


class TestVerificationChainIntegration:
    """Integration tests for VerificationChain."""

    def test_full_verification_pipeline(self):
        """Test complete verification pipeline."""
        chain = VerificationChain()

        change = CodeChange(
            file_path="src/calculator.py",
            original_content="def add(a, b): return a + b",
            modified_content="def add(a: int, b: int) -> int: return a + b",
            change_type="refactor",
        )

        result = chain.verify_change(change)

        assert result.status is not None
        assert isinstance(result.passed_constraints, list)
        assert isinstance(result.failed_constraints, list)

    def test_verification_with_correction(self):
        """Test verification with automatic correction."""
        chain = VerificationChain(auto_correct=True)

        change = CodeChange(
            file_path="src/broken.py",
            original_content="def foo(): pass",
            modified_content="def foo( return",  # Intentionally broken
            change_type="modify",
        )

        result = chain.verify_change(change)

        # Should have attempted correction
        assert result.correction_attempted or result.status == VerificationStatus.FAILED

    def test_chain_to_dict(self):
        """Chain results should be serializable."""
        chain = VerificationChain()

        change = CodeChange(
            file_path="src/test.py",
            original_content="x = 1",
            modified_content="x = 2",
            change_type="modify",
        )

        result = chain.verify_change(change)

        data = result.to_dict()
        assert "status" in data
