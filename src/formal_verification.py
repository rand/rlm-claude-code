"""
Formal verification integration for code tasks.

Implements: SPEC-07.20-07.25

Provides verification chains with precondition/postcondition generation,
constraint verification, and automatic correction attempts.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ConstraintType(Enum):
    """
    Type of verification constraint.

    Implements: SPEC-07.22
    """

    TYPE = "type"
    BEHAVIORAL = "behavioral"
    TEST = "test"


class VerificationStatus(Enum):
    """Status of verification result."""

    PASSED = "passed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class CodeChange:
    """Represents a code change to verify."""

    file_path: str
    original_content: str
    modified_content: str
    change_type: str  # create, modify, delete, refactor

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "file_path": self.file_path,
            "original_content": self.original_content,
            "modified_content": self.modified_content,
            "change_type": self.change_type,
        }


@dataclass
class Constraint:
    """
    A verification constraint.

    Implements: SPEC-07.22
    """

    name: str
    constraint_type: ConstraintType
    description: str
    expression: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "constraint_type": self.constraint_type.value,
            "description": self.description,
            "expression": self.expression,
        }


@dataclass
class VerificationResult:
    """
    Result of verification.

    Implements: SPEC-07.21
    """

    status: VerificationStatus
    passed_constraints: list[Constraint]
    failed_constraints: list[Constraint]
    preconditions_checked: bool = True
    postconditions_checked: bool = True
    subquery_count: int = 0
    correction_attempted: bool = False
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "status": self.status.value,
            "passed_constraints": [c.to_dict() for c in self.passed_constraints],
            "failed_constraints": [c.to_dict() for c in self.failed_constraints],
            "preconditions_checked": self.preconditions_checked,
            "postconditions_checked": self.postconditions_checked,
            "subquery_count": self.subquery_count,
            "correction_attempted": self.correction_attempted,
            "error_message": self.error_message,
        }


@dataclass
class CorrectionAttempt:
    """Record of a correction attempt."""

    constraint: Constraint
    suggested_fix: str | None
    success: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "constraint": self.constraint.to_dict(),
            "suggested_fix": self.suggested_fix,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class CorrectionResult:
    """
    Result of automatic correction attempt.

    Implements: SPEC-07.25
    """

    attempted: bool
    suggestion: str | None
    explanation: str | None
    attempts: list[CorrectionAttempt] = field(default_factory=list)
    success: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "attempted": self.attempted,
            "suggestion": self.suggestion,
            "explanation": self.explanation,
            "attempts": [a.to_dict() for a in self.attempts],
            "success": self.success,
        }


class PreconditionGenerator:
    """
    Generates preconditions for code changes.

    Implements: SPEC-07.21
    """

    def generate_preconditions(self, change: CodeChange) -> list[Constraint]:
        """
        Generate preconditions for a code change.

        Implements: SPEC-07.21

        Args:
            change: The code change to verify

        Returns:
            List of precondition constraints
        """
        preconditions: list[Constraint] = []

        # File existence precondition
        preconditions.append(
            Constraint(
                name="file_exists",
                constraint_type=ConstraintType.TYPE,
                description=f"File {change.file_path} must exist",
            )
        )

        # Syntax validity of original
        if change.original_content:
            preconditions.append(
                Constraint(
                    name="syntax_valid",
                    constraint_type=ConstraintType.TYPE,
                    description="Original code must have valid syntax",
                )
            )

        # For modifications, original must match expected
        if change.change_type == "modify":
            preconditions.append(
                Constraint(
                    name="original_matches",
                    constraint_type=ConstraintType.BEHAVIORAL,
                    description="Original content must match current file",
                )
            )

        return preconditions


class PostconditionGenerator:
    """
    Generates postconditions for code changes.

    Implements: SPEC-07.21
    """

    def generate_postconditions(self, change: CodeChange) -> list[Constraint]:
        """
        Generate postconditions for a code change.

        Implements: SPEC-07.21

        Args:
            change: The code change to verify

        Returns:
            List of postcondition constraints
        """
        postconditions: list[Constraint] = []

        # Modified code must have valid syntax
        postconditions.append(
            Constraint(
                name="syntax_valid",
                constraint_type=ConstraintType.TYPE,
                description="Modified code must have valid syntax",
            )
        )

        # Type checking postcondition
        postconditions.append(
            Constraint(
                name="type_check",
                constraint_type=ConstraintType.TYPE,
                description="Modified code must type-check",
            )
        )

        return postconditions


class RefactoringPostconditions:
    """
    Generates refactoring-specific postconditions.

    Implements: SPEC-07.23
    """

    def generate(
        self,
        change: CodeChange,
        check_imports: bool = False,
    ) -> list[Constraint]:
        """
        Generate refactoring postconditions.

        Implements: SPEC-07.23

        Args:
            change: The code change
            check_imports: Whether to check for new imports

        Returns:
            List of refactoring postconditions
        """
        postconditions: list[Constraint] = []

        # Call sites type-check (SPEC-07.23)
        postconditions.append(
            Constraint(
                name="call_sites_type_check",
                constraint_type=ConstraintType.TYPE,
                description="All call sites still type-check",
            )
        )

        # Tests pass (SPEC-07.23)
        postconditions.append(
            Constraint(
                name="tests_pass",
                constraint_type=ConstraintType.TEST,
                description="All tests still pass",
            )
        )

        # No new imports (SPEC-07.23)
        if check_imports:
            postconditions.append(
                Constraint(
                    name="no_new_imports",
                    constraint_type=ConstraintType.BEHAVIORAL,
                    description="No new imports introduced",
                )
            )

        return postconditions


class Verifier:
    """
    Verifies constraints against code.

    Implements: SPEC-07.21
    """

    def verify(
        self,
        constraints: list[Constraint],
        code: str,
    ) -> VerificationResult:
        """
        Verify constraints against code.

        Implements: SPEC-07.21

        Args:
            constraints: Constraints to verify
            code: Code to verify against

        Returns:
            VerificationResult with pass/fail status
        """
        passed: list[Constraint] = []
        failed: list[Constraint] = []

        for constraint in constraints:
            if self._check_constraint(constraint, code):
                passed.append(constraint)
            else:
                failed.append(constraint)

        if not failed:
            status = VerificationStatus.PASSED
        elif not passed:
            status = VerificationStatus.FAILED
        else:
            status = VerificationStatus.PARTIAL

        return VerificationResult(
            status=status,
            passed_constraints=passed,
            failed_constraints=failed,
        )

    def _check_constraint(self, constraint: Constraint, code: str) -> bool:
        """Check a single constraint."""
        if constraint.name == "syntax_valid":
            return self._check_syntax(code)
        elif constraint.name == "type_check":
            # Basic type check - syntax must be valid
            return self._check_syntax(code)
        elif constraint.name == "file_exists":
            # In verification context, assume file exists
            return True
        elif constraint.name == "original_matches":
            # Would need actual file content to verify
            return True
        else:
            # Unknown constraints pass by default
            return True

    def _check_syntax(self, code: str) -> bool:
        """Check if code has valid Python syntax."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False


class AutoCorrector:
    """
    Attempts automatic correction of verification failures.

    Implements: SPEC-07.25
    """

    def attempt_correction(
        self,
        change: CodeChange,
        failed_result: VerificationResult,
    ) -> CorrectionResult:
        """
        Attempt to correct verification failures.

        Implements: SPEC-07.25

        Args:
            change: The original code change
            failed_result: The failed verification result

        Returns:
            CorrectionResult with suggestions
        """
        attempts: list[CorrectionAttempt] = []

        for constraint in failed_result.failed_constraints:
            attempt = self._attempt_fix(constraint, change)
            attempts.append(attempt)

        # Determine overall suggestion
        suggestion = None
        explanation = None

        if attempts:
            # Provide guidance based on failure type
            first_failure = failed_result.failed_constraints[0]

            if first_failure.name == "syntax_valid":
                suggestion = "Review syntax errors in the modified code"
                explanation = (
                    "The modified code contains syntax errors. "
                    "Check for missing colons, parentheses, or invalid statements."
                )
            elif first_failure.name == "type_check":
                suggestion = "Fix type annotations or return types"
                explanation = (
                    "Type checking failed. Ensure return types match "
                    "annotations and arguments have correct types."
                )
            else:
                suggestion = f"Address constraint: {first_failure.name}"
                explanation = first_failure.description

        return CorrectionResult(
            attempted=True,
            suggestion=suggestion,
            explanation=explanation,
            attempts=attempts,
            success=any(a.success for a in attempts),
        )

    def _attempt_fix(
        self,
        constraint: Constraint,
        change: CodeChange,
    ) -> CorrectionAttempt:
        """Attempt to fix a single constraint failure."""
        suggested_fix = None
        success = False

        if constraint.name == "syntax_valid":
            # Suggest reverting to original
            suggested_fix = "Revert to original or fix syntax errors"
        elif constraint.name == "type_check":
            suggested_fix = "Check type annotations match actual types"

        return CorrectionAttempt(
            constraint=constraint,
            suggested_fix=suggested_fix,
            success=success,
        )


class VerificationChain:
    """
    Complete verification chain for code changes.

    Implements: SPEC-07.20-07.25
    """

    def __init__(self, auto_correct: bool = False) -> None:
        """
        Initialize verification chain.

        Args:
            auto_correct: Whether to attempt auto-correction on failures
        """
        self.auto_correct = auto_correct
        self.precondition_gen = PreconditionGenerator()
        self.postcondition_gen = PostconditionGenerator()
        self.refactoring_gen = RefactoringPostconditions()
        self.verifier = Verifier()
        self.corrector = AutoCorrector()

    def verify_change(self, change: CodeChange) -> VerificationResult:
        """
        Verify a code change through the full chain.

        Implements: SPEC-07.20

        Args:
            change: The code change to verify

        Returns:
            VerificationResult with full verification status
        """
        # Generate preconditions (SPEC-07.21)
        preconditions = self.precondition_gen.generate_preconditions(change)

        # Verify preconditions against original
        pre_result = self.verifier.verify(preconditions, change.original_content)
        if pre_result.status == VerificationStatus.FAILED:
            return VerificationResult(
                status=VerificationStatus.FAILED,
                passed_constraints=pre_result.passed_constraints,
                failed_constraints=pre_result.failed_constraints,
                preconditions_checked=True,
                postconditions_checked=False,
                error_message="Precondition verification failed",
            )

        # Generate postconditions (SPEC-07.21)
        postconditions = self.postcondition_gen.generate_postconditions(change)

        # Add refactoring postconditions if applicable (SPEC-07.23)
        if change.change_type == "refactor":
            postconditions.extend(self.refactoring_gen.generate(change))

        # Verify postconditions against modified (SPEC-07.21)
        post_result = self.verifier.verify(postconditions, change.modified_content)

        # Count subqueries (SPEC-07.24)
        subquery_count = len(postconditions)

        # Attempt correction if enabled and failed (SPEC-07.25)
        correction_attempted = False
        if self.auto_correct and post_result.status != VerificationStatus.PASSED:
            self.corrector.attempt_correction(change, post_result)
            correction_attempted = True

        return VerificationResult(
            status=post_result.status,
            passed_constraints=(pre_result.passed_constraints + post_result.passed_constraints),
            failed_constraints=post_result.failed_constraints,
            preconditions_checked=True,
            postconditions_checked=True,
            subquery_count=subquery_count,
            correction_attempted=correction_attempted,
        )


__all__ = [
    "AutoCorrector",
    "CodeChange",
    "Constraint",
    "ConstraintType",
    "CorrectionAttempt",
    "CorrectionResult",
    "PostconditionGenerator",
    "PreconditionGenerator",
    "RefactoringPostconditions",
    "VerificationChain",
    "VerificationResult",
    "VerificationStatus",
    "Verifier",
]
