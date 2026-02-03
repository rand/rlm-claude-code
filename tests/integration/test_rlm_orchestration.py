"""
Integration tests for RLM orchestration flow.

Tests the full integration of:
- Auto-activation based on complexity
- Intelligent orchestration decisions
- User preferences
- Strategy caching and learning
- Tool bridge access
"""

import tempfile
from pathlib import Path

from src.auto_activation import AutoActivator
from src.complexity_classifier import extract_complexity_signals, should_activate_rlm
from src.orchestration_schema import ExecutionMode, ToolAccessLevel
from src.strategy_cache import FeatureExtractor, StrategyCache
from src.tool_bridge import ToolBridge, ToolPermissions
from src.trajectory import TrajectoryEvent, TrajectoryEventType
from src.trajectory_analysis import StrategyType, TrajectoryAnalyzer
from src.types import Message, MessageRole, SessionContext, ToolOutput
from src.user_preferences import PreferencesManager, UserPreferences


class TestAutoActivationIntegration:
    """Integration tests for auto-activation flow."""

    def test_simple_query_skips_activation(self):
        """Simple queries skip RLM activation end-to-end."""
        context = SessionContext(
            messages=[],
            files={},
            tool_outputs=[],
        )

        # Check via auto-activator
        activator = AutoActivator()
        decision = activator.should_activate("ok", context)

        assert decision.should_activate is False

        # Also check via complexity classifier
        should_activate, reason = should_activate_rlm("ok", context)
        assert should_activate is False

    def test_complex_query_activates(self):
        """Complex queries activate RLM end-to-end."""
        context = SessionContext(
            messages=[],
            files={
                "src/auth/handler.py": "def authenticate(): pass",
                "src/api/routes.py": "def route(): pass",
                "src/db/models.py": "class User: pass",
            },
            tool_outputs=[
                ToolOutput(tool_name="Read", content="x" * 5000),
            ],
        )

        # Check complexity signals
        signals = extract_complexity_signals(
            "Why does auth.py fail when api.py calls the handler?",
            context,
        )
        assert signals.requires_cross_context_reasoning is True

        # Check auto-activation
        activator = AutoActivator()
        decision = activator.should_activate(
            "Why does auth.py fail when api.py calls the handler?",
            context,
        )

        assert decision.should_activate is True
        assert decision.confidence > 0.5

    def test_large_context_auto_activates(self):
        """Large contexts auto-activate regardless of query."""
        context = SessionContext(
            messages=[],
            files={"large.py": "x" * 600000},  # ~150k tokens
            tool_outputs=[],
        )

        activator = AutoActivator()
        decision = activator.should_activate("simple query", context)

        assert decision.should_activate is True
        assert "large_context" in decision.reason

    def test_preferences_override_activation(self):
        """User preferences can disable auto-activation."""
        prefs = UserPreferences(auto_activate=False)
        activator = AutoActivator(preferences=prefs)

        context = SessionContext(
            messages=[],
            files={"large.py": "x" * 600000},
            tool_outputs=[],
        )

        decision = activator.should_activate("complex query", context)

        assert decision.should_activate is False
        assert decision.reason == "auto_activate_disabled"


class TestOrchestrationPlanIntegration:
    """Integration tests for orchestration plan creation."""

    def test_activation_creates_valid_plan(self):
        """Activation decision creates valid orchestration plan."""
        activator = AutoActivator()
        context = SessionContext(
            messages=[],
            files={"large.py": "x" * 600000},
            tool_outputs=[],
        )

        decision = activator.should_activate("analyze this", context)
        plan = activator.create_plan_from_decision(decision)

        assert plan is not None
        assert plan.activate_rlm is True
        assert plan.depth_budget > 0
        assert plan.execution_mode in list(ExecutionMode)

    def test_preferences_reflected_in_plan(self):
        """User preferences are reflected in the plan."""
        prefs = UserPreferences(
            execution_mode=ExecutionMode.FAST,
            max_depth=1,
            tool_access=ToolAccessLevel.FULL,
            budget_dollars=10.0,
        )
        activator = AutoActivator(preferences=prefs)

        context = SessionContext(
            messages=[],
            files={"large.py": "x" * 600000},
            tool_outputs=[],
        )

        decision = activator.should_activate("query", context, force_rlm=True)
        plan = activator.create_plan_from_decision(decision)

        assert plan.execution_mode == ExecutionMode.FAST
        assert plan.depth_budget == 1
        assert plan.tool_access == ToolAccessLevel.FULL
        assert plan.max_cost_dollars == 10.0


class TestPreferencesManagerIntegration:
    """Integration tests for preferences management."""

    def test_command_flow(self):
        """Full command parsing and application flow."""
        manager = PreferencesManager()

        # Set mode
        msg, params = manager.parse_command("mode fast")
        assert params.get("execution_mode") == "fast"
        manager.prefs.execution_mode = ExecutionMode.FAST

        # Set budget
        msg, params = manager.parse_command("budget $5.00")
        assert params.get("budget_dollars") == 5.0
        manager.prefs.budget_dollars = 5.0

        # Verify preferences updated
        assert manager.prefs.execution_mode == ExecutionMode.FAST
        assert manager.prefs.budget_dollars == 5.0

    def test_preferences_persistence(self):
        """Preferences can be saved and loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prefs.json"

            # Save
            prefs1 = UserPreferences(
                execution_mode=ExecutionMode.THOROUGH,
                max_depth=3,
                budget_dollars=25.0,
            )
            prefs1.save(path)

            # Load
            prefs2 = UserPreferences.load(path)

            assert prefs2.execution_mode == ExecutionMode.THOROUGH
            assert prefs2.max_depth == 3
            assert prefs2.budget_dollars == 25.0


class TestToolBridgeIntegration:
    """Integration tests for tool bridge."""

    def test_read_only_permissions_enforced(self):
        """Read-only permissions are enforced end-to-end."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / "test.txt").write_text("content")

            perms = ToolPermissions.from_access_level(ToolAccessLevel.READ_ONLY)
            bridge = ToolBridge(permissions=perms, working_dir=tmpdir)

            # Read should work
            read_result = bridge.tool_call("read", "test.txt")
            assert read_result.success is True
            assert "content" in read_result.output

            # Bash should fail
            bash_result = bridge.tool_call("bash", "echo hello")
            assert bash_result.success is False
            assert "not permitted" in bash_result.error.lower()

    def test_full_permissions_allow_all(self):
        """Full permissions allow all operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / "test.txt").write_text("content")

            perms = ToolPermissions.from_access_level(ToolAccessLevel.FULL)
            bridge = ToolBridge(permissions=perms, working_dir=tmpdir)

            # Read should work
            read_result = bridge.tool_call("read", "test.txt")
            assert read_result.success is True

            # Bash should work
            bash_result = bridge.tool_call("bash", "echo hello")
            assert bash_result.success is True
            assert "hello" in bash_result.output

    def test_blocked_commands_denied(self):
        """Dangerous commands are blocked even with full permissions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            perms = ToolPermissions.from_access_level(ToolAccessLevel.FULL)
            bridge = ToolBridge(permissions=perms, working_dir=tmpdir)

            result = bridge.tool_call("bash", "rm -rf /")

            assert result.success is False
            assert "not allowed" in result.error.lower()


class TestStrategyLearningIntegration:
    """Integration tests for strategy learning."""

    def test_trajectory_analysis_to_cache_flow(self):
        """Trajectory analysis results can be cached."""
        # Create a trajectory with peeking strategy
        events = [
            TrajectoryEvent(
                type=TrajectoryEventType.REPL_EXEC,
                depth=0,
                content="peek(content[:100])",
            ),
            TrajectoryEvent(
                type=TrajectoryEventType.REPL_EXEC,
                depth=0,
                content="peek(files[:50])",
            ),
            TrajectoryEvent(
                type=TrajectoryEventType.FINAL,
                depth=0,
                content="The answer is 42",
            ),
        ]

        # Analyze
        analyzer = TrajectoryAnalyzer()
        analysis = analyzer.analyze(events)

        assert analysis.primary_strategy == StrategyType.PEEKING
        assert analysis.success is True

        # Cache
        cache = StrategyCache()
        result = cache.add("Read file.py content", analysis)
        assert result is True

        # Suggest for similar query
        suggestions = cache.suggest("Read data.py content")
        assert len(suggestions) >= 1
        assert suggestions[0].strategy == StrategyType.PEEKING

    def test_feature_extraction_consistency(self):
        """Feature extraction is consistent for similar queries."""
        extractor = FeatureExtractor()

        f1 = extractor.extract("Read the file content from main.py")
        f2 = extractor.extract("Read the file content from test.py")

        # Similar queries should have similar features
        assert f1.has_file_reference == f2.has_file_reference
        assert f1.is_question == f2.is_question

    def test_cache_persistence(self):
        """Strategy cache persists across sessions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.json"

            # Create and populate cache
            cache1 = StrategyCache(persistence_path=path)
            analysis = TrajectoryAnalyzer().analyze(
                [
                    TrajectoryEvent(
                        type=TrajectoryEventType.REPL_EXEC,
                        depth=0,
                        content="grep pattern in file",
                    ),
                    TrajectoryEvent(
                        type=TrajectoryEventType.FINAL,
                        depth=0,
                        content="Found matches",
                    ),
                ]
            )
            cache1.add("Search for pattern in code.py", analysis)
            cache1.save()

            # Load in new cache
            cache2 = StrategyCache(persistence_path=path)

            assert len(cache2._entries) == 1
            assert cache2._entries[0].strategy == StrategyType.GREPPING


class TestEndToEndFlow:
    """End-to-end integration tests."""

    def test_full_rlm_decision_flow(self):
        """Full flow from query to orchestration plan."""
        # Setup
        prefs = UserPreferences(
            execution_mode=ExecutionMode.BALANCED,
            max_depth=2,
        )
        activator = AutoActivator(preferences=prefs)

        # Complex context
        context = SessionContext(
            messages=[
                Message(role=MessageRole.USER, content="Help me debug"),
                Message(role=MessageRole.ASSISTANT, content="Looking at the code..."),
            ],
            files={
                "src/auth.py": "def auth(): raise Error",
                "src/api.py": "from auth import auth",
            },
            tool_outputs=[
                ToolOutput(tool_name="Bash", content="Error: Auth failed" * 1000),
            ],
        )

        # Make decision
        decision = activator.should_activate(
            "Why does auth fail when api calls it?",
            context,
        )

        # Create plan
        if decision.should_activate:
            plan = activator.create_plan_from_decision(decision)

            assert plan.activate_rlm is True
            assert plan.execution_mode == ExecutionMode.BALANCED
            assert plan.depth_budget == 2

    def test_strategy_suggestion_influences_approach(self):
        """Cached strategies can suggest approaches for new queries."""
        cache = StrategyCache()

        # Add successful peeking strategies
        for i in range(3):
            events = [
                TrajectoryEvent(
                    type=TrajectoryEventType.REPL_EXEC,
                    depth=0,
                    content=f"peek(file{i}[:100])",
                ),
                TrajectoryEvent(
                    type=TrajectoryEventType.FINAL,
                    depth=0,
                    content="Answer found",
                ),
            ]
            analysis = TrajectoryAnalyzer().analyze(events)
            cache.add(f"Read file{i}.py content quickly", analysis)

        # Query should get peeking suggestion
        suggestions = cache.suggest("Read data.py content quickly")

        assert len(suggestions) >= 1
        strategies = [s.strategy for s in suggestions]
        assert StrategyType.PEEKING in strategies

    def test_confused_assistant_triggers_activation(self):
        """Confusion in previous turn triggers activation."""
        context = SessionContext(
            messages=[
                Message(role=MessageRole.USER, content="Fix the bug"),
                Message(
                    role=MessageRole.ASSISTANT,
                    content="Actually, I was wrong about that. Let me reconsider...",
                ),
            ],
            files={},
            tool_outputs=[],
        )

        signals = extract_complexity_signals("Continue with the fix", context)
        assert signals.previous_turn_was_confused is True

        should_activate, reason = should_activate_rlm("Continue with the fix", context)
        assert should_activate is True
