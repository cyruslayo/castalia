
from blackboard_architecture import (
    Blackboard, BlackboardAgent, EventDrivenBlackboard, 
    ConflictResolvingBlackboard
)
import time

def test_basic_blackboard():
    print("\n--- Testing Basic Blackboard ---")
    bb = Blackboard("test")
    bb.set("topic", "AI Safety", "system")
    bb.set("research", "AI safety involves alignment and robustness.", "researcher")
    
    assert bb.get("topic") == "AI Safety"
    assert bb.get("research") == "AI safety involves alignment and robustness."
    assert "topic" in bb.keys()
    assert "research" in bb.keys()
    
    # Test versioning
    bb.set("topic", "Advanced AI Safety", "system")
    assert bb.get("topic") == "Advanced AI Safety"
    history = bb.get_history("topic")
    assert len(history) == 2
    assert history[0]["value"] == "AI Safety"
    assert history[1]["value"] == "Advanced AI Safety"
    print("[OK] Basic Blackboard passed")

def test_access_control():
    print("\n--- Testing Access Control ---")
    bb = Blackboard("secure")
    bb.set("secret", "top_secret_data", "admin")
    bb.set_access("secret", readers={"admin"}, writers={"admin"})
    
    # Unauthorized read
    assert bb.get("secret", reader="guest") is None
    # Authorized read
    assert bb.get("secret", reader="admin") == "top_secret_data"
    
    # Unauthorized write
    bb.set("secret", "hacked", "guest")
    assert bb.get("secret", reader="admin") == "top_secret_data"
    
    print("[OK] Access Control passed")

def test_event_driven_blackboard():
    print("\n--- Testing Event-Driven Blackboard ---")
    eboard = EventDrivenBlackboard("event_test")
    
    # Create a simple agent that just writes "Handled"
    class MockAgent(BlackboardAgent):
        def run(self, blackboard, task=""):
            blackboard.set(self.write_key, "Handled", self.name)
            return "Handled"

    agent = MockAgent("TriggerAgent", "handler", ["trigger"], "status")
    eboard.register_trigger("trigger", agent, "Respond to trigger")
    
    eboard.set("trigger", "Go", "system")
    
    assert eboard.get("status") == "Handled"
    assert len(eboard._activation_log) == 1
    print("[OK] Event-Driven Blackboard passed")

def test_conflict_resolution():
    print("\n--- Testing Conflict Resolution ---")
    
    # Test keep_all
    bb_all = ConflictResolvingBlackboard("keep_all", strategy="keep_all")
    bb_all.set("note", "Hello", "Agent1")
    bb_all.set("note", "World", "Agent2")
    
    result = bb_all.get("note")
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["author"] == "Agent1"
    assert result[1]["author"] == "Agent2"
    
    # Test last_write_wins
    bb_last = ConflictResolvingBlackboard("last", strategy="last_write_wins")
    bb_last.set("note", "Hello", "Agent1")
    bb_last.set("note", "World", "Agent2")
    assert bb_last.get("note") == "World"
    
    print("[OK] Conflict Resolution passed")

if __name__ == "__main__":
    try:
        test_basic_blackboard()
        test_access_control()
        test_event_driven_blackboard()
        test_conflict_resolution()
        print("\n[SUCCESS] ALL TESTS PASSED")
    except AssertionError as e:
        print(f"\n[FAILURE] TEST FAILED")
        raise e
