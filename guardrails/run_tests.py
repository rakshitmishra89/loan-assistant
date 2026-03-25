import json
from guardrails import moderate_input

# Load test cases
with open("tests.json", "r") as f:
    tests = json.load(f)

passed = 0

print("=" * 60)
print("RUNNING GUARDRAILS TESTS")
print("=" * 60)

for test in tests:
    result = moderate_input(test["input"])

    print("\n------------------------------")
    print("Test ID:", test["id"])
    print("Input:", test["input"])
    print("Expected:", test["expected"])
    print("Got:", result)

    if (
        result["allowed"] == test["expected"]["allowed"]
        and result["category"] == test["expected"]["category"]
        and result["agent_action"] == test["expected"]["agent_action"]
    ):
        print("✅ PASS")
        passed += 1
    else:
        print("❌ FAIL")

print("\n" + "=" * 60)
print(f"TOTAL PASSED: {passed}/{len(tests)}")
print("=" * 60)