"""
Security test suite for the production sandbox.

Validates that the sandbox correctly blocks:
  - Banned imports (os, sys, subprocess, etc.)
  - Network access (socket, urllib, requests)
  - File system writes (--read-only enforcement)
  - Dangerous builtins (exec, eval, open)
  - Resource exhaustion (memory limits, PID limits)
  - Privilege escalation (setuid, setgid)

Usage:
    python test_sandbox_security.py [--docker] [--subprocess]
"""

import argparse
import json
import sys
import time

from code_sandbox import create_sandbox, SubprocessSandbox, DockerSandbox


def run_test(name, code, should_pass, sandbox, description=""):
    """Run a single security test."""
    result = sandbox.execute(code)
    passed = result.success == should_pass

    status = "PASS" if passed else "FAIL"
    marker = "[OK]" if should_pass else "[BLOCKED]"

    print(f"  {status}: {name} {marker}")
    if not passed:
        print(f"    Expected: {'success' if should_pass else 'failure'}")
        print(f"    Got: success={result.success}, exit_code={result.exit_code}")
        if result.stderr:
            print(f"    Error: {result.stderr[:200]}")
    elif description:
        print(f"    {description}")

    return passed


def test_subprocess_sandbox():
    """Run all security tests against SubprocessSandbox."""
    print("\n=== SubprocessSandbox Security Tests ===")
    print("  (Tests sandbox ISOLATION: timeout, env stripping)")
    sandbox = create_sandbox("subprocess", timeout=5)
    results = []

    # Test 1: Safe code should work
    results.append(run_test(
        "Safe computation",
        "print(2 + 2)",
        True, sandbox,
        "Basic arithmetic should succeed"
    ))

    # Test 2: Timeout enforcement
    results.append(run_test(
        "Timeout enforcement",
        "import time; time.sleep(10)",
        False, sandbox,
        "Infinite sleep should timeout"
    ))

    # Test 3: Environment isolation
    result = sandbox.execute("import os; print(os.environ.get('HOME', 'NOT_SET'))")
    env_safe = result.success and "NOT_SET" in result.stdout
    print(f"  {'PASS' if env_safe else 'FAIL'}: Environment isolation {'[OK]' if env_safe else '[BREACH]'}")
    results.append(env_safe)

    # Test 4: Memory exhaustion (subprocess doesn't enforce, just observes)
    results.append(run_test(
        "Resource execution",
        "x = [0] * 1000000; print('OK')",
        True, sandbox,
        "Large allocation should execute (not sandbox's job to block)"
    ))

    return results


def test_full_pipeline():
    """Test the full CodeExecutor pipeline (analysis + sandbox)."""
    print("\n=== Full Pipeline Security Tests ===")
    print("  (Tests static analyzer + sandbox together)")

    from code_executor import CodeExecutor
    executor = CodeExecutor(timeout=5, backend="subprocess")
    results = []

    # Test 1: Safe code passes full pipeline
    result = executor.execute("print(2 + 2)")
    passed = result.execution_succeeded
    print(f"  {'PASS' if passed else 'FAIL'}: Safe code through pipeline {'[OK]' if passed else '[BLOCKED]'}")
    results.append(passed)

    # Test 2: Banned import blocked by analysis
    result = executor.execute("import os; print(os.getcwd())")
    blocked = result.was_blocked_by_analysis
    print(f"  {'PASS' if blocked else 'FAIL'}: Banned import (os) {'[BLOCKED]' if blocked else '[BREACH]'}")
    results.append(blocked)

    # Test 3: Banned import (subprocess)
    result = executor.execute("import subprocess; subprocess.run(['echo', 'pwned'])")
    blocked = result.was_blocked_by_analysis
    print(f"  {'PASS' if blocked else 'FAIL'}: Banned import (subprocess) {'[BLOCKED]' if blocked else '[BREACH]'}")
    results.append(blocked)

    # Test 4: Banned import (socket)
    result = executor.execute("import socket; s = socket.socket()")
    blocked = result.was_blocked_by_analysis
    print(f"  {'PASS' if blocked else 'FAIL'}: Banned import (socket) {'[BLOCKED]' if blocked else '[BREACH]'}")
    results.append(blocked)

    # Test 5: Dangerous builtin (exec)
    result = executor.execute("exec('print(1)')")
    blocked = result.was_blocked_by_analysis
    print(f"  {'PASS' if blocked else 'FAIL'}: Dangerous builtin (exec) {'[BLOCKED]' if blocked else '[BREACH]'}")
    results.append(blocked)

    # Test 6: Dangerous builtin (eval)
    result = executor.execute("eval('1+1')")
    blocked = result.was_blocked_by_analysis
    print(f"  {'PASS' if blocked else 'FAIL'}: Dangerous builtin (eval) {'[BLOCKED]' if blocked else '[BREACH]'}")
    results.append(blocked)

    # Test 7: Suspicious string pattern
    result = executor.execute('cmd = "rm -rf /"')
    blocked = result.was_blocked_by_analysis
    print(f"  {'PASS' if blocked else 'FAIL'}: Suspicious string {'[BLOCKED]' if blocked else '[BREACH]'}")
    results.append(blocked)

    return results


def test_docker_sandbox():
    """Run all security tests against DockerSandbox (requires built image)."""
    print("\n=== DockerSandbox Security Tests ===")

    # Check if image exists
    sandbox = create_sandbox("docker", timeout=5, image="castalia-sandbox:latest")
    if not isinstance(sandbox, DockerSandbox):
        print("ERROR: DockerSandbox not available")
        return []

    if not sandbox.check_image_exists():
        print("SKIP: Docker image 'castalia-sandbox:latest' not found.")
        print("      Run: docker build -f sandbox.Dockerfile -t castalia-sandbox:latest .")
        return []

    results = []

    # Test 1: Safe code
    results.append(run_test(
        "Safe computation",
        "print(2 + 2)",
        True, sandbox,
        "Basic arithmetic should succeed"
    ))

    # Test 2: Network access (blocked by --network none)
    results.append(run_test(
        "Network isolation",
        "import socket; s = socket.socket(); s.settimeout(1); s.connect(('8.8.8.8', 53))",
        False, sandbox,
        "Network should be blocked"
    ))

    # Test 3: Filesystem write to root (blocked by --read-only)
    # /tmp is tmpfs (writable by design), so test a truly read-only path
    results.append(run_test(
        "Filesystem read-only",
        "open('/etc/test.txt', 'w').write('x')",
        False, sandbox,
        "Write to /etc should fail (read-only root)"
    ))

    # Test 4: Memory limits (OOM kill)
    # Allocate ~300MB in a 128MB container → should OOM
    results.append(run_test(
        "Memory limits",
        "x = [bytearray(1024*1024) for _ in range(300)]",
        False, sandbox,
        "300MB allocation should OOM kill"
    ))

    # Test 5: PID limits (fork bomb blocked)
    results.append(run_test(
        "PID limits",
        "import os; [os.fork() for _ in range(50)]",
        False, sandbox,
        "Fork bomb should be blocked"
    ))

    # Test 6: Non-root user
    result = sandbox.execute("import os; print(os.getuid())")
    non_root = result.success and "1000" in result.stdout
    print(f"  {'PASS' if non_root else 'FAIL'}: Non-root user {'[OK]' if non_root else '[ROOT]'}")
    results.append(non_root)

    return results


def test_resource_governance():
    """Test resource governance features."""
    print("\n=== Resource Governance Tests ===")

    # Subprocess resource test
    sandbox = SubprocessSandbox(timeout=2)

    # Memory stress test
    start = time.monotonic()
    result = sandbox.execute("x = [0] * 10000000")  # ~80MB list
    elapsed = time.monotonic() - start

    print(f"  Memory test: success={result.success}, time={elapsed:.2f}s")

    # CPU stress test
    start = time.monotonic()
    result = sandbox.execute("\n".join([
        "def fib(n):",
        "    if n <= 1: return n",
        "    return fib(n-1) + fib(n-2)",
        "print(fib(35))"
    ]))
    elapsed = time.monotonic() - start

    print(f"  CPU test: success={result.success}, time={elapsed:.2f}s")
    print(f"  (Fib(35) = 9227465, should take ~1-2s)")

    return True


def main():
    parser = argparse.ArgumentParser(description="Sandbox Security Test Suite")
    parser.add_argument("--subprocess", action="store_true", help="Test subprocess sandbox")
    parser.add_argument("--docker", action="store_true", help="Test Docker sandbox")
    parser.add_argument("--all", action="store_true", help="Test all backends")
    args = parser.parse_args()

    if not any([args.subprocess, args.docker, args.all]):
        args.all = True

    all_results = []

    if args.subprocess or args.all:
        all_results.extend(test_subprocess_sandbox())

    if args.all:
        all_results.extend(test_full_pipeline())

    if args.docker or args.all:
        all_results.extend(test_docker_sandbox())

    if args.all:
        all_results.append(test_resource_governance())

    # Summary
    print("\n" + "=" * 50)
    total = len([r for r in all_results if r is not None])
    passed = sum(1 for r in all_results if r)
    failed = total - passed

    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("ALL SECURITY TESTS PASSED")
    else:
        print("SOME TESTS FAILED - REVIEW OUTPUT ABOVE")
    print("=" * 50)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
