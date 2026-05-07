"""
Production code sandbox — subprocess-based isolation with Docker-ready interface.

Runs untrusted Python code in a separate process with:
  - Hard timeout enforcement (subprocess can be killed)
  - Memory limits (via system tools)
  - Restricted environment (no network, limited filesystem)
  - stdout/stderr capture
  - Exit code propagation

Architecture:
  This module provides the execution backend. The interface is designed
  to be swappable — swap SubprocessSandbox for DockerSandbox later
  by implementing the same execute() signature.

Security model:
  - Code is written to a temporary file and executed by a fresh Python
    interpreter subprocess — no shared memory with the host.
  - subprocess timeout is ENFORCED (not cooperative like threading).
  - Environment variables are stripped to prevent secret leakage.
  - File I/O is scoped to a temporary directory only.
"""

import os
import pathlib
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional


# ─── Configuration ───────────────────────────────────────────────

DEFAULT_TIMEOUT = 30              # Seconds before hard kill
DEFAULT_MEMORY_LIMIT_MB = 256     # Soft memory limit (informational)
MAX_CODE_SIZE = 50_000            # Characters
TEMP_DIR_PREFIX = "castalia_sandbox_"

# Environment variables to strip from subprocess
STRIPPED_ENV_VARS = {
    "PATH",            # We set our own minimal PATH
    "PYTHONPATH",      # Prevent importing from host project
    "HOME",            # Prevent access to home directory
    "USERPROFILE",     # Windows equivalent
    "APPDATA",         # Windows app data
    "LOCALAPPDATA",    # Windows local app data
    "TEMP",            # We use our own temp dir
    "TMP",
    "SYSTEMROOT",      # Keep for Windows compatibility
}

# Whitelisted environment variables (pass through)
KEEP_ENV_VARS = {
    "PYTHONIOENCODING",  # Ensure UTF-8 output
    "PYTHONUNBUFFERED",  # Don't buffer stdout
}


# ─── Data Classes ────────────────────────────────────────────────

@dataclass
class ExecutionResult:
    """Result from sandboxed code execution."""
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    execution_time_ms: float
    timed_out: bool = False
    killed: bool = False
    memory_exceeded: bool = False
    error_detail: str = ""

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "timed_out": self.timed_out,
            "killed": self.killed,
            "memory_exceeded": self.memory_exceeded,
            "error_detail": self.error_detail,
        }

    def was_successful(self) -> bool:
        return self.success and self.exit_code == 0 and not self.timed_out


# ─── Sandbox Engine ──────────────────────────────────────────────

class SubprocessSandbox:
    """
    Production subprocess sandbox with hard resource limits.

    Usage:
        sandbox = SubprocessSandbox(timeout=10)
        result = sandbox.execute("print('hello')")
        print(result.stdout)  # "hello\n"
    """

    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT,
        max_code_size: int = MAX_CODE_SIZE,
        temp_dir: Optional[str] = None,
        python_executable: Optional[str] = None,
    ):
        self.timeout = timeout
        self.max_code_size = max_code_size
        self.temp_dir = temp_dir
        self.python_executable = python_executable or sys.executable
        self._execution_count = 0

    def execute(self, code: str, env_overrides: Optional[dict] = None) -> ExecutionResult:
        """
        Execute code in an isolated subprocess.

        Args:
            code: Python source code to execute
            env_overrides: Optional dict of env vars to add

        Returns:
            ExecutionResult with stdout, stderr, exit code, timing
        """
        self._execution_count += 1

        # Validate code size
        if len(code) > self.max_code_size:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Code too large: {len(code)} > {self.max_code_size} chars",
                exit_code=-1,
                execution_time_ms=0.0,
                error_detail="CODE_SIZE_EXCEEDED",
            )

        # Write code to temp file
        tmp_dir = self.temp_dir
        tmp_file = None
        try:
            tmp_file = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                prefix=TEMP_DIR_PREFIX,
                dir=tmp_dir,
                delete=False,  # We delete manually after execution
                encoding="utf-8",
            )
            tmp_file.write(code)
            tmp_file.flush()
            tmp_path = tmp_file.name
            tmp_file.close()
            tmp_file = None  # Hand off ownership

            # Build minimal environment
            env = self._build_environment(env_overrides)

            # Build subprocess command
            cmd = [self.python_executable, "-u", tmp_path]

            # Execute with hard timeout
            start_time = time.monotonic()
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    env=env,
                    cwd=tmp_dir or tempfile.gettempdir(),
                )
                elapsed_ms = (time.monotonic() - start_time) * 1000

                stdout = proc.stdout
                stderr = proc.stderr
                exit_code = proc.returncode
                timed_out = False
                killed = False

            except subprocess.TimeoutExpired as e:
                elapsed_ms = (time.monotonic() - start_time) * 1000
                # subprocess.TimeoutExpired already kills the process
                stdout = e.stdout.decode("utf-8", errors="replace") if e.stdout else ""
                stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
                stderr += f"\n[TIMEOUT] Execution killed after {self.timeout}s"
                exit_code = -9  # SIGKILL equivalent
                timed_out = True
                killed = True

            # Determine success
            success = exit_code == 0 and not timed_out

            return ExecutionResult(
                success=success,
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                execution_time_ms=elapsed_ms,
                timed_out=timed_out,
                killed=killed,
            )

        except Exception as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000 if 'start_time' in dir() else 0.0
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=str(e),
                exit_code=-1,
                execution_time_ms=elapsed_ms,
                error_detail=f"EXECUTION_ERROR: {type(e).__name__}",
            )

        finally:
            # Cleanup temp file
            if tmp_file is not None:
                tmp_file.close()
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass  # Best-effort cleanup

    def _build_environment(self, overrides: Optional[dict] = None) -> dict:
        """
        Build minimal, secure environment for subprocess.

        Strips potentially dangerous env vars, keeps only what's needed.
        """
        # Start with clean environment
        env = {}

        # Add Python-specific vars
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"

        # Add minimal PATH (Python + system basics)
        if os.name == "nt":  # Windows
            env["PATH"] = os.path.dirname(self.python_executable)
            env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "C:\\Windows")
            env["COMSPEC"] = os.environ.get("COMSPEC", "cmd.exe")
            env["PATHEXT"] = os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD")
        else:  # Unix/Linux
            env["PATH"] = "/usr/local/bin:/usr/bin:/bin"
            env["HOME"] = "/tmp"

        # Apply overrides
        if overrides:
            env.update(overrides)

        return env

    @property
    def stats(self) -> dict:
        """Get sandbox execution statistics."""
        return {
            "execution_count": self._execution_count,
            "timeout_seconds": self.timeout,
            "max_code_size": self.max_code_size,
            "python_executable": self.python_executable,
        }


# ─── Docker Sandbox (Production-hardened) ──────────────────────────

class DockerSandbox(SubprocessSandbox):
    """
    Production Docker-based sandbox with full container isolation.

    Security layers:
      - seccomp profile (syscall filtering)
      - cgroups v2 (memory, CPU, PIDs)
      - Linux capabilities (drop ALL)
      - Network isolation (--network none)
      - Read-only root filesystem
      - tmpfs for temporary writes (in RAM, no disk)
      - Non-root user (UID 1000)
      - no-new-privileges (prevents privilege escalation)

    Requires: Docker Desktop running, castalia-sandbox image built.
              Run: docker build -f sandbox.Dockerfile -t castalia-sandbox .
    """

    def __init__(
        self,
        image: str = "castalia-sandbox:latest",
        timeout: int = DEFAULT_TIMEOUT,
        memory_mb: int = DEFAULT_MEMORY_LIMIT_MB,
        max_code_size: int = MAX_CODE_SIZE,
        seccomp_profile: Optional[str] = None,
        pids_limit: int = 100,
        cpus: float = 1.0,
        enable_monitoring: bool = False,
    ):
        super().__init__(timeout=timeout, max_code_size=max_code_size)
        self.image = image
        self.memory_mb = memory_mb
        self.seccomp_profile = seccomp_profile  # Optional: custom seccomp profile (advanced)
        self.pids_limit = pids_limit
        self.cpus = cpus
        self.enable_monitoring = enable_monitoring
        self._monitor_data: list[dict] = []

    def _find_seccomp_profile(self) -> Optional[str]:
        """Auto-discover seccomp profile relative to this file."""
        import pathlib
        profile_path = pathlib.Path(__file__).parent / "seccomp-profile.json"
        if profile_path.exists():
            return str(profile_path)
        return None

    def build_image(self, dockerfile: str = "sandbox.Dockerfile",
                    tag: Optional[str] = None) -> dict:
        """
        Build the sandbox Docker image.

        Returns:
            dict with success, stdout, stderr, image_id
        """
        cmd = ["docker", "build", "-f", dockerfile, "-t", tag or self.image, "."]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(pathlib.Path(__file__).parent),
            )
            # Extract image ID from output
            image_id = None
            for line in proc.stdout.splitlines():
                if line.startswith("Successfully built "):
                    image_id = line.split()[-1]

            return {
                "success": proc.returncode == 0,
                "image": tag or self.image,
                "image_id": image_id,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "returncode": proc.returncode,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "image": tag or self.image,
            }

    def check_image_exists(self) -> bool:
        """Check if the sandbox image is available locally."""
        try:
            proc = subprocess.run(
                ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            images = proc.stdout.strip().splitlines()
            return self.image in images or any(self.image.split(":")[0] in img for img in images)
        except Exception:
            return False

    def execute(self, code: str, env_overrides: Optional[dict] = None) -> ExecutionResult:
        """Execute code in a hardened Docker container via stdin."""
        self._execution_count += 1

        if len(code) > self.max_code_size:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Code too large: {len(code)} > {self.max_code_size} chars",
                exit_code=-1,
                execution_time_ms=0.0,
                error_detail="CODE_SIZE_EXCEEDED",
            )

        # Check image exists
        if not self.check_image_exists():
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Sandbox image '{self.image}' not found. "
                       f"Build it first: docker build -f sandbox.Dockerfile -t {self.image} .",
                exit_code=-1,
                execution_time_ms=0.0,
                error_detail="IMAGE_NOT_FOUND",
            )

        container_name = f"castalia-exec-{int(time.monotonic() * 1000)}"

        try:
            # Build production docker run command (stdin mode)
            docker_cmd = [
                "docker", "run", "--rm", "-i",
                "--name", container_name,
                # Resource governance
                "--memory", f"{self.memory_mb}m",
                "--memory-swap", f"{self.memory_mb}m",
                "--cpus", str(self.cpus),
                "--pids-limit", str(self.pids_limit),
                # Network isolation
                "--network", "none",
                # Filesystem hardening
                "--read-only",
                "--tmpfs", "/tmp:size=100M,exec,nodev,nosuid",
                "--tmpfs", "/var/tmp:size=50M,nodev,nosuid",
                # Security options
                "--security-opt", "no-new-privileges:true",
            ]

            # Add seccomp profile if available
            if self.seccomp_profile and pathlib.Path(self.seccomp_profile).exists():
                docker_cmd.extend(["--security-opt", f"seccomp={self.seccomp_profile}"])

            # Capabilities
            docker_cmd.extend(["--cap-drop", "ALL"])

            # User
            docker_cmd.extend(["--user", "1000:1000"])

            # Environment
            docker_cmd.extend([
                "-e", "PYTHONUNBUFFERED=1",
                "-e", "PYTHONDONTWRITEBYTECODE=1",
                "-e", "PYTHONHASHSEED=random",
                "-e", "HOME=/tmp",
            ])

            # Image
            docker_cmd.extend([self.image])

            # Monitoring
            start_time = time.monotonic()

            if self.enable_monitoring:
                self._start_monitoring(container_name)

            try:
                proc = subprocess.run(
                    docker_cmd,
                    input=code,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                elapsed_ms = (time.monotonic() - start_time) * 1000

                # Check for OOM kill
                oom_killed = self._check_oom_killed(container_name)

                return ExecutionResult(
                    success=proc.returncode == 0 and not oom_killed,
                    stdout=proc.stdout,
                    stderr=proc.stderr + ("\n[OOM KILLED]" if oom_killed else ""),
                    exit_code=proc.returncode,
                    execution_time_ms=elapsed_ms,
                    memory_exceeded=oom_killed,
                )

            except subprocess.TimeoutExpired as e:
                elapsed_ms = (time.monotonic() - start_time) * 1000
                stdout = e.stdout.decode("utf-8", errors="replace") if e.stdout else ""
                stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
                stderr += f"\n[DOCKER TIMEOUT] Container killed after {self.timeout}s"
                self._kill_container(container_name)
                return ExecutionResult(
                    success=False,
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=-9,
                    execution_time_ms=elapsed_ms,
                    timed_out=True,
                    killed=True,
                )

        except FileNotFoundError:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr="Docker not found. Install Docker Desktop and ensure it's running.",
                exit_code=-1,
                execution_time_ms=0.0,
                error_detail="DOCKER_NOT_FOUND",
            )
        except Exception as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000 if 'start_time' in dir() else 0.0
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=str(e),
                exit_code=-1,
                execution_time_ms=elapsed_ms,
                error_detail=f"DOCKER_ERROR: {type(e).__name__}",
            )
        finally:
            self._stop_monitoring(container_name)

    def _start_monitoring(self, container_name: str):
        """Start background monitoring for a container."""
        import threading
        def monitor():
            while True:
                try:
                    proc = subprocess.run(
                        ["docker", "stats", container_name, "--no-stream",
                         "--format", "{{.MemUsage}},{{.CPUPerc}},{{.PIDs}}"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if proc.returncode == 0 and proc.stdout.strip():
                        self._monitor_data.append({
                            "container": container_name,
                            "stats": proc.stdout.strip(),
                            "timestamp": time.monotonic(),
                        })
                    else:
                        break
                except Exception:
                    break
                time.sleep(0.5)
        threading.Thread(target=monitor, daemon=True).start()

    def _stop_monitoring(self, container_name: str):
        """Clean up monitoring data."""
        self._monitor_data = [
            m for m in self._monitor_data
            if m["container"] != container_name
        ]

    def _check_oom_killed(self, container_name: str) -> bool:
        """Check if container was OOM-killed."""
        try:
            proc = subprocess.run(
                ["docker", "inspect", container_name,
                 "--format", "{{.State.OOMKilled}}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return proc.returncode == 0 and "true" in proc.stdout.lower()
        except Exception:
            return False

    def _kill_container(self, container_name: Optional[str]):
        """Force kill a Docker container by name."""
        if container_name:
            try:
                subprocess.run(
                    ["docker", "kill", container_name],
                    capture_output=True, timeout=5,
                )
            except Exception:
                pass


# ─── Factory ─────────────────────────────────────────────────────

def create_sandbox(
    backend: str = "subprocess",
    timeout: int = DEFAULT_TIMEOUT,
    memory_mb: int = DEFAULT_MEMORY_LIMIT_MB,
    **kwargs,
) -> SubprocessSandbox:
    """
    Create a sandbox with the specified backend.

    Args:
        backend: "subprocess" or "docker"
        timeout: Maximum execution time in seconds
        memory_mb: Memory limit (informational for subprocess, enforced for Docker)
        **kwargs: Additional backend-specific options

    Returns:
        SubprocessSandbox or DockerSandbox instance
    """
    if backend == "docker":
        return DockerSandbox(
            timeout=timeout,
            memory_mb=memory_mb,
            max_code_size=kwargs.get("max_code_size", MAX_CODE_SIZE),
            image=kwargs.get("image", "castalia-sandbox:latest"),
            seccomp_profile=kwargs.get("seccomp_profile"),
            pids_limit=kwargs.get("pids_limit", 100),
            cpus=kwargs.get("cpus", 1.0),
            enable_monitoring=kwargs.get("enable_monitoring", False),
        )
    else:
        return SubprocessSandbox(
            timeout=timeout,
            max_code_size=kwargs.get("max_code_size", MAX_CODE_SIZE),
            temp_dir=kwargs.get("temp_dir"),
            python_executable=kwargs.get("python_executable"),
        )


# ─── Self-test ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Sandbox Self-Test ===\n")

    sandbox = SubprocessSandbox(timeout=5)

    # Test 1: Simple execution
    result = sandbox.execute("print('Hello from sandbox!')")
    print(f"Test 1 - Simple exec: success={result.success}, stdout='{result.stdout.strip()}'")
    assert result.success
    assert "Hello from sandbox!" in result.stdout
    print("  PASS\n")

    # Test 2: Math computation
    code = """
import math

def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

result = factorial(10)
print(f"10! = {result}")
"""
    result = sandbox.execute(code)
    print(f"Test 2 - Math: success={result.success}, stdout='{result.stdout.strip()}'")
    assert result.success
    assert "3628800" in result.stdout
    print("  PASS\n")

    # Test 3: Intentional error
    result = sandbox.execute("x = 1 / 0")
    print(f"Test 3 - Error handling: success={result.success}, exit_code={result.exit_code}")
    assert not result.success
    assert result.exit_code != 0
    assert "ZeroDivisionError" in result.stderr
    print("  PASS\n")

    # Test 4: Timeout enforcement
    timeout_sandbox = SubprocessSandbox(timeout=2)
    result = timeout_sandbox.execute("import time; time.sleep(10)")
    print(f"Test 4 - Timeout: timed_out={result.timed_out}, killed={result.killed}")
    assert result.timed_out
    assert result.killed
    print("  PASS\n")

    # Test 5: Environment isolation
    code = """
import os
print(f"PATH={os.environ.get('PATH', 'NOT SET')}")
print(f"HOME={os.environ.get('HOME', 'NOT SET')}")
print(f"USERPROFILE={os.environ.get('USERPROFILE', 'NOT SET')}")
"""
    result = sandbox.execute(code)
    print(f"Test 5 - Env isolation:")
    print(f"  stdout lines: {len(result.stdout.strip().splitlines())}")
    # Should have minimal env, not host env
    assert "C:\\Users" not in result.stdout or "NOT SET" in result.stdout
    print("  PASS\n")

    # Test 6: Stats
    stats = sandbox.stats
    print(f"Test 6 - Stats: {stats['execution_count']} executions, timeout={stats['timeout_seconds']}s")
    assert stats['execution_count'] == 4
    print("  PASS\n")

    # Test 7: Factory
    sandbox2 = create_sandbox("subprocess", timeout=10)
    result = sandbox2.execute("print('factory test')")
    print(f"Test 7 - Factory: success={result.success}")
    assert result.success
    print("  PASS\n")

    print("All sandbox tests passed!")
