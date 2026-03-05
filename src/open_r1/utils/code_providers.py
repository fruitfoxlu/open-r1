# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Code execution providers for executing and evaluating code snippets."""

import abc
import asyncio
import concurrent.futures
import os
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import List, Optional

from ..utils import is_e2b_available, is_morph_available


if is_e2b_available():
    from e2b_code_interpreter import AsyncSandbox
    from e2b_code_interpreter.models import Execution

    from .routed_sandbox import RoutedSandbox
else:
    AsyncSandbox = None
    Execution = None
    RoutedSandbox = None

if is_morph_available():
    from morphcloud.api import MorphCloudClient
    from morphcloud.sandbox import Sandbox

    from .routed_morph import RoutedMorphSandbox
else:
    MorphCloudClient = None
    Sandbox = None
    RoutedMorphSandbox = None


class CodeExecutionProvider(abc.ABC):
    """Abstract base class for code execution providers."""

    @abc.abstractmethod
    def execute_scripts(self, scripts: List[str], languages: List[str]) -> List[float]:
        """Execute multiple scripts and return their reward values.

        Args:
            scripts: List of code scripts to execute
            language: The programming language of the scripts

        Returns:
            List of float rewards (one per script)
        """
        pass


class E2BProvider(CodeExecutionProvider):
    """Provider that executes code using E2B sandboxes."""

    def __init__(self, num_parallel: int = 2, e2b_router_url: Optional[str] = None):
        """Initialize the E2B provider.

        Args:
            num_parallel: Number of parallel sandboxes to use
            e2b_router_url: URL for the E2B router (if using router mode)
        """
        if not is_e2b_available():
            raise ImportError(
                "E2B is not available and required for this provider. Please install E2B with "
                "`pip install e2b-code-interpreter` and add an API key to a `.env` file."
            )

        self.num_parallel = num_parallel
        self.e2b_router_url = e2b_router_url

    def execute_scripts(self, scripts: List[str], languages: List[str]) -> List[float]:
        """Execute scripts using E2B sandboxes.

        If e2b_router_url is provided, uses the RoutedSandbox for batch processing.
        Otherwise, uses direct AsyncSandbox with parallelization.
        """
        if self.e2b_router_url is not None:
            routed_sandbox = RoutedSandbox(router_url=self.e2b_router_url)

            executions = routed_sandbox.run_code(
                scripts=scripts,
                languages=languages,
                timeout=30,
                request_timeout=28,
            )

            return [self._extract_reward(execution) for execution in executions]

        try:
            rewards = self._run_async_from_sync(scripts, languages, self.num_parallel)
        except Exception as e:
            print(f"Error from E2B executor: {e}")
            rewards = [0.0] * len(scripts)

        return rewards

    def _run_async_from_sync(self, scripts: List[str], languages: List[str], num_parallel: int) -> List[float]:
        """Function wrapping the `_run_async` function."""
        try:
            # Keep a reusable event loop for repeated reward calls during training.
            # `asyncio.run` creates/closes loops and can trigger "Event loop is closed"
            # issues in some client libraries across consecutive invocations.
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    raise RuntimeError("event loop is closed")
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            if loop.is_running():
                temp_loop = asyncio.new_event_loop()
                try:
                    rewards = temp_loop.run_until_complete(self._run_async(scripts, languages, num_parallel))
                finally:
                    temp_loop.close()
            else:
                rewards = loop.run_until_complete(self._run_async(scripts, languages, num_parallel))
        except Exception as e:
            print(f"Error from E2B executor async: {e}")
            raise e

        return rewards

    async def _run_async(self, scripts: List[str], languages: List[str], num_parallel: int) -> List[float]:
        semaphore = asyncio.Semaphore(num_parallel)

        tasks = [self._run_script(script, language, semaphore) for script, language in zip(scripts, languages)]

        results = await asyncio.gather(*tasks)
        rewards = list(results)

        return rewards

    async def _run_script(self, script: str, language: str, semaphore: asyncio.Semaphore) -> float:
        # We set a timeout margin, as the AsyncSandbox timeout does not seem to work
        # These values are based on running 256 examples with the gold solution
        # from open-r1/verifiable-coding-problems-python_decontaminated
        # see scripts/benchmark_e2b.py

        SANDBOX_TIMEOUT = 30
        MARGIN = 2
        REQUEST_TIMEOUT = SANDBOX_TIMEOUT - MARGIN
        ASYNCIO_TIMEOUT = SANDBOX_TIMEOUT + MARGIN

        sandbox = None
        async with semaphore:
            try:
                sandbox = await AsyncSandbox.create(timeout=SANDBOX_TIMEOUT, request_timeout=REQUEST_TIMEOUT)
                # e2b-code-interpreter changed from `languages=` (old) to `language=` (new).
                try:
                    execution = await asyncio.wait_for(
                        sandbox.run_code(script, language=language),
                        timeout=ASYNCIO_TIMEOUT,
                    )
                except TypeError:
                    execution = await asyncio.wait_for(
                        sandbox.run_code(script, languages=[language]),
                        timeout=ASYNCIO_TIMEOUT,
                    )
                return self._extract_reward(execution)
            except (TypeError, ValueError):
                return 0.0
            except asyncio.TimeoutError:
                print("Operation timed out")
                return 0.0
            except Exception as e:
                sandbox_id = getattr(sandbox, "sandbox_id", "unknown")
                print(f"Error in `_run_script` from E2B sandbox ID {sandbox_id} : {e}")
                return 0.0
            finally:
                if sandbox is not None:
                    try:
                        await sandbox.kill()
                    except Exception as e:
                        sandbox_id = getattr(sandbox, "sandbox_id", "unknown")
                        print(f"Error from E2B executor kill with sandbox ID {sandbox_id} : {e}")

    @staticmethod
    def _extract_reward(execution) -> float:
        """Extract float reward from different E2B execution result formats."""
        if execution is None:
            return 0.0

        # e2b v1 style
        text = getattr(execution, "text", None)
        if text not in (None, ""):
            try:
                return float(str(text).strip().splitlines()[-1])
            except (TypeError, ValueError):
                pass

        # e2b v2 style: results list
        results = getattr(execution, "results", None)
        if results:
            try:
                result_text = getattr(results[-1], "text", None)
                if result_text not in (None, ""):
                    return float(str(result_text).strip().splitlines()[-1])
            except (TypeError, ValueError, IndexError):
                pass

        # fallback: parse stdout logs
        logs = getattr(execution, "logs", None)
        stdout = getattr(logs, "stdout", None) if logs is not None else getattr(execution, "stdout", None)
        if stdout:
            try:
                output = "".join(stdout) if isinstance(stdout, list) else str(stdout)
                return float(output.strip().splitlines()[-1])
            except (TypeError, ValueError):
                pass

        return 0.0


class MorphProvider(CodeExecutionProvider):
    """Provider that executes code using MorphCloud's Sandbox API."""

    def __init__(self, num_parallel: int = 2, morph_router_url: Optional[str] = None):
        """Initialize the Morph provider.

        Args:
            num_parallel: Number of parallel executions to use
            morph_router_url: URL for the MorphCloud router (if using router mode)
        """
        if not is_morph_available():
            raise ImportError(
                "MorphCloud is not available and required for this provider. Please install MorphCloud with "
                "`pip install morphcloud` and add an API key to a `.env` file."
            )

        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            print("Warning: python-dotenv not installed. Environment variables must be set directly.")

        self.num_parallel = num_parallel
        self.morph_router_url = morph_router_url

        if self.morph_router_url is not None:
            self.routed_sandbox = RoutedMorphSandbox(router_url=self.morph_router_url)
            return

        import os

        self.api_key = os.getenv("MORPH_API_KEY")
        if not self.api_key:
            raise ValueError("MorphCloud API key not found. Please set the MORPH_API_KEY environment variable.")

        try:
            self.client = MorphCloudClient(api_key=self.api_key)
            self.Sandbox = Sandbox
        except ImportError as e:
            raise ImportError(f"Required MorphCloud dependencies not installed: {e}")

    def execute_scripts(self, scripts: List[str], languages: List[str]) -> List[float]:
        """Execute scripts using MorphCloud Sandbox API.

        Args:
            scripts: List of Python scripts to execute
            language: Programming language

        Returns:
            List of float rewards (one per script)
        """

        if hasattr(self, "routed_sandbox"):
            try:
                results = self.routed_sandbox.run_code(
                    scripts=scripts,
                    languages=languages,
                    timeout=90,
                    request_timeout=96,
                )

                rewards = []
                for result in results:
                    try:
                        reward = float(result.text)
                        rewards.append(reward)
                    except (ValueError, AttributeError):
                        rewards.append(0.0)
                return rewards
            except Exception as e:
                print(f"Error from MorphCloud router: {e}")
                return [0.0] * len(scripts)

        import asyncio

        try:
            rewards = asyncio.run(self._run_async(scripts, languages, self.num_parallel))
        except Exception as e:
            print(f"Error from MorphCloud executor: {e}")
            rewards = [0.0] * len(scripts)

        return rewards

    async def _run_async(self, scripts: List[str], languages: List[str], num_parallel: int) -> List[float]:
        """Run multiple scripts concurrently with limited parallelism.

        Args:
            scripts: List of scripts to execute
            language: Programming language
            num_parallel: Maximum number of concurrent executions

        Returns:
            List of rewards
        """

        semaphore = asyncio.Semaphore(num_parallel)

        tasks = [self._run_script(script, languages, semaphore) for script in scripts]

        results = await asyncio.gather(*tasks)

        return list(results)

    async def _run_script(self, script: str, languages: List[str], semaphore: asyncio.Semaphore) -> float:
        """Execute a single script in a MorphCloud Sandbox.

        Args:
            script: The script to execute
            language: Programming language
            semaphore: Semaphore to limit concurrency

        Returns:
            Float reward from script execution
        """
        SANDBOX_TIMEOUT = 90
        MARGIN = 6
        ASYNCIO_TIMEOUT = SANDBOX_TIMEOUT + MARGIN

        sandbox = None
        async with semaphore:
            try:
                sandbox = await asyncio.to_thread(self.Sandbox.new, client=self.client, ttl_seconds=SANDBOX_TIMEOUT)
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        sandbox.run_code,
                        script,
                        languages=languages,
                        timeout=SANDBOX_TIMEOUT,
                    ),
                    timeout=ASYNCIO_TIMEOUT,
                )

                reward = 0.0
                try:
                    if hasattr(result, "text") and result.text:
                        lines = result.text.strip().split("\n")
                        if lines:
                            try:
                                reward = float(lines[-1])
                            except ValueError:
                                try:
                                    reward = float(result.text.strip())
                                except ValueError:
                                    pass
                    elif hasattr(result, "stdout") and result.stdout:
                        lines = result.stdout.strip().split("\n")
                        if lines:
                            try:
                                reward = float(lines[-1])
                            except ValueError:
                                pass
                except (ValueError, AttributeError):
                    pass

                return reward

            except asyncio.TimeoutError:
                return 0.0
            except Exception:
                return 0.0
            finally:
                if sandbox:
                    try:
                        await asyncio.to_thread(sandbox.close)
                        await asyncio.to_thread(sandbox.shutdown)
                    except Exception:
                        pass


class LocalDockerProvider(CodeExecutionProvider):
    """Provider that executes code in hardened local Docker sandboxes."""

    def __init__(
        self,
        num_parallel: int = 2,
        docker_image: Optional[str] = None,
        timeout: int = 30,
        memory_limit: str = "512m",
        cpu_limit: str = "1.0",
        pids_limit: int = 64,
    ):
        self.num_parallel = max(1, num_parallel)
        self.docker_image = docker_image or os.getenv("LOCAL_CODE_DOCKER_IMAGE", "python:3.11-slim")
        self.timeout = int(os.getenv("LOCAL_CODE_TIMEOUT_SEC", str(timeout)))
        self.memory_limit = os.getenv("LOCAL_CODE_MEMORY_LIMIT", memory_limit)
        self.cpu_limit = os.getenv("LOCAL_CODE_CPU_LIMIT", cpu_limit)
        self.pids_limit = int(os.getenv("LOCAL_CODE_PIDS_LIMIT", str(pids_limit)))

    def execute_scripts(self, scripts: List[str], languages: List[str]) -> List[float]:
        if len(scripts) != len(languages):
            raise ValueError("scripts and languages must have the same length")

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_parallel) as executor:
            futures = [executor.submit(self._execute_script, script, language) for script, language in zip(scripts, languages)]
            return [future.result() for future in futures]

    def _execute_script(self, script: str, language: str) -> float:
        if language.lower() != "python":
            # The current code reward pipeline executes Python snippets.
            return 0.0

        script = self._prepare_script(script)

        with tempfile.TemporaryDirectory(prefix="open_r1_local_exec_") as tmpdir:
            os.chmod(tmpdir, 0o755)
            script_path = Path(tmpdir) / "script.py"
            script_path.write_text(script, encoding="utf-8")
            os.chmod(script_path, 0o644)

            cmd = self._build_docker_cmd(script_path)
            try:
                completed = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout + 5,
                    check=False,
                )
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                return 0.0

            if completed.returncode != 0:
                return 0.0

            return self._parse_reward(completed.stdout)

    def _build_docker_cmd(self, script_path: Path) -> List[str]:
        script_path = script_path.resolve()
        mount_src = str(script_path.parent)
        return [
            "docker",
            "run",
            "--rm",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--pids-limit",
            str(self.pids_limit),
            "--memory",
            self.memory_limit,
            "--cpus",
            str(self.cpu_limit),
            "--user",
            "65534:65534",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=64m",
            "--mount",
            f"type=bind,src={mount_src},dst=/workspace,readonly",
            "--workdir",
            "/tmp",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            self.docker_image,
            "python3",
            "-B",
            "/workspace/script.py",
        ]

    @staticmethod
    def _prepare_script(script: str) -> str:
        script = textwrap.dedent(script)
        marker = "evaluate_code(code_snippet, test_cases)"
        if marker in script and f"print({marker})" not in script:
            script = script.replace(marker, f"print({marker})")
        return script

    @staticmethod
    def _parse_reward(stdout: str) -> float:
        if not stdout:
            return 0.0
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        if not lines:
            return 0.0
        try:
            return float(lines[-1])
        except ValueError:
            return 0.0


def get_provider(provider_type: str = "e2b", **kwargs) -> CodeExecutionProvider:
    """Factory function to get the appropriate code execution provider.

    Args:
        provider_type: Type of provider to use ("e2b", "morph", "local")
        **kwargs: Additional arguments to pass to the provider

    Returns:
        An instance of CodeExecutionProvider
    """
    num_parallel = kwargs.pop("num_parallel", 2)

    if provider_type == "e2b":
        # Extract E2B-specific arguments
        e2b_router_url = kwargs.pop("e2b_router_url", None)
        return E2BProvider(
            num_parallel=num_parallel,
            e2b_router_url=e2b_router_url,
        )
    elif provider_type == "morph":
        # Extract Morph-specific arguments
        morph_router_url = kwargs.pop("morph_router_url", None)
        return MorphProvider(
            num_parallel=num_parallel,
            morph_router_url=morph_router_url,
        )
    elif provider_type == "local":
        return LocalDockerProvider(
            num_parallel=num_parallel,
            docker_image=kwargs.pop("local_docker_image", None),
            timeout=kwargs.pop("local_timeout", 30),
            memory_limit=kwargs.pop("local_memory_limit", "512m"),
            cpu_limit=kwargs.pop("local_cpu_limit", "1.0"),
            pids_limit=kwargs.pop("local_pids_limit", 64),
        )
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
