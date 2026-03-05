import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from open_r1.rewards import code_reward
from open_r1.utils.code_providers import LocalDockerProvider, get_provider


class TestLocalDockerProvider(unittest.TestCase):
    def test_get_provider_local(self):
        provider = get_provider(provider_type="local", num_parallel=3)
        self.assertIsInstance(provider, LocalDockerProvider)
        self.assertEqual(provider.num_parallel, 3)

    def test_execute_scripts_parses_numeric_reward(self):
        provider = LocalDockerProvider(num_parallel=1)

        completed = subprocess.CompletedProcess(
            args=["docker", "run"],
            returncode=0,
            stdout="0.75\n",
            stderr="",
        )
        with patch("open_r1.utils.code_providers.subprocess.run", return_value=completed) as run_mock:
            rewards = provider.execute_scripts(["print('x')"], ["python"])

        self.assertEqual(rewards, [0.75])
        self.assertEqual(run_mock.call_count, 1)

    def test_execute_scripts_unsupported_language_is_zero(self):
        provider = LocalDockerProvider(num_parallel=1)
        with patch("open_r1.utils.code_providers.subprocess.run") as run_mock:
            rewards = provider.execute_scripts(["console.log('x')"], ["javascript"])
        self.assertEqual(rewards, [0.0])
        run_mock.assert_not_called()

    def test_hardened_docker_command(self):
        provider = LocalDockerProvider(num_parallel=1)
        cmd = provider._build_docker_cmd(Path("/tmp/openr1/script.py"))

        self.assertIn("--network=none", cmd)
        self.assertIn("--read-only", cmd)
        self.assertIn("--cap-drop=ALL", cmd)
        self.assertIn("--security-opt", cmd)
        self.assertIn("no-new-privileges:true", cmd)
        self.assertIn("--pids-limit", cmd)
        self.assertIn("--memory", cmd)
        self.assertIn("--cpus", cmd)
        self.assertIn("--user", cmd)
        self.assertIn("65534:65534", cmd)
        self.assertIn("--tmpfs", cmd)

        mount_idx = cmd.index("--mount")
        mount_value = cmd[mount_idx + 1]
        self.assertIn("type=bind", mount_value)
        self.assertIn("dst=/workspace", mount_value)
        self.assertIn("readonly", mount_value)

    def test_timeout_returns_zero(self):
        provider = LocalDockerProvider(num_parallel=1)
        with patch(
            "open_r1.utils.code_providers.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["docker"], timeout=35),
        ):
            rewards = provider.execute_scripts(["print('x')"], ["python"])
        self.assertEqual(rewards, [0.0])

    def test_code_reward_uses_local_provider(self):
        completions = [[{"role": "assistant", "content": "```python\nprint('ok')\n```"}]]
        verification_info = [{"language": "python", "test_cases": [{"input": "", "output": "ok"}]}]
        completed = subprocess.CompletedProcess(
            args=["docker", "run"],
            returncode=0,
            stdout="1.0\n",
            stderr="",
        )
        with patch("open_r1.utils.code_providers.subprocess.run", return_value=completed):
            rewards = code_reward(completions, provider_type="local", verification_info=verification_info, num_parallel=1)
        self.assertEqual(rewards, [1.0])


if __name__ == "__main__":
    unittest.main()
