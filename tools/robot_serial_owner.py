#!/usr/bin/env python3
"""Identify a reusable agent and refuse unknown or duplicate serial owners."""
import os
from pathlib import Path
import sys


def inspect(device, domain):
    device = os.path.realpath(device)
    agents, holders = [], set()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().decode().strip('\0').split('\0')
            if args and Path(args[0]).name in {'micro_ros_agent', 'MicroXRCEAgent'}:
                if '--dev' in args and os.path.realpath(args[args.index('--dev') + 1]) == device:
                    env = (proc / 'environ').read_bytes().split(b'\0')
                    if f'ROS_DOMAIN_ID={domain}'.encode() not in env:
                        raise RuntimeError(f'agent PID {proc.name} has a different/unknown ROS domain')
                    agents.append(proc.name)
            for fd in (proc / 'fd').iterdir():
                try:
                    if os.path.realpath(fd) == device:
                        holders.add(proc.name)
                except OSError:
                    pass
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            # fuser below also checks accessible holders; launcher never kills unknown owners.
            continue
    import subprocess
    probe = subprocess.run(['fuser', device], capture_output=True, text=True)
    if probe.returncode not in (0, 1):
        raise RuntimeError('cannot inspect UART ownership with fuser')
    holders.update(probe.stdout.split())
    if len(agents) > 1 or holders - set(agents):
        raise RuntimeError(f'conflicting UART owners: agents={agents}, holders={sorted(holders)}')
    return 'reuse:' + agents[0] if agents else 'new'


if __name__ == '__main__':
    try:
        print(inspect(sys.argv[1], sys.argv[2]))
    except Exception as exc:
        print(f'UART ownership check failed: {exc}', file=sys.stderr)
        raise SystemExit(1)
