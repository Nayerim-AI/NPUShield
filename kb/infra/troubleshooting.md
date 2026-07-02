# Infrastructure Troubleshooting

## Service won't start

```bash
docker logs <container_name> --tail 30
docker inspect <container_name> --format='{{.State.ExitCode}}'
```

Common causes: port conflict, missing volume mount, missing env var, OOM killed.

## Reverse proxy returns 502

```bash
curl -sS http://localhost:<backend_port>/health
journalctl -u traefik --since "5 min ago" --no-pager | grep -i error
```

Check: backend running, port correct, proxy config route matches.

## Disk full

```bash
df -h /
du -sh /var/log/* /var/lib/docker/* /tmp/* 2>/dev/null | sort -rh | head -10
```

Safe cleanup: `journalctl --vacuum-time=3d`, `docker system prune --filter "until=72h"`.

## High memory / OOM

```bash
free -h
ps aux --sort=-%mem | head -10
dmesg | grep -i "out of memory" | tail -5
```

For ARM: check zram, swap, cgroup limits.

## Tailscale node unreachable

```bash
tailscale status
tailscale ping <peer_hostname>
```

Check: both nodes online, ACLs allow traffic.

## NPU inference slow or failing

```bash
dmesg | grep -i rknpu
cat /sys/kernel/debug/rknpu/load 2>/dev/null
```

Check: rknpu driver loaded, no thermal throttling, CPU governor = `performance`.

```bash
sudo cpupower frequency-set -g performance
```

## NPUShield API not responding

```bash
curl http://localhost:18999/health
systemctl status npushield
journalctl -u npushield --since "10 min ago" --no-pager
```

Restart: `sudo systemctl restart npushield`

## Security reminders

- Never log or display API keys, passwords, private keys.
- Redact real IPs/hostnames in public responses.
- Use allowlisted commands only for tool execution.
- Always confirm before destructive actions.
