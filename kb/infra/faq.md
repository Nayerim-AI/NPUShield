# Homelab Infra FAQ

This knowledge base is a public template. Replace placeholders with private deployment data in your own environment.

## Which server should host public-facing services?

Use a server with a real public IP address or a managed public gateway. Do not expose services from NAT-only machines unless they are routed through a trusted tunnel, reverse proxy, or VPN overlay.

## How do I check disk usage on a Linux server?

```bash
ssh user@server 'df -h /'
```

Use this for read-only status checks. Replace `user@server` with your actual SSH target.

## How do I check memory usage?

```bash
ssh user@server 'free -h'
```

For ARM boards with limited memory, also check zram and swap configuration.

## How do I restart a Docker service safely?

```bash
ssh user@server 'docker restart service-name'
```

Only restart services listed in the trusted service registry. For production systems, prefer health checks and rolling restart procedures.

## How do I inspect reverse proxy routing?

Check the gateway configuration and route table:

```bash
ssh user@gateway 'sudo grep -R "Host(" /etc/traefik /opt/*/traefik 2>/dev/null'
```

Do not print API tokens, private keys, or credentials from configuration files.

## Where should secrets live?

Secrets should live in a private secret store, environment file, or deployment-specific configuration outside the public KB. Never commit API tokens, passwords, SSH private keys, or real customer credentials.

## What should the assistant do when data is missing?

If the KB does not contain enough information, the assistant must say:

```text
I do not have that information in the knowledge base.
```

It must not invent private infrastructure details.
