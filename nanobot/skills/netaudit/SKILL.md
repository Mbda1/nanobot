---
name: netaudit
description: "Audit the home network for exposure: open risky ports on the LAN, services on this machine, router UPnP, and what the internet sees on the public IP."
metadata: {"nanobot":{"emoji":"🛡️","requires":{"bins":["python3"]}}}
---

# Home network exposure audit

Use when the user asks "am I exposed?", "scan my network", "check my home network
security", or similar. Only scan networks the user owns.

## Run

```bash
python3 "$(python3 -c 'import nanobot,os;print(os.path.dirname(nanobot.__file__))')/skills/netaudit/netaudit.py" --external
```

Takes ~10-30s. Output is a short report; exit code 1 means high-risk findings.

Flags:
- `--subnet 192.168.1.0/24` — sweep a specific LAN (required when running inside
  Docker/WSL2; the script warns if the auto-detected subnet is 172.16-31.x)
- `--external` — look up the public IP on Shodan InternetDB (sends the public IP
  to api.ipify.org and internetdb.shodan.io). This is the real "exposed to the
  internet" check; drop it if the user does not want that.
- `--skip-lan` — local + UPnP + external only

If you don't know the user's LAN subnet and the script warns, ask them (on
Windows: `ipconfig`, look for the IPv4 address of the Wi-Fi/Ethernet adapter).

## Interpreting results

- `INTERNET-EXPOSED` lines are the urgent ones: a port open on the public IP.
  Fix by removing the router port-forward / disabling UPnP / turning off the service.
- `[high]` on the LAN (Telnet, SMB, RDP, VNC, Docker 2375, Redis, Mongo,
  Ollama 11434, cameras on 554) is risky if any LAN device is compromised or
  guests share the Wi-Fi. Recommend auth, firewalling, or a guest/IoT VLAN.
- Ollama on 11434 has no authentication — it should bind to 127.0.0.1 or be
  firewalled, never port-forwarded.
- UPnP enabled = any device can open router ports silently; recommend disabling.

Reply with: a one-line verdict (exposed / not exposed to internet), the high-risk
items, and concrete fixes. Don't paste the raw report unless asked.
