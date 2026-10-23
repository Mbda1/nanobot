#!/usr/bin/env python3
"""Home network exposure audit (stdlib only, no root needed).

Checks, in order:
  1. local   - services on THIS machine listening on all interfaces
  2. lan     - TCP-connect sweep of the local /24 for risky open ports
  3. upnp    - whether the router answers UPnP/SSDP (can auto-open ports)
  4. external- (opt-in, --external) public IP + Shodan InternetDB lookup,
               i.e. what the internet already sees on your public IP

Only scan networks you own. Output is a compact text report, bounded in size.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import ipaddress
import json
import socket
import urllib.error
import urllib.request
from pathlib import Path

# port -> (label, risk). "high" = should never be reachable unauthenticated.
PORTS: dict[int, tuple[str, str]] = {
    21: ("FTP", "high"),
    22: ("SSH", "medium"),
    23: ("Telnet", "high"),
    53: ("DNS", "low"),
    80: ("HTTP", "low"),
    139: ("NetBIOS", "high"),
    443: ("HTTPS", "low"),
    445: ("SMB", "high"),
    554: ("RTSP camera", "high"),
    1883: ("MQTT", "high"),
    2375: ("Docker API (no TLS)", "high"),
    3389: ("RDP", "high"),
    5000: ("UPnP/NAS web", "medium"),
    5900: ("VNC", "high"),
    6379: ("Redis", "high"),
    8080: ("HTTP-alt", "medium"),
    8443: ("HTTPS-alt", "low"),
    9200: ("Elasticsearch", "high"),
    11434: ("Ollama API (no auth)", "high"),
    27017: ("MongoDB", "high"),
    32400: ("Plex", "medium"),
}

MAX_HOSTS = 256  # bound the sweep to one /24


def risk_of(port: int) -> str:
    return PORTS.get(port, ("?", "medium"))[1]


def label_of(port: int) -> str:
    return PORTS.get(port, (f"port {port}", "medium"))[0]


def local_ip() -> str:
    """Primary outbound interface IP (no packets are actually sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def default_subnet(ip: str) -> ipaddress.IPv4Network:
    return ipaddress.ip_network(f"{ip}/24", strict=False)


def subnet_warning(net: ipaddress.IPv4Network) -> str | None:
    """Docker / WSL2 NAT ranges mean we're not looking at the real home LAN."""
    if net.subnet_of(ipaddress.ip_network("172.16.0.0/12")):
        return (
            f"{net} looks like a Docker/WSL2 internal network, not your home LAN. "
            "Re-run with --subnet <your LAN, e.g. 192.168.1.0/24>."
        )
    if not net.is_private:
        return f"{net} is not a private range - refusing to sweep it."
    return None


def parse_proc_net(text: str, v6: bool = False) -> list[tuple[str, int]]:
    """Return (bind_addr, port) for LISTEN sockets in /proc/net/tcp{,6} text."""
    out = []
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 4 or parts[3] != "0A":  # 0A = TCP_LISTEN
            continue
        addr_hex, port_hex = parts[1].split(":")
        port = int(port_hex, 16)
        if v6:
            addr = "::" if int(addr_hex, 16) == 0 else "v6-specific"
        else:
            addr = socket.inet_ntoa(bytes.fromhex(addr_hex)[::-1])
        out.append((addr, port))
    return out


def check_local() -> list[str]:
    findings = []
    seen = set()
    for path, v6 in (("/proc/net/tcp", False), ("/proc/net/tcp6", True)):
        try:
            entries = parse_proc_net(Path(path).read_text(), v6)
        except OSError:
            continue
        for addr, port in entries:
            if addr in ("0.0.0.0", "::") and port not in seen:
                seen.add(port)
                findings.append(f"[{risk_of(port)}] {label_of(port)} listening on all interfaces (:{port})")
    if not seen and not Path("/proc/net/tcp").exists():
        findings.append("[info] /proc/net/tcp unavailable (not Linux) - use `netstat -an` instead")
    return sorted(findings, key=_risk_sort)


def probe(host: str, port: int, timeout: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def sweep(net: ipaddress.IPv4Network, ports: list[int], timeout: float, workers: int) -> dict[str, list[int]]:
    hosts = [str(h) for h in list(net.hosts())[:MAX_HOSTS]]
    found: dict[str, list[int]] = {}
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(probe, h, p, timeout): (h, p) for h in hosts for p in ports}
        for f in cf.as_completed(futs):
            if f.result():
                h, p = futs[f]
                found.setdefault(h, []).append(p)
    return {h: sorted(ps) for h, ps in sorted(found.items(), key=lambda kv: ipaddress.ip_address(kv[0]))}


def check_upnp(timeout: float = 2.0) -> list[str]:
    msg = (
        "M-SEARCH * HTTP/1.1\r\nHOST: 239.255.255.250:1900\r\n"
        'MAN: "ssdp:discover"\r\nMX: 1\r\n'
        "ST: urn:schemas-upnp-org:device:InternetGatewayDevice:1\r\n\r\n"
    ).encode()
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(timeout)
    responders = set()
    try:
        s.sendto(msg, ("239.255.255.250", 1900))
        while True:
            _, (addr, _) = s.recvfrom(2048)
            responders.add(addr)
    except OSError:
        pass
    finally:
        s.close()
    if not responders:
        return ["[ok] No UPnP gateway answered (or multicast blocked)"]
    return [
        f"[medium] UPnP IGD enabled on {a} - devices can open router ports without asking. "
        "Disable UPnP in the router unless you need it."
        for a in sorted(responders)
    ]


def check_external(timeout: float = 8.0) -> list[str]:
    try:
        with urllib.request.urlopen("https://api.ipify.org", timeout=timeout) as r:
            ip = r.read().decode().strip()
        ipaddress.ip_address(ip)
    except (OSError, ValueError) as e:
        return [f"[info] Could not determine public IP: {e}"]
    out = [f"[info] Public IP: {ip}"]
    try:
        with urllib.request.urlopen(f"https://internetdb.shodan.io/{ip}", timeout=timeout) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return out + ["[ok] Shodan has no record of open ports on your public IP"]
        return out + [f"[info] Shodan lookup failed: HTTP {e.code}"]
    except (OSError, ValueError) as e:
        return out + [f"[info] Shodan lookup failed: {e}"]
    ports = data.get("ports") or []
    if not ports:
        out.append("[ok] Shodan sees no open ports on your public IP")
    for p in ports:
        out.append(f"[{risk_of(p)}] INTERNET-EXPOSED: {label_of(p)} (:{p}) seen by Shodan")
    vulns = data.get("vulns") or []
    if vulns:
        out.append(f"[high] Shodan lists {len(vulns)} CVE(s): {', '.join(vulns[:10])}")
    return out


def _risk_sort(line: str) -> int:
    return {"[high]": 0, "[medium]": 1, "[low]": 2}.get(line.split(" ", 1)[0], 3)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subnet", help="LAN to sweep, e.g. 192.168.1.0/24 (default: auto /24)")
    ap.add_argument("--external", action="store_true", help="also look up public IP on Shodan InternetDB")
    ap.add_argument("--skip-lan", action="store_true", help="skip the LAN sweep")
    ap.add_argument("--timeout", type=float, default=0.4, help="per-port connect timeout (s)")
    ap.add_argument("--workers", type=int, default=128)
    args = ap.parse_args(argv)

    report: list[str] = []
    report.append("== Local listeners ==")
    report += check_local() or ["[ok] Nothing listening on all interfaces"]

    if not args.skip_lan:
        net = ipaddress.ip_network(args.subnet, strict=False) if args.subnet else default_subnet(local_ip())
        report.append(f"== LAN sweep {net} ==")
        warn = subnet_warning(net)
        if warn and not (args.subnet and net.is_private):
            report.append(f"[warn] {warn}")
        if net.is_private and (warn is None or args.subnet):
            if net.num_addresses > MAX_HOSTS + 2:
                report.append(f"[info] Only the first {MAX_HOSTS} hosts of {net} are swept")
            hosts = sweep(net, sorted(PORTS), args.timeout, args.workers)
            if not hosts:
                report.append("[info] No hosts answered on the checked ports")
            lines = []
            for h, ps in hosts.items():
                for p in ps:
                    lines.append(f"[{risk_of(p)}] {h} {label_of(p)} (:{p})")
            report += sorted(lines, key=_risk_sort)

    report.append("== Router UPnP ==")
    report += check_upnp()

    if args.external:
        report.append("== Internet exposure (public IP) ==")
        report += check_external()

    highs = sum(1 for line in report if line.startswith("[high]"))
    report.append(f"== Summary: {highs} high-risk finding(s) ==")
    print("\n".join(report))
    return 1 if highs else 0


if __name__ == "__main__":
    raise SystemExit(main())
