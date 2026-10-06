import ipaddress
import socket
import sys
from pathlib import Path

# Pastikan output console/pipe selalu mendukung UTF-8 di Windows
for stream in (sys.stdout, sys.stderr):
    if stream and hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

import uvicorn
from app.core.config import settings


def get_network_urls(port: int) -> dict[str, list[str]]:
    urls: dict[str, list[str]] = {"local": [f"http://127.0.0.1:{port}"], "lan": [], "tailscale": []}
    try:
        hostname = socket.gethostname()
        for item in socket.getaddrinfo(hostname, None):
            if item[0] == socket.AF_INET:
                ip = item[4][0]
                if ip.startswith("127."):
                    continue
                ip_obj = ipaddress.IPv4Address(ip)
                url = f"http://{ip}:{port}"
                if ip_obj in ipaddress.IPv4Network("100.64.0.0/10"):
                    if url not in urls["tailscale"]:
                        urls["tailscale"].append(url)
                elif ip_obj.is_private:
                    if url not in urls["lan"]:
                        urls["lan"].append(url)
    except Exception:
        pass
    return urls


if __name__ == "__main__":
    net = get_network_urls(settings.PORT)
    print("=" * 60, flush=True)
    print(f"Manga Translator API Server (Port {settings.PORT})", flush=True)
    print(f"  * Localhost : {net['local'][0]}", flush=True)
    for lan_url in net["lan"]:
        print(f"  * LAN/Wi-Fi : {lan_url}", flush=True)
    for ts_url in net["tailscale"]:
        print(f"  * Tailscale : {ts_url}", flush=True)
    print(f"  * Swagger   : http://127.0.0.1:{settings.PORT}/docs", flush=True)
    print("=" * 60, flush=True)

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=1,
        reload=settings.RELOAD,
        reload_dirs=[str(Path(__file__).resolve().parents[1] / "app")] if settings.RELOAD else None,
    )
