"""Build the public application route map from two Vast instance JSON exports."""

import argparse
import ipaddress
import json
from pathlib import Path


def route(instance: dict, internal_port: int, suffix: str = "") -> str:
    try:
        ip = str(ipaddress.ip_address(instance["public_ipaddr"]))
        mappings = instance["ports"][f"{internal_port}/tcp"]
        port = int(mappings[0]["HostPort"])
        if not 1 <= port <= 65535:
            raise ValueError("invalid mapped port")
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError(f"Missing or invalid Vast mapping for internal port {internal_port}") from error
    host = f"[{ip}]" if ":" in ip else ip
    return f"http://{host}:{port}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-json", type=Path, required=True)
    parser.add_argument("--aux-json", type=Path, required=True)
    parser.add_argument("--llm-port", type=int, default=8001)
    args = parser.parse_args()
    llm = json.loads(args.llm_json.read_text())
    aux = json.loads(args.aux_json.read_text())
    print(json.dumps({
        "llm": route(llm, args.llm_port, "/v1"),
        "embedding": route(aux, 8000),
        "rerank": route(aux, 8002),
    }, indent=2))


if __name__ == "__main__":
    main()
