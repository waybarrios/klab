"""Normalize compute settings while retaining full display snapshots elsewhere."""


def execution_context(snapshot, *, scope="graph"):
    lines = snapshot.get("system76_power_profile", "").splitlines()
    profile = next((line.strip() for line in lines if line.startswith("Power Profile: ")), None)
    cpu = next((line.strip() for line in lines if line.startswith("CPU: ")), None)
    if scope not in ("graph", "eager"):
        raise ValueError("Expected graph or eager execution scope")
    if profile is None or cpu is None or not isinstance(snapshot.get("ac_online"), bool):
        raise ValueError(f"Missing usable power/CPU/AC observation: {snapshot!r}")
    return {"system76_power_profile": profile if scope == "graph" else profile + "\n" + cpu,
            "ac_online": snapshot["ac_online"], "scope": scope}
