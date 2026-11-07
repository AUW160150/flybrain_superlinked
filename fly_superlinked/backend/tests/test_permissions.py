import asyncio

import pytest

from backend.memory.store import DRONES, POLICY, LocalBackend, SwarmMemory, can


def test_policy_table():
    for d in DRONES:
        assert can(d, "swarm_map", "read") and can(d, "outcomes", "read")
        assert not can(d, "outcomes", "write")
        assert can(d, f"private_{d}", "write") and can(d, f"private_{d}", "read")
        for other in DRONES:
            if other != d:
                assert not can(d, f"private_{other}", "read") and not can(d, f"private_{other}", "write")
    assert can("scout-1", "swarm_map", "write") and can("scout-2", "swarm_map", "write")
    assert not can("courier", "swarm_map", "write")
    assert can("operator", "outcomes", "write")


def _mem():
    m = SwarmMemory(LocalBackend())
    asyncio.run(m.setup())
    return m


@pytest.mark.parametrize("user,dataset", [("courier", "swarm_map"), ("scout-1", "outcomes"), ("scout-1", "private_courier"),
                                          ("courier", "private_scout-2"), ("operator", "swarm_map")])
def test_write_without_permission_is_denied(user, dataset):
    m = _mem()
    ok = asyncio.run(m.write(user, dataset, "x [fact kind=hazard hazard=gust x=1 y=2 r=3]"))
    assert ok is False
    assert m.events[-1].kind == "denied"
    assert m.backend.docs[dataset] == []


@pytest.mark.parametrize("user,dataset", [("scout-1", "swarm_map"), ("operator", "outcomes"), ("courier", "private_courier")])
def test_write_with_permission_is_stored(user, dataset):
    m = _mem()
    assert asyncio.run(m.write(user, dataset, "hello"))
    assert m.events[-1].kind == "write" and len(m.backend.docs[dataset]) == 1


def test_read_private_of_other_is_denied():
    m = _mem()
    asyncio.run(m.write("scout-1", "private_scout-1", "secret"))
    assert asyncio.run(m.recall("courier", "secret", ["private_scout-1"])) == []
    assert m.events[-1].kind == "denied"
