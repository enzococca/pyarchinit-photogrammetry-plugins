"""The client of the live sync of EMtools 1.6 against a host that speaks its wire."""
from __future__ import annotations

import pytest
from fake_host import FakeHost, free_port

from ppg_blender_em import sync


def test_the_envelope_of_wire_2():
    message = sync.envelope("select", {"node_id": "a", "source": "the edge's own field"})
    assert message == {"v": 2, "type": "select", "source": "pyarchinit-photogrammetry",
                       "payload": {"node_id": "a", "source": "the edge's own field"}}
    assert sync.read(message) == ("select", {"node_id": "a", "source": "the edge's own field"})
    with pytest.raises(sync.WireError):
        sync.read({"v": 1, "type": "select", "node_id": "a"})
    with pytest.raises(sync.WireError):
        sync.read({"v": 2, "payload": {}})


def test_connect_select_both_ways_and_connect_again():
    port = free_port()
    host = FakeHost(port)
    got, states = [], []
    client = sync.SyncClient(port=port, retry=0.1, on_message=lambda k, p: got.append((k, p)),
                             on_state=lambda s, t: states.append((s, t)),
                             info=lambda: {"graph_id": "Scavo", "graph_name": "Scavo"})
    client.start()
    try:
        assert sync.wait_for(lambda: host.kinds()[:2] == ["client_info", "request_snapshot"])
        info = host.received[0]
        assert info["v"] == 2 and info["source"] == "pyarchinit-photogrammetry"
        assert info["payload"] == {"tool": "pyarchinit-photogrammetry", "graph_id": "Scavo", "graph_name": "Scavo"}
        # the snapshot of a large matrix (a frame of more than 64 KiB) and a selection made in Blender
        nodes = [{"id": f"n{k}", "node_type": "US", "name": f"US{k}"} for k in range(3000)]
        host.send(sync.envelope("snapshot", {"doc": {"header": {"format": "em.json"},
                                                     "graph": {"graph_id": "Scavo", "nodes": nodes, "edges": []}}},
                                source="emtools"))
        host.send(sync.envelope("select", {"node_id": "n7", "node_ids": ["n7", "n8"]}, source="emtools"))
        assert sync.wait_for(lambda: [k for k, _ in got] == ["snapshot", "select"])
        assert len(got[0][1]["doc"]["graph"]["nodes"]) == 3000
        # the same selection does not go back to Blender (the echo of its own); another one does
        assert not client.select(["n7", "n8"], active="n7")
        assert client.select(["n2"])
        assert sync.wait_for(lambda: host.kinds()[-1:] == ["select"])
        assert host.received[-1]["payload"] == {"node_id": "n2"}
        # a message of another wire, and an echo of our own, are not delivered
        host.send({"v": 1, "type": "select", "node_id": "n1"})
        host.send(sync.envelope("select", {"node_id": "n9"}))
        host.send(sync.envelope("host_info", {"tool": "Blender · EMtools"}, source="emtools"))
        assert sync.wait_for(lambda: got[-1][0] == "host_info")
        assert [k for k, _ in got] == ["snapshot", "select", "host_info"]
        # Blender closed and opened again: the client connects again by itself
        host.close()
        assert sync.wait_for(lambda: not client.connected)
        host = FakeHost(port)
        assert sync.wait_for(lambda: host.kinds()[:2] == ["client_info", "request_snapshot"])
        assert [t for s, t in states if s == "connected"] == [f"127.0.0.1:{port}", "refused: wire v1, this client "
                                                               "speaks v2", f"127.0.0.1:{port}"]
    finally:
        client.stop()
        host.close()
    assert client.state == "stopped" and not client.connected


def test_no_host():
    states = []
    client = sync.SyncClient(port=free_port(), retry=0.05, on_state=lambda s, t: states.append((s, t)))
    client.start()
    try:
        assert sync.wait_for(lambda: any(s == "waiting" for s, _ in states))
        assert not client.select(["x"]) and client.last_selection == ()
    finally:
        client.stop()


def test_the_id_of_a_command_is_the_one_of_emtools():
    # computed with make_cmd_id of EMtools 1.6 (sync_manager/commands.py): the same command, the same id
    params = {"size": 1.5, "location": [1.0, 2.0, -0.5]}
    assert sync.make_cmd_id("create_proxy_for_unit", "abc-123", params) == "d0b79c5a-f501-59e6-a87b-ce380ed8c93a"
    assert sync.make_cmd_id("create_proxy_for_unit", "abc-123", dict(reversed(list(params.items())))) == \
        "d0b79c5a-f501-59e6-a87b-ce380ed8c93a"


def test_a_command_to_emtools():
    port = free_port()
    host = FakeHost(port)
    client = sync.SyncClient(port=port, retry=0.1)
    client.start()
    try:
        assert sync.wait_for(lambda: client.connected and host.kinds()[:1] == ["client_info"])
        cmd_id = client.command(sync.PROXY_VERB, "node-7", {"size": 2.0})
        assert sync.wait_for(lambda: host.kinds()[-1:] == ["command"])
        assert host.received[-1]["payload"] == {"cmd_id": cmd_id, "verb": "create_proxy_for_unit", "target": "node-7",
                                                "params": {"size": 2.0}}
    finally:
        client.stop()
        host.close()
    assert client.command(sync.PROXY_VERB, "node-7") is None  # not connected
