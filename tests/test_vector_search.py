from column_mapping.vector_search import pick_best_active_endpoint


class _FakeVSC:
    def __init__(self, endpoints):
        self._endpoints = endpoints

    def list_endpoints(self):
        return {"endpoints": self._endpoints}


def test_pick_best_active_endpoint_prefers_online_and_is_deterministic() -> None:
    vsc = _FakeVSC(
        endpoints=[
            {"name": "z_ep", "state": "ONLINE"},
            {"name": "a_ep", "state": "ONLINE"},
            {"name": "b_ep", "state": "PROVISIONING"},
        ]
    )
    assert pick_best_active_endpoint(vsc) == "a_ep"


def test_pick_best_active_endpoint_none_when_no_online() -> None:
    vsc = _FakeVSC(endpoints=[{"name": "x", "state": "PROVISIONING"}])
    assert pick_best_active_endpoint(vsc) is None

