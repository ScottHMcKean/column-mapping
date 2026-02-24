import pytest

from column_mapping.vector_search import pick_best_active_endpoint, validate_endpoint


class _FakeVSC:
    def __init__(self, endpoints):
        self._endpoints = endpoints

    def list_endpoints(self):
        return {"endpoints": self._endpoints}

    def get_endpoint(self, name):
        for ep in self._endpoints:
            if ep["name"] == name:
                return ep
        raise Exception(f"Endpoint {name} not found")


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


def test_validate_endpoint_returns_name_when_online() -> None:
    vsc = _FakeVSC(endpoints=[{"name": "my_ep", "state": "ONLINE"}])
    assert validate_endpoint(vsc=vsc, endpoint_name="my_ep") == "my_ep"


def test_validate_endpoint_errors_on_blank_name() -> None:
    vsc = _FakeVSC(endpoints=[])
    with pytest.raises(ValueError, match="endpoint name is blank"):
        validate_endpoint(vsc=vsc, endpoint_name="")


def test_validate_endpoint_errors_on_not_found() -> None:
    vsc = _FakeVSC(endpoints=[])
    with pytest.raises(ValueError, match="was not found"):
        validate_endpoint(vsc=vsc, endpoint_name="missing_ep")


def test_validate_endpoint_errors_on_not_online() -> None:
    vsc = _FakeVSC(endpoints=[{"name": "prov_ep", "state": "PROVISIONING"}])
    with pytest.raises(ValueError, match="not ONLINE"):
        validate_endpoint(vsc=vsc, endpoint_name="prov_ep")
