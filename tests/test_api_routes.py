from backend.app.app import app


def test_public_api_routes_excludes_hello_and_keeps_core_endpoints() -> None:
    public_paths = {route.path for route in app.routes if route.include_in_schema}

    assert "/hello" not in public_paths
    assert "/notes/upload" in public_paths
    assert "/notes/content" in public_paths
