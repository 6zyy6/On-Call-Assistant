def _dependency_error(package_name: str) -> SystemExit:
    return SystemExit(
        f"Missing dependency: {package_name}. Install project dependencies with:\n"
        "python -m pip install -r requirements.txt"
    )


try:
    from oncall.app import app
except ModuleNotFoundError as exc:
    raise _dependency_error(exc.name) from exc


if __name__ == "__main__":
    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        raise _dependency_error(exc.name) from exc

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
