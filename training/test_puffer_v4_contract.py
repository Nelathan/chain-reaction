import importlib.util


def main():
    if importlib.util.find_spec("pufferlib") is None:
        print("PufferLib v4 contract smoke skipped: pufferlib not installed.")
        return

    import pufferlib

    assert str(pufferlib.__version__).startswith("4"), pufferlib.__version__
    assert importlib.util.find_spec("pufferlib.emulation") is None
    print("PufferLib v4 contract smoke passed.")


if __name__ == "__main__":
    main()
