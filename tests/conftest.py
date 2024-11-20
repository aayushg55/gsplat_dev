def pytest_addoption(parser):
    parser.addoption(
        "--use_pytorch",
        action="store_true",
        default=False,
        help="Use PyTorch rasterization instead of default"
    )

