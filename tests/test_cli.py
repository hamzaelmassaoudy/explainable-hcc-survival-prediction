from hcc_survival.cli import build_parser


def test_cli_help_parses():
    parser = build_parser()
    args = parser.parse_args(["validate-data"])
    assert args.command == "validate-data"
