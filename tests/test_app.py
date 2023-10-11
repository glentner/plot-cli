# SPDX-FileCopyrightText: 2022 Geoffrey Lentner
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for command-line interface."""


# external libs
from pytest import mark, CaptureFixture

# internal libs
from plot_cli import __version__, PlotApp


@mark.unit
@mark.parametrize('opt', ['-v', '--version'])
def test_version(capsys: CaptureFixture, opt: str) -> None:
    """Expects version to match."""
    PlotApp.main([opt, ])
    captured = capsys.readouterr()
    assert captured.out.strip() == __version__
    assert captured.err.strip() == ''
