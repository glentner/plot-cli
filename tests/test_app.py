# SPDX-FileCopyrightText: 2022 Geoffrey Lentner
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for command-line interface."""


# external libs
from pytest import mark

# internal libs
from plot_cli import __version__, PlotApp


@mark.unit
def test_version_1(capsys) -> None:
    """Expects version to match."""
    PlotApp.main(['--version', ])
    captured = capsys.readouterr()
    assert captured.out.strip() == __version__
    assert captured.err.strip() == ''


@mark.unit
def test_version_2(capsys) -> None:
    """Expects version to match."""
    PlotApp.main(['-v', ])
    captured = capsys.readouterr()
    assert captured.out.strip() == __version__
    assert captured.err.strip() == ''
