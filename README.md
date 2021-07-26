# waver

[![License](https://img.shields.io/pypi/l/waver.svg?color=green)](https://github.com/sofroniewn/waver/raw/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/waver.svg?color=green)](https://pypi.org/project/waver)
[![Python Version](https://img.shields.io/pypi/pyversions/waver.svg?color=green)](https://python.org)
[![tests](https://github.com/sofroniewn/waver/workflows/tests/badge.svg)](https://github.com/sofroniewn/waver/actions)
[![codecov](https://codecov.io/gh/sofroniewn/waver/branch/master/graph/badge.svg)](https://codecov.io/gh/sofroniewn/waver)

Run simulations of the [wave equation](https://en.wikipedia.org/wiki/Wave_equation) in 1D, 2D, or 3D in Python. This library owes a lot of its design and approach to the [fdtd](https://github.com/flaport/fdtd) library, a Python 3D electromagnetic FDTD simulator.

Some of the examples use [napari](https://napari.org/), a multi-dimensional image viewer for Python, to allow for easy visualization of the detected wave. Some functionality is also available as a napari plugin to allow for running simulations
from a graphical user interface.

This package is still pre-alpha and under construction!!

----------------------------------

## Installation

You can install `waver` via [pip]:

    pip install waver

## Usage

TO BE ADDED ......

## Known Limitations

Right now boundary handling is not done very well. I'd like to add a [perfectly matched layer](https://en.wikipedia.org/wiki/Perfectly_matched_layer) boundary, but havn't done so yet. Contributions would be welcome.

Right now the simulations are quite slow. I'd like to add a [JAX](https://github.com/google/jax) backend, but 
havn't done so yet. Contributions would be welcome.

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"waver" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin
[file an issue]: https://github.com/sofroniewn/waver/issues
[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
