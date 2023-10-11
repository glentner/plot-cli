Simple Command-line Plotting Tool
=================================

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
&nbsp;
[![Version: 0.3.0](https://img.shields.io/badge/Version-0.3.0-green.svg)](https://github.com/glentner/plot-cli)
&nbsp;
[![Python 3.10 | 3.11 | 3.12](https://img.shields.io/badge/Python-3.10%20%20|%203.11%20|%203.12-blue.svg)](https://www.python.org/downloads)

---

A simple command-line plotting tool. 

This project is merely a wrapper around [tplot](https://pypi.org/project/tplot/) and 
[pandas](https://pypi.org/project/pandas/) for the CSV input and data manipulation (e.g., resampling).
The project should be considered to be in _beta_ and subject to change. It works extremely well for 
well-behaved input, but can be rough around the edges when data is not clean. 


Install
-------

This project should not be confused for an older, abandoned project already on the 
package index by the same name. Install directly from GitHub:

```shell
pipx install git+https://github.com/glentner/plot-cli@v0.3.0
```


Example
-------

Using the basic line plot example from 
[seaborn](https://seaborn.pydata.org/examples/wide_data_lineplot.html):

![plot-cli example](https://github.com/glentner/plot-cli/assets/8965948/fa5179c8-93b5-427e-a562-a26f6599de39)