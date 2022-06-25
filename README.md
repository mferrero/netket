<div align="center">
<img src="https://www.netket.org/logo/logo_simple.jpg" alt="logo" width="400"></img>
</div>

# __NetKet__

[![Release](https://img.shields.io/github/release/netket/netket.svg)](https://github.com/netket/netket/releases)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/netket/badges/version.svg)](https://anaconda.org/conda-forge/netket)
[![Paper (v3)](https://img.shields.io/badge/paper%20%28v3%29-arXiv%3A2112.10526-B31B1B)](https://arxiv.org/abs/2112.10526)
[![codecov](https://codecov.io/gh/netket/netket/branch/master/graph/badge.svg?token=gzcOlpO5lB)](https://codecov.io/gh/netket/netket)
[![Slack](https://img.shields.io/badge/slack-chat-green.svg)](https://join.slack.com/t/mlquantum/shared_invite/zt-19wibmfdv-LLRI6i43wrLev6oQX0OfOw) 

NetKet is an open-source project delivering cutting-edge methods for the study
of many-body quantum systems with artificial neural networks and machine learning techniques.
It is a Python library built on [JAX](https://github.com/google/jax).

- **Homepage:** <https://www.netket.org>
- **Citing:** <https://www.netket.org/cite/>
- **Documentation:** <https://netket.readthedocs.io/en/latest/index.html>
- **Tutorials:** <https://netket.readthedocs.io/en/latest/tutorials/gs-ising.html>
- **Examples:** <https://github.com/netket/netket/tree/master/Examples>
- **Source code:** <https://github.com/netket/netket>

## Installation and Usage

NetKet runs on MacOS and Linux. We recommend to install NetKet using `pip`, but it can also be installed with `conda`.
It is often necessary to first update `pip` to a recent release (`>=20.3`) in order for upper compatibility bounds to be considered and avoid a broken installation.
For instructions on how to install the latest stable/beta release of NetKet see the [Get Started](https://www.netket.org/get_started/) page of our website or run the following command (Apple M1 users, follow that link for more instructions):

```sh
pip install --upgrade pip
pip install --upgrade netket
```

If you wish to install the current development version of NetKet, which is the master branch of this GitHub repository, together with the additional dependencies, you can run the following command:

```sh
pip install --upgrade pip
pip install 'git+https://github.com/netket/netket.git#egg=netket[all]'
```

To speed-up NetKet-computations, even on a single machine, you
can install the MPI-related dependencies by using `[mpi]` between square brackets.

```sh
pip install --upgrade pip
pip install --upgrade "netket[mpi]"
```

We recommend to install NetKet with all it's extra dependencies, which are documented below.
However, if you do not have a working MPI compiler in your PATH this installation will most likely fail because
it will attempt to install `mpi4py`, which enables MPI support in netket.

The latest release of NetKet is always available on PyPi and can be installed with `pip`.
NetKet is also available on conda-forge, however the version available through `conda install`
can be slightly out of date compared to PyPi.
To check what is the latest version released on both distributions you can inspect the badges at the top of this readme.

### Extra dependencies
When installing `netket` with pip, you can pass the following extra variants as square brakets. You can install several of them by separating them with a comma.
 - `"[dev]"`: installs development-related dependencies such as black, pytest and testing dependencies
 - `"[mpi]"`: Installs `mpi4py` to enable multi-process parallelism. Requires a working MPI compiler in your path
 - `"[extra]"`: Installs `tensorboardx` to enable logging to tensorboard, and openfermion to convert the QubitOperators.
 - `"[all]"`: Installs all extra dependencies

### MPI Support
To enable MPI support you must install [mpi4jax](https://github.com/PhilipVinc/mpi4jax). Please note that we advise to install mpi4jax  with the same tool (conda or pip) with which you install it's dependency `mpi4py`.

To check whether MPI support is enabled, check the flags
```python
>>> import netket
>>> netket.utils.mpi.available
True
```

### Installation on Windows
**WARNING:** Windows support is **experimental**, and you should expect suboptimal performance.

We suggest to use Windows Subsystem for Linux (WSL), on which you can install NetKet following the same instructions as above, and CUDA and MPI work as intended.

However, if you just want to quickly get started with NetKet, it is also possible to install it natively on Windows. First, download an unofficial `jaxlib` wheel from [cloudhan/jax-windows-builder](https://github.com/cloudhan/jax-windows-builder):
```sh
pip install --upgrade pip
pip install jaxlib -f https://whls.blob.core.windows.net/unstable/index.html
```
Alternatively, you may specify a wheel version with CUDA support.

Then install NetKet as usual:
```sh
pip install --upgrade netket
```

If you want MPI support, please follow the discussion in [mpi4jax](https://github.com/mpi4jax/mpi4jax/issues/24).

## Getting Started

To get started with NetKet, we recommend you give a look at our [tutorials page](https://netket.readthedocs.io/en/latest/tutorials/gs-ising.htmls), by running them on your computer or on [Google Colaboratory](https://colab.research.google.com).
There are also many example scripts that you can download, run and edit that showcase some use-cases of NetKet, although they are not commented.

If you want to get in touch with us, feel free to open an issue or a discussion here on GitHub, or to join the MLQuantum slack group where several people involved with NetKet hang out. To join the slack channel just accept [this invitation](https://join.slack.com/t/mlquantum/shared_invite/zt-19wibmfdv-LLRI6i43wrLev6oQX0OfOw)

## License

[Apache License 2.0](https://github.com/netket/netket/blob/master/LICENSE)

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/PhilipVinc"><img src="https://avatars.githubusercontent.com/u/2407108?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Filippo Vicentini</b></sub></a><br /><a href="https://github.com/netket/netket/commits?author=PhilipVinc" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/gcarleo"><img src="https://avatars.githubusercontent.com/u/28149892?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Giuseppe Carleo</b></sub></a><br /><a href="https://github.com/netket/netket/commits?author=gcarleo" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/femtobit"><img src="https://avatars.githubusercontent.com/u/4601206?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Damian Hofmann</b></sub></a><br /><a href="https://github.com/netket/netket/commits?author=femtobit" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/kchoo1118"><img src="https://avatars.githubusercontent.com/u/39584601?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Kenny Choo</b></sub></a><br /><a href="https://github.com/netket/netket/commits?author=kchoo1118" title="Code">💻</a></td>
    <td align="center"><a href="https://jamesetsmith.github.io/"><img src="https://avatars.githubusercontent.com/u/11277676?v=4?s=100" width="100px;" alt=""/><br /><sub><b>James E T Smith</b></sub></a><br /><a href="https://github.com/netket/netket/commits?author=jamesETsmith" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/chrisrothUT"><img src="https://avatars.githubusercontent.com/u/61942272?v=4?s=100" width="100px;" alt=""/><br /><sub><b>chrisrothUT</b></sub></a><br /><a href="https://github.com/netket/netket/commits?author=chrisrothUT" title="Code">💻</a></td>
    <td align="center"><a href="https://www.researchgate.net/profile/Vladimir_Vargas-Calderon"><img src="https://avatars.githubusercontent.com/u/31494271?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Vladimir Vargas-Calderón</b></sub></a><br /><a href="https://github.com/netket/netket/commits?author=VolodyaCO" title="Code">💻</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://evert.info/"><img src="https://avatars.githubusercontent.com/u/1933169?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Evert van Nieuwenburg</b></sub></a><br /><a href="https://github.com/netket/netket/commits?author=everthemore" title="Code">💻</a></td>
    <td align="center"><a href="https://shhslin.github.io/"><img src="https://avatars.githubusercontent.com/u/23406538?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Sheng-Hsuan Lin</b></sub></a><br /><a href="https://github.com/netket/netket/commits?author=ShHsLin" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/wdphy16"><img src="https://avatars.githubusercontent.com/u/43414703?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Dian Wu</b></sub></a><br /><a href="https://github.com/netket/netket/commits?author=wdphy16" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/GTorlai"><img src="https://avatars.githubusercontent.com/u/9124752?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Giacomo Torlai</b></sub></a><br /><a href="https://github.com/netket/netket/commits?author=GTorlai" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/nikita-astronaut"><img src="https://avatars.githubusercontent.com/u/5345808?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Nikita Astrakhantsev</b></sub></a><br /><a href="https://github.com/netket/netket/commits?author=nikita-astronaut" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/attila-i-szabo"><img src="https://avatars.githubusercontent.com/u/33730178?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Attila Szabó</b></sub></a><br /><a href="https://github.com/netket/netket/commits?author=attila-i-szabo" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/inailuig"><img src="https://avatars.githubusercontent.com/u/7287577?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Clemens Giuliani</b></sub></a><br /><a href="https://github.com/netket/netket/commits?author=inailuig" title="Code">💻</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/tvieijra"><img src="https://avatars.githubusercontent.com/u/31245481?v=4?s=100" width="100px;" alt=""/><br /><sub><b>tvieijra</b></sub></a><br /><a href="https://github.com/netket/netket/commits?author=tvieijra" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/fabienalet"><img src="https://avatars.githubusercontent.com/u/39381856?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Fabien Alet</b></sub></a><br /><a href="https://github.com/netket/netket/commits?author=fabienalet" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/stavros11"><img src="https://avatars.githubusercontent.com/u/35475381?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Stavros Efthymiou</b></sub></a><br /><a href="https://github.com/netket/netket/commits?author=stavros11" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/jwnys"><img src="https://avatars.githubusercontent.com/u/17109783?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Jannes Nys</b></sub></a><br /><a href="https://github.com/netket/netket/commits?author=jwnys" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/ChenAo-Phys"><img src="https://avatars.githubusercontent.com/u/64438142?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Chen Ao</b></sub></a><br /><a href="https://github.com/netket/netket/commits?author=ChenAo-Phys" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/emilyjd"><img src="https://avatars.githubusercontent.com/u/33588842?v=4?s=100" width="100px;" alt=""/><br /><sub><b>emilyjd</b></sub></a><br /><a href="https://github.com/netket/netket/commits?author=emilyjd" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/gpescia"><img src="https://avatars.githubusercontent.com/u/79276688?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Gabriel Pescia</b></sub></a><br /><a href="https://github.com/netket/netket/commits?author=gpescia" title="Code">💻</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/wuyukai"><img src="https://avatars.githubusercontent.com/u/5736909?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Yukai Wu</b></sub></a><br /><a href="https://github.com/netket/netket/commits?author=wuyukai" title="Code">💻</a></td>
    <td align="center"><a href="https://www.linkedin.com/in/erinaldi2"><img src="https://avatars.githubusercontent.com/u/5943556?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Enrico Rinaldi</b></sub></a><br /><a href="https://github.com/netket/netket/commits?author=erinaldi" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/nikosavola"><img src="https://avatars.githubusercontent.com/u/7860886?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Niko Savola</b></sub></a><br /><a href="https://github.com/netket/netket/commits?author=nikosavola" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/yannra"><img src="https://avatars.githubusercontent.com/u/43750364?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Yannic Rath</b></sub></a><br /><a href="https://github.com/netket/netket/commits?author=yannra" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/imi-hub"><img src="https://avatars.githubusercontent.com/u/80409000?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Imelda Romero</b></sub></a><br /><a href="https://github.com/netket/netket/commits?author=imi-hub" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/awietek"><img src="https://avatars.githubusercontent.com/u/5948847?v=4?s=100" width="100px;" alt=""/><br /><sub><b>awietek</b></sub></a><br /><a href="https://github.com/netket/netket/commits?author=awietek" title="Code">💻</a></td>
    <td align="center"><a href="http://amitkumarj441.github.io/"><img src="https://avatars.githubusercontent.com/u/14039450?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Amit Kumar Jaiswal</b></sub></a><br /><a href="https://github.com/netket/netket/commits?author=amitkumarj441" title="Code">💻</a></td>
  </tr>
  <tr>
    <td align="center"><a href="http://christian.mendl.net/"><img src="https://avatars.githubusercontent.com/u/9061478?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Christian B. Mendl</b></sub></a><br /><a href="https://github.com/netket/netket/commits?author=cmendl" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/evmckinney9"><img src="https://avatars.githubusercontent.com/u/47376937?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Evan McKinney</b></sub></a><br /><a href="https://github.com/netket/netket/commits?author=evmckinney9" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/martamau"><img src="https://avatars.githubusercontent.com/u/46056167?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Marta Mauri</b></sub></a><br /><a href="https://github.com/netket/netket/commits?author=martamau" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/maxbortone"><img src="https://avatars.githubusercontent.com/u/2243400?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Massimo Bortone</b></sub></a><br /><a href="https://github.com/netket/netket/commits?author=maxbortone" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/mmezic"><img src="https://avatars.githubusercontent.com/u/18701830?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Matĕj Mezera</b></sub></a><br /><a href="https://github.com/netket/netket/commits?author=mmezic" title="Code">💻</a></td>
    <td align="center"><a href="http://nicky.pro/"><img src="https://avatars.githubusercontent.com/u/52249105?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Nicky Pochinkov</b></sub></a><br /><a href="https://github.com/netket/netket/commits?author=pesvut" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/nkaeming"><img src="https://avatars.githubusercontent.com/u/12357446?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Niklas</b></sub></a><br /><a href="https://github.com/netket/netket/commits?author=nkaeming" title="Code">💻</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/noamwies"><img src="https://avatars.githubusercontent.com/u/3121971?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Noam Wies</b></sub></a><br /><a href="https://github.com/netket/netket/commits?author=noamwies" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/ooreilly"><img src="https://avatars.githubusercontent.com/u/13162518?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Ossian O'Reilly</b></sub></a><br /><a href="https://github.com/netket/netket/commits?author=ooreilly" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/theveniaut"><img src="https://avatars.githubusercontent.com/u/39381230?v=4?s=100" width="100px;" alt=""/><br /><sub><b>theveniaut</b></sub></a><br /><a href="https://github.com/netket/netket/commits?author=theveniaut" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/alexandercbooth"><img src="https://avatars.githubusercontent.com/u/15242567?v=4?s=100" width="100px;" alt=""/><br /><sub><b>alexandercbooth</b></sub></a><br /><a href="https://github.com/netket/netket/commits?author=alexandercbooth" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/artemborin"><img src="https://avatars.githubusercontent.com/u/39701087?v=4?s=100" width="100px;" alt=""/><br /><sub><b>artemborin</b></sub></a><br /><a href="https://github.com/netket/netket/commits?author=artemborin" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/orialb"><img src="https://avatars.githubusercontent.com/u/1208196?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Ori Alberton</b></sub></a><br /><a href="https://github.com/netket/netket/commits?author=orialb" title="Code">💻</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!