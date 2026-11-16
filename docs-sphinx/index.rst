.. meta::
   :description: Uproot reads and writes ROOT files in pure Python and NumPy,
      independent of the C++ ROOT toolkit.

.. toctree::
    :hidden:

    changelog

.. toctree::
    :caption: Tutorials
    :hidden:

    basic
    uproot3-to-4

.. include:: main.toctree

.. toctree::
    :caption: Modules
    :hidden:

.. include:: uproot.toctree

.. include:: uproot.reading.toctree

.. include:: uproot.writing.toctree

.. include:: uproot.behaviors.toctree

.. include:: uproot.behavior.toctree

.. include:: uproot.model.toctree

.. include:: uproot.streamers.toctree

.. include:: uproot.cache.toctree

.. include:: uproot.compression.toctree

.. include:: uproot.deserialization.toctree

.. include:: uproot.serialization.toctree

.. include:: uproot.pyroot.toctree

.. include:: uproot.source.toctree

.. include:: uproot.sink.toctree

.. include:: uproot.interpretation.toctree

.. include:: uproot.containers.toctree

.. include:: uproot.language.toctree

.. include:: uproot.models.toctree

.. include:: uproot.exceptions.toctree

.. |br| raw:: html

    <br/>

.. image:: https://github.com/scikit-hep/uproot/raw/main/docs-img/logo/logo.svg
    :width: 300px
    :alt: Uproot
    :target: https://github.com/scikit-hep/uproot

.. role:: raw-html(raw)
    :format: html

|br|

:raw-html:`<p>`

.. image:: https://badge.fury.io/py/uproot.svg
   :alt: PyPI version
   :target: https://pypi.org/project/uproot

.. image:: https://img.shields.io/conda/vn/conda-forge/uproot
   :alt: Conda-Forge
   :target: https://github.com/conda-forge/uproot-feedstock

.. image:: https://img.shields.io/badge/python-3.7%E2%80%923.11-blue
   :alt: Python 3.7‒3.11
   :target: https://www.python.org

.. image:: https://img.shields.io/badge/license-BSD%203--Clause-blue.svg
   :alt: BSD-3 Clause License
   :target: https://opensource.org/licenses/BSD-3-Clause

.. image:: https://github.com/scikit-hep/uproot/actions/workflows/build-test.yml/badge.svg
   :alt: Continuous integration tests
   :target: https://github.com/scikit-hep/uproot/actions

:raw-html:`</p><p>`

.. image:: https://scikit-hep.org/assets/images/Scikit--HEP-Project-blue.svg
   :alt: Scikit-HEP
   :target: https://scikit-hep.org/

.. image:: https://img.shields.io/badge/NSF-1836650-blue.svg
   :alt: NSF-1836650
   :target: https://nsf.gov/awardsearch/showAward?AWD_ID=1836650

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4340632.svg
   :alt: DOI 10.5281/zenodo.4340632
   :target: https://doi.org/10.5281/zenodo.4340632

.. image:: https://img.shields.io/badge/docs-online-success
   :alt: Documentation
   :target: https://uproot.readthedocs.io/

:raw-html:`</p>`

|br| Uproot is a library for reading and writing ROOT files in pure Python and NumPy.

Unlike the standard C++ ROOT implementation, Uproot is only an I/O library, primarily intended to stream data into machine learning libraries in Python. Unlike PyROOT and root_numpy, Uproot does not depend on C++ ROOT. Instead, it uses Numpy to cast blocks of data from the ROOT file as Numpy arrays.

:raw-html:`<p align="center"><img src="https://raw.githubusercontent.com/scikit-hep/uproot/main/docs-img/diagrams/abstraction-layers.svg" width="700px"></p>`

How to install
==============

Uproot can be installed `from PyPI <https://pypi.org/project/uproot>`__ using pip. `Awkward Array <https://pypi.org/project/awkward>`__ is optional but highly recommended:

.. code-block:: bash

    pip install uproot awkward

Uproot is also available using `conda <https://anaconda.org/conda-forge/uproot>`__ (in this case, `Awkward Array <https://anaconda.org/conda-forge/awkward>`__ is automatically installed):

.. code-block:: bash

    conda install -c conda-forge uproot

If you have already added ``conda-forge`` as a channel, the ``-c conda-forge`` is unnecessary. Adding the channel is recommended because it ensures that all of your packages use compatible versions (see `conda-forge docs <https://conda-forge.org/docs/user/introduction.html#how-can-i-install-packages-from-conda-forge>`__):

.. code-block:: bash

    conda config --add channels conda-forge
    conda update --all

Documentation
=============

**ROOT** is a C++ toolkit for data analysis, part of which is the ROOT file format. Over an exabyte of particle physics data are stored in ROOT files around the world.

**Uproot** is a Python implementation of ROOT I/O, independent of the ROOT toolkit itself (including ROOT's Python interface, PyROOT).

- If you need help understanding ROOT and its ecosystem, see the `ROOT project documentation <https://root.cern/get_started/>`__.
- If you know what a ROOT file is but are unfamiliar with Uproot, see the :doc:`basic`.
- If you are migrating from an older version to Uproot 4 or 5, see the :doc:`uproot3-to-4`.
- If you need detailed descriptions of a class's properties or a function's parameters, see the left-bar on this site (≡ button on mobile) or use ``help`` in Python, ``?`` or shift-tab in iPython/Jupyter.

Getting help
============

- Report bugs, request features, and ask for additional documentation on `GitHub Issues <https://github.com/scikit-hep/uproot/issues>`__.
- If you have a "How do I...?" question, start a `GitHub Discussion <https://github.com/scikit-hep/uproot/discussions>`__ with category "Q&A".
- Alternatively, ask about it on `StackOverflow with the [uproot] tag <https://stackoverflow.com/questions/tagged/uproot>`__. Be sure to include tags for any other libraries that you use, such as Pandas or PyTorch.
