# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
Uproot: ROOT I/O in pure Python and NumPy.

Nearly all of the functions needed for general use are imported here, but the
documentation gives fully qualified names. For example, the most frequently
used function in Uproot is

.. code-block:: python

    uproot.open("path/to/filename.root")

but we refer to it in the documentation as :doc:`uproot.reading.open`.

Typical entry points for file-reading are

* :doc:`uproot.reading.open`
* :doc:`uproot.behaviors.TBranch.iterate`
* :doc:`uproot.behaviors.TBranch.concatenate`
* :doc:`uproot._dask.dask`

though they would usually be accessed as ``uproot.iterate``,
``uproot.concatenate``, and ``uproot.dask``.

The most useful classes are

* :doc:`uproot.behaviors.TBranch.HasBranches` (``TTree`` or ``TBranch``)
* :doc:`uproot.behaviors.TBranch.TBranch`
* :doc:`uproot.behaviors.TH1`
* :doc:`uproot.behaviors.TH2`
* :doc:`uproot.behaviors.TProfile`

though they would usually be accessed through instances that have been read
from files.

The submodules of Uproot are:

* :doc:`uproot.reading`: entry-point for reading files, as well as classes
  for the three basic types that can't be modeled: ``TFile``, ``TDirectory``,
  and ``TKey``.
* :doc:`uproot.behaviors`: methods and properties to mix into instantiated
  models, for a high-level user interface.
* :doc:`uproot.model`: utilities for modeling C++ objects as Python objects.
* :doc:`uproot.streamers`: models for ``TStreamerInfo`` and its elements
  to generate code for new models for classes in ROOT files.
* :doc:`uproot.cache`: defines caches with least-recently used eviction
  policies.
* :doc:`uproot.compression`: functions for compressing and decompressing data.
* :doc:`uproot.deserialization`: utility functions for deserialization,
  including the generation of new classes.
* :doc:`uproot.source`: the "physical layer," which reads bytes without
  interpreting them from various backends, like files, HTTP(S), and XRootD.
* :doc:`uproot.interpretation`: prescriptions for converting ROOT types
  into Pythonic arrays.
* :doc:`uproot.containers`: interpretations and models for standard
  containers, such as ``std::vector`` and arrays.
* :doc:`uproot.language`: computational backends for expressions in
  :ref:`uproot.behaviors.TBranch.HasBranches.arrays`.
* :doc:`uproot.models`: predefined models for classes that are too basic
  to rely on ``TStreamerInfo`` or too common to justify reading it.
* ``uproot.const``: integer constants used in ROOT serialization and
  deserialization.
* ``uproot.extras``: import functions for the libraries that Uproot can
  use, but does not require as dependencies. If a library can't be imported,
  these functions provide instructions for installing them.
* ``uproot.version``: for access to the version number.
* ``uproot.dynamic``: initially empty module, in which dynamically
  generated classes are defined.
* ``uproot._util``: non-public utilities used by the above.

isort:skip_file
"""

from uproot.version import __version__
import uproot.const
import uproot.extras
import uproot.dynamic

classes = {}
unknown_classes = {}

from uproot.cache import LRUCache
from uproot.cache import LRUArrayCache

from uproot.source.file import MemmapSource
from uproot.source.file import MultithreadedFileSource
from uproot.source.http import HTTPSource
from uproot.source.http import MultithreadedHTTPSource
from uproot.source.xrootd import XRootDSource
from uproot.source.xrootd import MultithreadedXRootDSource
from uproot.source.object import ObjectSource
from uproot.source.fsspec import FSSpecSource
from uproot.source.cursor import Cursor
from uproot.source.futures import TrivialExecutor
from uproot.source.futures import ThreadPoolExecutor
from uproot.deserialization import DeserializationError

from uproot.compression import ZLIB
from uproot.compression import LZMA
from uproot.compression import LZ4
from uproot.compression import ZSTD

from uproot.reading import open
from uproot.reading import ReadOnlyFile
from uproot.reading import ReadOnlyDirectory

from uproot.exceptions import CoalescedReadError, KeyInFileError

from uproot.model import Model
from uproot.model import classname_decode
from uproot.model import classname_encode
from uproot.model import has_class_named
from uproot.model import class_named
from uproot.model import reset_classes

from uproot.writing import create
from uproot.writing import recreate
from uproot.writing import update
from uproot.writing import WritableFile
from uproot.writing import WritableDirectory
from uproot.writing import WritableTree
from uproot.writing import WritableBranch
from uproot.writing import to_writable
from uproot.writing import dask_write
from uproot.writing import graphed_write
from uproot.writing import root_write

from uproot.writing.interpret import as_TGraph

import uproot.models.TObject
import uproot.models.TString
import uproot.models.TArray
import uproot.models.TNamed
import uproot.models.TList
import uproot.models.THashList
import uproot.models.TObjArray
import uproot.models.TClonesArray
import uproot.models.TObjString
import uproot.models.TAtt
import uproot.models.TDatime
import uproot.models.TRef

import uproot.models.TTable
import uproot.models.TTree
import uproot.models.TBranch
import uproot.models.TLeaf
import uproot.models.TBasket
import uproot.models.RNTuple
import uproot.models.TH
import uproot.models.TGraph
import uproot.models.TMatrixT
import uproot.models.TTime

from uproot.models.TTree import num_entries

from uproot.containers import STLVector
from uproot.containers import STLSet
from uproot.containers import STLMap

import uproot.interpretation
import uproot.interpretation.identify
import uproot.interpretation.library
from uproot.interpretation.numerical import AsDtype
from uproot.interpretation.numerical import AsDtypeInPlace
from uproot.interpretation.numerical import AsDouble32
from uproot.interpretation.numerical import AsFloat16
from uproot.interpretation.numerical import AsSTLBits
from uproot.interpretation.jagged import AsJagged
from uproot.interpretation.strings import AsStrings
from uproot.interpretation.objects import AsObjects
from uproot.interpretation.objects import AsStridedObjects
from uproot.interpretation.grouped import AsGrouped
from uproot.interpretation.identify import register_interpretation
from uproot.interpretation.identify import unregister_interpretation
from uproot.containers import AsString
from uproot.containers import AsPointer
from uproot.containers import AsArray
from uproot.containers import AsDynamic
from uproot.containers import AsRVec
from uproot.containers import AsVector
from uproot.containers import AsSet
from uproot.containers import AsMap

default_library = "ak"

from uproot.behaviors.TTree import TTree
from uproot.behaviors.TBranch import TBranch
from uproot.behaviors.TBranch import iterate
from uproot.behaviors.TBranch import concatenate

from uproot.behavior import behavior_of

from uproot._util import no_filter
from uproot._dask import dask, ImplementsFormMapping, ImplementsFormMappingInfo
from uproot._graphed import graphed, graphed_head

from uproot.pyroot import from_pyroot
from uproot.pyroot import to_pyroot
