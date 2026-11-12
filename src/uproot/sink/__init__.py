# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines the "physical layer" of file-writing, which interacts with local
filesystems (and might one day include remote protocols). The "physical layer"
manages only the act of writing and sometimes reading bytes from files. Reading is
sometimes needed so that Uproot can update a preexisting file.

Unlike reading, writing has no threads and the file sink can be any object with
``read``, ``write``, ``seek``, ``tell``, and ``flush`` methods. Like reading, a
context manager (Python's ``with`` statement) ensures that files are properly closed
(although files are flushed after every object-write).
"""
