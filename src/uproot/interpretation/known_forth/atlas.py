# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines ATLAS specific known forth code
"""

from __future__ import annotations

import re


class VectorVectorElementLink:
    """
    Known forth and awkward form for ``std::vector<std::vector<ElementLink<T>>`` types in ATLAS (D)AODs

    The forth code was adjusted from what was provided in
    ``branch._complete_forth_code`` after running ``.array()`` once.

    The binary data of one vector<vector<ElementLink<T>> looks as follows:

    * 6 bytes header for the outer vector
    * 4 bytes big endian uint for the size of the outer vector (node1)
    * for each outer vector element:
    * 4 bytes big endian uint for the size of the inner vector (node2)
    * for each inner vector element:
    * 20 bytes header for the ElementLink object
    * 4 bytes big endian uint for the ``m_persKey`` member (node3)
    * 4 bytes big endian uint for the ``m_persIndex`` member (node4)
    """

    forth_code = """
input stream
input byteoffsets
input bytestops
output node1-offsets int64
output node2-offsets int64
output node3-data uint32
output node4-data uint32

0 node1-offsets <- stack
0 node2-offsets <- stack

0 do
    byteoffsets I-> stack
    stream seek
    6 stream skip
    stream !I-> stack
    dup node1-offsets +<- stack
    0 do
        stream !I-> stack
        dup node2-offsets +<- stack
        0 do
            20 stream skip
            stream !I-> node3-data
            stream !I-> node4-data
        loop
    loop
loop
"""

    def __init__(self, typename):
        self.typename = typename
        self.inner_typename = re.sub(
            "std::vector<std::vector<(.*)>>", r"\1", self.typename
        )

    @property
    def awkward_form(self):
        return {
            "class": "ListOffsetArray",
            "offsets": "i64",
            "form_key": "node1",
            "content": {
                "class": "ListOffsetArray",
                "offsets": "i64",
                "form_key": "node2",
                "content": {
                    "class": "RecordArray",
                    "fields": ["m_persKey", "m_persIndex"],
                    "contents": [
                        {
                            "class": "NumpyArray",
                            "primitive": "uint32",
                            "inner_shape": [],
                            "parameters": {},
                            "form_key": "node3",
                        },
                        {
                            "class": "NumpyArray",
                            "primitive": "uint32",
                            "inner_shape": [],
                            "parameters": {},
                            "form_key": "node4",
                        },
                    ],
                    "parameters": {"__record__": f"{self.inner_typename}"},
                },
            },
        }
