# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import json
import queue
import sys

import numpy
import skhep_testdata

import uproot


def test():
    filename = skhep_testdata.data_path("ntpl001_staff_rntuple_v1-0-0-0.root")
    with uproot.open(filename) as f:
        obj = f["Staff"]
        assert obj.member("fVersionEpoch") == 1
        assert obj.member("fVersionMajor") == 0
        assert obj.member("fVersionMinor") == 0
        assert obj.member("fVersionPatch") == 0
        assert obj.member("fSeekHeader") == 266
        assert obj.member("fNBytesHeader") == 319
        assert obj.member("fLenHeader") == 997
        assert obj.member("fSeekFooter") == 24504
        assert obj.member("fNBytesFooter") == 84
        assert obj.member("fLenFooter") == 148
        assert obj.member("fMaxKeySize") == 1073741824

        header_start = obj.member("fSeekHeader")
        header_stop = header_start + obj.member("fNBytesHeader")
        header_chunk = f.file.source.chunk(header_start, header_stop)

        # print("HEADER")
        # cursor = uproot.Cursor(header_start)
        # cursor.debug(header_chunk, limit_bytes=80)
        # print("\n")

        notifications = queue.Queue()
        footer_start = obj.member("fSeekFooter")
        footer_stop = footer_start + obj.member("fNBytesFooter")
        header_chunk, footer_chunk = f.file.source.chunks(
            [(header_start, header_stop), (footer_start, footer_stop)],
            notifications,
        )

        # print("FOOTER")
        # cursor = uproot.Cursor(footer_start)
        # cursor.debug(footer_chunk, limit_bytes=80)
        # print("\n")

        header_chunk.wait()
        footer_chunk.wait()
        assert (header_chunk.start, header_chunk.stop) == (header_start, header_stop)
        assert (footer_chunk.start, footer_chunk.stop) == (footer_start, footer_stop)
        # both envelopes are ZSTD compressed ("ZS" is ROOT's tag for that)
        assert bytes(header_chunk.raw_data[:2]) == b"ZS"
        assert bytes(footer_chunk.raw_data[:2]) == b"ZS"


# HEADER
# --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
#  90  83   1  54   1   0 229   3   0  40 181  47 253  96 229   2 101   9   0 164
#   Z   S ---   6 --- --- --- --- ---   ( ---   / ---   ` --- ---   e --- --- ---
# --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
#  11   1   0 229   3   0   5   0   0   0  83 116  97 102 102  14   0   0   0  82
# --- --- --- --- --- --- --- --- --- ---   S   t   a   f   f --- --- --- ---   R
# --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
#  79  79  84  32 118  54  46  51  53  46  48  48  49 122 253 255  11   0   0   0
#   O   O   T       v   6   .   3   5   .   0   0   1   z --- --- --- --- --- ---
# --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
#  60   0   8   0   0   0  67  97 116 101 103 111 114 121  12   0   0   0 115 116
#   < --- --- --- --- ---   C   a   t   e   g   o   r   y --- --- --- ---   s   t


# FOOTER
# --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
#  90  83   1  75   0   0 148   0   0  40 181  47 253  32 148  21   2   0 116   2
#   Z   S ---   K --- --- --- --- ---   ( ---   / ---     --- --- --- ---   t ---
# --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
#   2   0 148   0 121 133  33  93 184 252  16 152  56 244 255 196   1   0   0   0
# --- --- --- ---   y ---   !   ] --- --- --- ---   8 --- --- --- --- --- --- ---
# --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
#  48   0  26  13  92   2 194   0   0   0 212  94  62  61 172  86  23 254 154  10
#   0 --- --- ---   \ --- --- --- --- --- ---   ^   >   = ---   V --- --- --- ---
# --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
#   0 111 200   5  12 254  12  97 101 192  56  89 192 189  47  37 224  48 158 153
# ---   o --- --- --- --- ---   a   e ---   8   Y --- ---   /   % ---   0 --- ---
