# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import numpy
import pytest
import skhep_testdata

import uproot

pytest.importorskip("awkward")


@pytest.mark.parametrize("is_forth", [False, True])
def test_00(is_forth):
    # see AwkwardForth testing: L, P, R
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-offsets int64
    #     output node2-data uint8
    #     output node3-offsets int64
    #     output node4-data float64
    #
    #         0 node0-offsets <- stack
    #     0 node1-offsets <- stack
    #     0 node3-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         12 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     6 stream skip
    #     dup 0 do
    #     stream !B-> stack dup 255 = if drop stream !I-> stack then dup node1-offsets +<- stack stream #!B-> node2-data
    #     loop
    #     6 stream skip
    #     0 do
    #     stream !I-> stack
    #      dup node3-offsets +<- stack
    #     stream #!d-> node4-data
    #     loop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("issue367b.root")) as file:
        branch = file["tree/weights"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[0]["0"][0] == "expskin_FluxUnisim"
        # py[-1] == <STLMap {'expskin_FluxUnisim': [0.944759093019904, 1.0890682745548674, ..., 1.1035170311451232, 0.8873957186284592], ...} at 0x7fbc4c1325e0>
        assert py.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


@pytest.mark.parametrize("is_forth", [False, True])
def test_01(is_forth):
    # see AwkwardForth testing: A, B, D, E, J, N
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-data float64
    #     output node2-data float64
    #     output node3-data float64
    #     output node4-data float64
    #
    #         0 node0-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         0 bytestops I-> stack
    #     begin
    #     dup stream pos <>
    #     while
    #     swap 1 + swap
    #     0 stream skip
    #     6 stream skip
    #     10 stream skip
    #     0 stream skip
    #     6 stream skip
    #     10 stream skip
    #     stream !d-> node1-data
    #     stream !d-> node2-data
    #     stream !d-> node3-data
    #     stream !d-> node4-data
    #     repeat
    #     swap node0-offsets +<- stack drop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-delphes-pr442.root")) as file:
        branch = file["Delphes/GenJet/GenJet.SoftDroppedSubJet1"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[0][0]["fE"] == pytest.approx(84.56447925448748)
        assert py[0][0]["fP"]["fZ"] == pytest.approx(-81.600465)
        # py[-1] == array([<TLorentzVector (version 4) at 0x7fac4bb8d2b0>, <TLorentzVector (version 4) at 0x7fac4bb8d2e0>], dtype=object)
        assert py.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


@pytest.mark.parametrize("is_forth", [False, True])
def test_02(is_forth):
    # see AwkwardForth testing: A, B, D, E, J, N
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-data float64
    #     output node2-data float64
    #     output node3-data float64
    #     output node4-data float64
    #
    #         0 node0-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         0 bytestops I-> stack
    #     begin
    #     dup stream pos <>
    #     while
    #     swap 1 + swap
    #     0 stream skip
    #     6 stream skip
    #     10 stream skip
    #     0 stream skip
    #     6 stream skip
    #     10 stream skip
    #     stream !d-> node1-data
    #     stream !d-> node2-data
    #     stream !d-> node3-data
    #     stream !d-> node4-data
    #     repeat
    #     swap 5 / node0-offsets +<- stack drop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-delphes-pr442.root")) as file:
        branch = file["Delphes/GenJet/GenJet.TrimmedP4[5]"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak")
        assert py[0][0][0]["fE"] == 0
        assert py[-1][8][3]["fE"] == 0
        # py[-1] == array([[<TLorentzVector (version 4) at 0x7fbfea1fa3d0>, <TLorentzVector (version 4) at 0x7fbfea1fa370>, <TLorentzVector (version 4) at 0x7fbfea1fa310>, <TLorentzVector (version 4) at 0x7fbfea1fa2b0>, <TLorentzVector (version 4) at 0x7fbfea1fa250>], [<TLorentzVector (version 4) at 0x7fbfea1fa1f0>, <TLorentzVector (version 4) at 0x7fbfea1fa190>, <TLorentzVector (version 4) at 0x7fbfea1fa130>, <TLorentzVector (version 4) at 0x7fbfea1fa0d0>, <TLorentzVector (version 4) at 0x7fbfea1fa070>]], dtype=object)
        assert py.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


@pytest.mark.parametrize("is_forth", [False, True])
def test_03(is_forth):
    # see AwkwardForth testing: L, P
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-offsets int64
    #     output node2-data uint8
    #
    #         0 node0-offsets <- stack
    #     0 node1-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         6 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     0 do
    #     stream !B-> stack dup 255 = if drop stream !I-> stack then dup node1-offsets +<- stack stream #!B-> node2-data
    #     loop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-FCCDelphesOutput.root")) as file:
        branch = file["metadata/gaudiConfigOptions"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[0][0] == 'Converter.hepmcStatusList = "[]";\n'
        assert (
            py[0][14]
            == 'ToolSvc.photons.delphesArrayName = "PhotonEfficiency/photons";\nToolSvc.photons.isolationTags = "photonITags";\nToolSvc.photons.mcAssociations = "photonsToMC";\nToolSvc.photons.particles = "photons";\n'
        )
        # py[-1] == <STLVector ['Converter.hepmcStatusList = "[]";\n', ...] at 0x7f7fc23ca2e0>
        assert py.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


@pytest.mark.parametrize("is_forth", [False, True])
def test_04(is_forth):
    # see AwkwardForth testing: P
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-offsets int64
    #     output node2-data uint32
    #
    #         0 node0-offsets <- stack
    #     0 node1-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         6 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     0 do
    #     stream !I-> stack
    #      dup node1-offsets +<- stack
    #     stream #!I-> node2-data
    #     loop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-issue-123a.root")) as file:
        branch = file["POOLContainerForm/DataHeaderForm/m_uints"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[0][0][0] == 2
        assert py[0][3][2] == 589824
        # py[-1] == <STLVector [[1], ...] at 0x7fc153ed6670>
        assert py.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


@pytest.mark.parametrize("is_forth", [False, True])
def test_05(is_forth):
    # see AwkwardForth testing: A, E (previously, this hadn't been tested: library="np")
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-data uint32
    #     output node1-data uint32
    #     output node2-data uint32
    #
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         0 stream skip
    #     stream !I-> node0-data
    #     stream !I-> node1-data
    #     stream !I-> node2-data
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-issue-123a.root")) as file:
        branch = file["CollectionTree/TrigConfKeys"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        # py[-1] == <xAOD::TrigConfKeys_v1 (version 1) at 0x7fecf9212760>
        assert py.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


@pytest.mark.parametrize("is_forth", [False, True])
def test_06(is_forth):
    # see AwkwardForth testing: P
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-offsets int64
    #     output node2-data int32
    #
    #         0 node0-offsets <- stack
    #     0 node1-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         6 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     0 do
    #     stream !I-> stack
    #      dup node1-offsets +<- stack
    #     stream #!i-> node2-data
    #     loop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-issue-123a.root")) as file:
        branch = file["CollectionTree/AnalysisJetsAuxDyn.NumTrkPt500"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[1][-1][-2] == 0
        # py[-1] == <STLVector [[7, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ...] at 0x7f1377d56610>
        assert py.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


@pytest.mark.parametrize("is_forth", [False, True])
def test_07(is_forth):
    # see AwkwardForth testing: P
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-offsets int64
    #     output node2-data float32
    #
    #         0 node0-offsets <- stack
    #     0 node1-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         6 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     0 do
    #     stream !I-> stack
    #      dup node1-offsets +<- stack
    #     stream #!f-> node2-data
    #     loop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-issue-123a.root")) as file:
        branch = file["CollectionTree/AnalysisJetsAuxDyn.SumPtTrkPt500"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[1][2][0] == pytest.approx(53949.015625)
        # py[-1] == <STLVector [[28132.615, 38298.348, 0.0, 0.0, 0.0, 665.71674, ..., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], ...] at 0x7f225f7496a0>
        assert py.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


@pytest.mark.parametrize("is_forth", [False, True])
def test_08(is_forth):
    # see AwkwardForth testing: A, B, E, P (previously, this hadn't been tested: library="np")
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-offsets int64
    #     output node2-data uint32
    #     output node3-data uint32
    #
    #         0 node0-offsets <- stack
    #     0 node1-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         6 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     0 do
    #     stream !I-> stack
    #      dup node1-offsets +<- stack
    #     0 do
    #     0 stream skip
    #     6 stream skip
    #     4 stream skip
    #     6 stream skip
    #     4 stream skip
    #     stream !I-> node2-data
    #     stream !I-> node3-data
    #     loop
    #     loop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-issue-123a.root")) as file:
        branch = file["CollectionTree/AnalysisJetsAuxDyn.GhostTrack"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        # py[-1] == <STLVector [[<ElementLink<DataVector<xAOD::IParticle>> (version 1) at 0x7fc6a08f2f70>, ...], ...] at 0x7fc6a08f2f10>
        assert py.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


@pytest.mark.parametrize("is_forth", [False, True])
def test_09(is_forth):
    # see AwkwardForth testing: A, B, E, P
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-offsets int64
    #     output node2-data uint32
    #     output node3-data uint32
    #
    #         0 node0-offsets <- stack
    #     0 node1-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         6 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     0 do
    #     stream !I-> stack
    #      dup node1-offsets +<- stack
    #     0 do
    #     0 stream skip
    #     6 stream skip
    #     4 stream skip
    #     6 stream skip
    #     4 stream skip
    #     stream !I-> node2-data
    #     stream !I-> node3-data
    #     loop
    #     loop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-issue-123a.root")) as file:
        branch = file["CollectionTree"][
            "AntiKt10UFOCSSKJetsAuxDyn.GhostVR30Rmax4Rmin02TrackJet_BTagging201903"
        ]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[0][0][0]["m_persKey"] == 352341021
        # py[-1] == <STLVector [[<ElementLink<DataVector<xAOD::IParticle>> (version 1) at 0x7febbf1b2fa0>], ...] at 0x7febbf1b2f40>
        assert py.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


@pytest.mark.parametrize("is_forth", [False, True])
def test_10(is_forth):
    # see AwkwardForth testing: P
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-offsets int64
    #     output node2-data float32
    #
    #         0 node0-offsets <- stack
    #     0 node1-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         6 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     0 do
    #     stream !I-> stack
    #      dup node1-offsets +<- stack
    #     stream #!f-> node2-data
    #     loop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-issue-123a.root")) as file:
        branch = file["CollectionTree/CaloCalTopoClustersAuxDyn.e_sampl"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=3)
        assert py[0][0][0] == pytest.approx(168.03048706054688)
        # py[-1] == <STLVector [[4499.692, 26934.783, 75009.02, 676.8455, 15.539482], ...] at 0x7fe3240a09a0>
        assert py.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


@pytest.mark.parametrize("is_forth", [False, True])
def test_11(is_forth):
    # see AwkwardForth testing: A, B, E, P
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-offsets int64
    #     output node2-data uint32
    #     output node3-data uint32
    #
    #         0 node0-offsets <- stack
    #     0 node1-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         6 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     0 do
    #     stream !I-> stack
    #      dup node1-offsets +<- stack
    #     0 do
    #     0 stream skip
    #     6 stream skip
    #     4 stream skip
    #     6 stream skip
    #     4 stream skip
    #     stream !I-> node2-data
    #     stream !I-> node3-data
    #     loop
    #     loop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-issue-123a.root")) as file:
        branch = file["CollectionTree"][
            "TruthBosonsWithDecayVerticesAuxDyn.incomingParticleLinks"
        ]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[0][0][0]["m_persKey"] == 921521854
        # py[-1] == <STLVector [[<ElementLink<DataVector<xAOD::TruthParticle_v1>> (version 1) at 0x7f636a9484c0>], ...] at 0x7f636a948eb0>
        assert py.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


@pytest.mark.parametrize("is_forth", [False, True])
def test_12(is_forth):
    # see AwkwardForth testing: A, B, E, P
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-offsets int64
    #     output node2-data uint32
    #     output node3-data uint32
    #
    #         0 node0-offsets <- stack
    #     0 node1-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         6 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     0 do
    #     stream !I-> stack
    #      dup node1-offsets +<- stack
    #     0 do
    #     0 stream skip
    #     6 stream skip
    #     4 stream skip
    #     6 stream skip
    #     4 stream skip
    #     stream !I-> node2-data
    #     stream !I-> node3-data
    #     loop
    #     loop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-issue-123a.root")) as file:
        branch = file["CollectionTree/TruthBottomAuxDyn.parentLinks"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[0][0][0]["m_persIndex"] == 2
        # py[-1] == <STLVector [[<ElementLink<DataVector<xAOD::TruthParticle_v1>> (version 1) at 0x7fc259ae37c0>], ...] at 0x7fc259ae3f10>
        assert py.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


@pytest.mark.parametrize("is_forth", [False, True])
def test_13(is_forth):
    # see AwkwardForth testing: A, B, E, P
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-offsets int64
    #     output node2-data uint32
    #     output node3-data uint32
    #
    #         0 node0-offsets <- stack
    #     0 node1-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         6 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     0 do
    #     stream !I-> stack
    #      dup node1-offsets +<- stack
    #     0 do
    #     0 stream skip
    #     6 stream skip
    #     4 stream skip
    #     6 stream skip
    #     4 stream skip
    #     stream !I-> node2-data
    #     stream !I-> node3-data
    #     loop
    #     loop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-issue-123a.root")) as file:
        branch = file["CollectionTree/egammaClustersAuxDyn.constituentClusterLinks"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=3)
        assert py[0][0][0]["m_persIndex"] == 0
        # py[-1] == <STLVector [[<ElementLink<DataVector<xAOD::CaloCluster_v1>> (version 1) at 0x7fa94e968c10>], ...] at 0x7fa94e968c70>
        assert py.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


@pytest.mark.parametrize("is_forth", [False, True])
def test_14(is_forth):
    # see AwkwardForth testing: P
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-offsets int64
    #     output node2-data float32
    #
    #         0 node0-offsets <- stack
    #     0 node1-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         6 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     0 do
    #     stream !I-> stack
    #      dup node1-offsets +<- stack
    #     stream #!f-> node2-data
    #     loop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-issue-123a.root")) as file:
        branch = file["CollectionTree/egammaClustersAuxDyn.eta_sampl"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=3)
        assert py[0][0][-1] == pytest.approx(0.4663555920124054)
        # py[-1] == <STLVector [[-0.53503126, -0.5374735, -0.5373216, -0.52376634]] at 0x7f5d2ba80d30>
        assert py.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


@pytest.mark.parametrize("is_forth", [False, True])
def test_15(is_forth):
    # see AwkwardForth testing: A, B, E, P
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-offsets int64
    #     output node2-data uint32
    #     output node3-data uint32
    #
    #         0 node0-offsets <- stack
    #     0 node1-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         6 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     0 do
    #     stream !I-> stack
    #      dup node1-offsets +<- stack
    #     0 do
    #     0 stream skip
    #     6 stream skip
    #     4 stream skip
    #     6 stream skip
    #     4 stream skip
    #     stream !I-> node2-data
    #     stream !I-> node3-data
    #     loop
    #     loop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-issue-123a.root")) as file:
        branch = file["CollectionTree/AnalysisHLT_mu24_ilooseAuxDyn.TrigMatchedObjects"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[1][-1][-1]["m_persKey"] == 980095599
        # py[-1] == <STLVector [[<ElementLink<DataVector<xAOD::IParticle>> (version 1) at 0x7fb9b9d24c10>], ...] at 0x7fb9b9d29250>
        assert py.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


@pytest.mark.parametrize("is_forth", [False, True])
def test_16(is_forth):
    # see AwkwardForth testing: A, B, E, P
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-offsets int64
    #     output node2-data uint32
    #     output node3-data uint32
    #
    #         0 node0-offsets <- stack
    #     0 node1-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         6 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     0 do
    #     stream !I-> stack
    #      dup node1-offsets +<- stack
    #     0 do
    #     0 stream skip
    #     6 stream skip
    #     4 stream skip
    #     6 stream skip
    #     4 stream skip
    #     stream !I-> node2-data
    #     stream !I-> node3-data
    #     loop
    #     loop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-issue-123a.root")) as file:
        branch = file["CollectionTree/AnalysisHLT_mu40AuxDyn.TrigMatchedObjects"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[1][0]["m_persKey"][0] == 980095599
        # py[-1] == <STLVector [[<ElementLink<DataVector<xAOD::IParticle>> (version 1) at 0x7f6e29be5cd0>], ...] at 0x7f6e29bea250>
        assert py.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


@pytest.mark.parametrize("is_forth", [False, True])
def test_17(is_forth):
    # see AwkwardForth testing: P
    # (never finishes producing AwkwardForth code)
    with uproot.open(skhep_testdata.data_path("uproot-issue390.root")) as file:
        branch = file["E/Evt/AAObject/usr_names"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=None)
        assert len(py[0]) == 0
        # py[-1] == <STLVector [] at 0x7f6afa6ead00>
        assert py.layout.form == interp.awkward_form(branch.file)


@pytest.mark.parametrize("is_forth", [False, True])
def test_18(is_forth):
    # see AwkwardForth testing: (none?)
    # That's right: this interpretation is AsJagged(AsDtype('>f8')), which doesn't ever use AwkwardForth.
    with uproot.open(skhep_testdata.data_path("uproot-issue390.root")) as file:
        branch = file["E/Evt/hits/hits.t"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[0][0] == pytest.approx(70104010.0)
        # py[-1] == array([80860738., 80861689., 80862014., 80861709., 80861737., 80861158., 80862362., 80860821., 80862271., 80862273., 80861294., 80861860., 80862548., 80861733., 80861605., 80860467., 80860408., 80861562., 80862012., 80862350., 80861491., 80860384., 80860930., 80861541., 80861461., 80861749., 80862352., 80861813., 80861822., 80861871., 80862000., 80862255., 80862253., 80862249., 80862266., 80862248., 80862246., 80862847., 80863032., 80861952., 80861954., 80861953., 80861957., 80861951., 80861961., 80861959., 80861955., 80861994., 80862060., 80861971., 80862004., 80862002., 80862059., 80861695., 80861813., 80861967., 80862919., 80862043., 80862054., 80862044., 80862044., 80862040., 80862043., 80862037., 80862040., 80862039., 80862070., 80862042., 80862322., 80861605., 80861865., 80863034., 80862987., 80861545., 80860392., 80861003., 80861564., 80862109., 80861821., 80862083., 80861121., 80862513., 80862513., 80862731., 80861604., 80862003., 80861910., 80861854., 80862297., 80860989., 80862948., 80862075., 80862141., 80862117., 80862039., 80862114., 80862075., 80862042., 80862072., 80862439., 80862481., 80861656., 80862096., 80862215., 80862215., 80862195., 80862458., 80862432., 80861915., 80861012., 80862208., 80861885., 80861888., 80861994., 80861883., 80862194., 80861812., 80862184., 80862309., 80862297., 80862840., 80862400., 80861565., 80862226., 80862149.])
        assert py.layout.form == interp.awkward_form(branch.file)


@pytest.mark.parametrize("is_forth", [False, True])
def test_19(is_forth):
    # see AwkwardForth testing: (none?)
    # That's right: this interpretation is AsJagged(AsDtype('>f8')), which doesn't ever use AwkwardForth.
    with uproot.open(skhep_testdata.data_path("uproot-issue390.root")) as file:
        branch = file["E/Evt/hits/hits.a"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[0][-1] == 0.0
        # py[-1] == array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        assert py.layout.form == interp.awkward_form(branch.file)


@pytest.mark.parametrize("is_forth", [False, True])
def test_20(is_forth):
    # see AwkwardForth testing: (none?)
    # That's right: this interpretation is AsJagged(AsDtype('>i4')), which doesn't ever use AwkwardForth.
    with uproot.open(skhep_testdata.data_path("uproot-issue390.root")) as file:
        branch = file["E/Evt/hits/hits.trig"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[0][5] == 1
        # py[-1] == array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=int32)
        assert py.layout.form == interp.awkward_form(branch.file)


@pytest.mark.parametrize("is_forth", [False, True])
def test_21(is_forth):
    # see AwkwardForth testing: (none?)
    # That's right: this interpretation is AsJagged(AsDtype('>u4')), which doesn't ever use AwkwardForth.
    with uproot.open(skhep_testdata.data_path("uproot-issue390.root")) as file:
        branch = file["E/Evt/hits/hits.tot"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[0][0] == 24
        # py[-1] == array([29, 26, 22, 18, 22, 28, 28, 28, 21, 24, 28, 30, 25, 24, 30, 28, 29, 4, 21, 25, 26, 22, 23, 22, 23, 29, 23, 30, 24, 29, 31, 27, 32, 28, 30, 33, 33, 31, 29, 18, 23, 34, 21, 33, 33, 29, 37, 23, 21, 40, 25, 29, 22, 17, 31, 25, 28, 26, 21, 20, 25, 51, 38, 64, 42, 28, 29, 26, 21, 31, 22, 18, 41, 28, 29, 28, 29, 15, 25, 27, 24, 28, 28, 34, 28, 21, 19, 21, 20, 24, 26, 24, 13, 22, 30, 25, 17, 27, 24, 16, 31, 27, 29, 23, 26, 25, 26, 28, 12, 18, 30, 27, 48, 16, 25, 24, 27, 10, 21, 25, 30, 26, 26, 28, 24], dtype=uint32)
        assert py.layout.form == interp.awkward_form(branch.file)


@pytest.mark.parametrize("is_forth", [False, True])
def test_22(is_forth):
    # see AwkwardForth testing: (none?)
    # That's right: this interpretation is AsJagged(AsDtype('>f8')), which doesn't ever use AwkwardForth.
    with uproot.open(skhep_testdata.data_path("uproot-issue390.root")) as file:
        branch = file["E/Evt/hits/hits.pos.x"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[0][2] == 0.0
        # py[-1] == array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        assert py.layout.form == interp.awkward_form(branch.file)


@pytest.mark.parametrize("is_forth", [False, True])
def test_23(is_forth):
    # see AwkwardForth testing: N, P
    # (never finishes producing AwkwardForth code)
    with uproot.open(skhep_testdata.data_path("uproot-issue390.root")) as file:
        branch = file["E/Evt/trks/trks.usr_names"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert len(py[0][0]) == 0
        # py[-1] == array([<STLVector [] at 0x7f09da748f70>, <STLVector [] at 0x7f09da75a5b0>, <STLVector [] at 0x7f09da75a610>, <STLVector [] at 0x7f09da75a670>, <STLVector [] at 0x7f09da75a6d0>, <STLVector [] at 0x7f09da75a730>, <STLVector [] at 0x7f09da75a790>, <STLVector [] at 0x7f09da75a7f0>, <STLVector [] at 0x7f09da75a850>, <STLVector [] at 0x7f09da75a8b0>, <STLVector [] at 0x7f09da75a910>, <STLVector [] at 0x7f09da75a970>, <STLVector [] at 0x7f09da75a9d0>, <STLVector [] at 0x7f09da75aa30>, <STLVector [] at 0x7f09da75aa90>, <STLVector [] at 0x7f09da75aaf0>, <STLVector [] at 0x7f09da75ab50>, <STLVector [] at 0x7f09da75abb0>, <STLVector [] at 0x7f09da75ac10>, <STLVector [] at 0x7f09da75ac70>, <STLVector [] at 0x7f09da75acd0>, <STLVector [] at 0x7f09da75ad30>, <STLVector [] at 0x7f09da75ad90>, <STLVector [] at 0x7f09da75adf0>, <STLVector [] at 0x7f09da75ae50>, <STLVector [] at 0x7f09da75aeb0>, <STLVector [] at 0x7f09da75af10>, <STLVector [] at 0x7f09da75af70>, <STLVector [] at 0x7f09da75afd0>, <STLVector [] at 0x7f09da75f070>, <STLVector [] at 0x7f09da75f0d0>, <STLVector [] at 0x7f09da75f130>, <STLVector [] at 0x7f09da75f190>, <STLVector [] at 0x7f09da75f1f0>, <STLVector [] at 0x7f09da75f250>, <STLVector [] at 0x7f09da75f2b0>, <STLVector [] at 0x7f09da75f310>, <STLVector [] at 0x7f09da75f370>, <STLVector [] at 0x7f09da75f3d0>, <STLVector [] at 0x7f09da75f430>, <STLVector [] at 0x7f09da75f490>, <STLVector [] at 0x7f09da75f4f0>, <STLVector [] at 0x7f09da75f550>, <STLVector [] at 0x7f09da75f5b0>, <STLVector [] at 0x7f09da75f610>, <STLVector [] at 0x7f09da75f670>, <STLVector [] at 0x7f09da75f6d0>, <STLVector [] at 0x7f09da75f730>, <STLVector [] at 0x7f09da75f790>, <STLVector [] at 0x7f09da75f7f0>, <STLVector [] at 0x7f09da75f850>, <STLVector [] at 0x7f09da75f8b0>, <STLVector [] at 0x7f09da75f910>, <STLVector [] at 0x7f09da75f970>, <STLVector [] at 0x7f09da75f9d0>], dtype=object)
        assert py.layout.form == interp.awkward_form(branch.file)


@pytest.mark.parametrize("is_forth", [False, True])
def test_24(is_forth):
    # see AwkwardForth testing: N, P
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-offsets int64
    #     output node2-data int32
    #
    #         0 node0-offsets <- stack
    #     0 node1-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         6 stream skip
    #     0 bytestops I-> stack
    #     begin
    #     dup stream pos <>
    #     while
    #     swap 1 + swap
    #     stream !I-> stack
    #      dup node1-offsets +<- stack
    #     stream #!i-> node2-data
    #     repeat
    #     swap node0-offsets +<- stack drop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-issue390.root")) as file:
        branch = file["E/Evt/trks/trks.rec_stages"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[0][1][-1] == 5
        # py[-1] == array([<STLVector [1, 3, 5, 4] at 0x7fea61938d60>, <STLVector [1, 3, 5] at 0x7fea6194a4f0>, <STLVector [1, 3] at 0x7fea6194a580>, <STLVector [1, 3] at 0x7fea6194a5e0>, <STLVector [1, 3] at 0x7fea6194a640>, <STLVector [1, 3] at 0x7fea6194a6a0>, <STLVector [1, 3] at 0x7fea6194a700>, <STLVector [1, 3] at 0x7fea6194a760>, <STLVector [1, 3] at 0x7fea6194a7c0>, <STLVector [1, 3] at 0x7fea6194a820>, <STLVector [1, 3] at 0x7fea6194a880>, <STLVector [1, 3] at 0x7fea6194a8e0>, <STLVector [1, 3] at 0x7fea6194a940>, <STLVector [1, 3] at 0x7fea6194a9a0>, <STLVector [1, 3] at 0x7fea6194aa00>, <STLVector [1, 3] at 0x7fea6194aa60>, <STLVector [1, 3] at 0x7fea6194aac0>, <STLVector [1, 3] at 0x7fea6194ab20>, <STLVector [1, 3] at 0x7fea6194ab80>, <STLVector [1] at 0x7fea6194abe0>, <STLVector [1] at 0x7fea6194ac40>, <STLVector [1] at 0x7fea6194aca0>, <STLVector [1] at 0x7fea6194ad00>, <STLVector [1] at 0x7fea6194ad60>, <STLVector [1] at 0x7fea6194adc0>, <STLVector [1] at 0x7fea6194ae20>, <STLVector [1] at 0x7fea6194ae80>, <STLVector [1] at 0x7fea6194aee0>, <STLVector [1] at 0x7fea6194af40>, <STLVector [1] at 0x7fea6194afa0>, <STLVector [1] at 0x7fea6194e040>, <STLVector [1] at 0x7fea6194e0a0>, <STLVector [1] at 0x7fea6194e100>, <STLVector [1] at 0x7fea6194e160>, <STLVector [1] at 0x7fea6194e1c0>, <STLVector [1] at 0x7fea6194e220>, <STLVector [1] at 0x7fea6194e280>, <STLVector [1] at 0x7fea6194e2e0>, <STLVector [1] at 0x7fea6194e340>, <STLVector [1] at 0x7fea6194e3a0>, <STLVector [1] at 0x7fea6194e400>, <STLVector [1] at 0x7fea6194e460>, <STLVector [1] at 0x7fea6194e4c0>, <STLVector [1] at 0x7fea6194e520>, <STLVector [1] at 0x7fea6194e580>, <STLVector [1] at 0x7fea6194e5e0>, <STLVector [1] at 0x7fea6194e640>, <STLVector [1] at 0x7fea6194e6a0>, <STLVector [1] at 0x7fea6194e700>, <STLVector [1] at 0x7fea6194e760>, <STLVector [1] at 0x7fea6194e7c0>, <STLVector [1] at 0x7fea6194e820>, <STLVector [1] at 0x7fea6194e880>, <STLVector [1] at 0x7fea6194e8e0>, <STLVector [1] at 0x7fea6194e940>], dtype=object)
        assert py.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


@pytest.mark.parametrize("is_forth", [False, True])
def test_25(is_forth):
    # see AwkwardForth testing: L, N
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-offsets int64
    #     output node2-data uint8
    #
    #         0 node0-offsets <- stack
    #     0 node1-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         6 stream skip
    #     0 bytestops I-> stack
    #     begin
    #     dup stream pos <>
    #     while
    #     swap 1 + swap
    #     stream !B-> stack dup 255 = if drop stream !I-> stack then dup node1-offsets +<- stack stream #!B-> node2-data
    #     repeat
    #     swap node0-offsets +<- stack drop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-issue-208.root")) as file:
        branch = file["config/VERSION/VERSION._name"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[0][-1] == "numuCCAnalysis"
        # py[-1] == array(['psychePolicy', 'psycheEventModel', 'psycheCore', 'psycheUtils', 'psycheND280Utils', 'psycheIO', 'psycheSelections', 'psycheSystematics', 'highlandEventModel', 'highlandTools', 'highlandCore', 'highlandCorrections', 'highlandIO', 'baseAnalysis', 'baseTrackerAnalysis', 'numuCCAnalysis'], dtype=object)
        assert py.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


def test_26():
    # see AwkwardForth testing: N, P
    with uproot.open(skhep_testdata.data_path("uproot-issue-208.root")) as file:
        branch = file["config/SEL/SEL._firstSteps"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        py = branch.array(interp, library="np", entry_stop=2)
        assert py.shape == (1,)  # there's only one entry
        assert py[-1].shape == (1,)
        assert isinstance(py[-1][0], uproot.containers.STLVector)
        assert py[-1][0].tolist() == []
        # py[-1] == array([<STLVector [] at 0x7f5060ef49d0>], dtype=object)


@pytest.mark.parametrize("is_forth", [False, True])
def test_27(is_forth):
    # see AwkwardForth testing: L, N, P
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-offsets int64
    #     output node2-offsets int64
    #     output node3-data uint8
    #
    #         0 node0-offsets <- stack
    #     0 node1-offsets <- stack
    #     0 node2-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         6 stream skip
    #     0 bytestops I-> stack
    #     begin
    #     dup stream pos <>
    #     while
    #     swap 1 + swap
    #     stream !I-> stack
    #      dup node1-offsets +<- stack
    #     0 do
    #     stream !B-> stack dup 255 = if drop stream !I-> stack then dup node2-offsets +<- stack stream #!B-> node3-data
    #     loop
    #     repeat
    #     swap node0-offsets +<- stack drop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-issue-208.root")) as file:
        branch = file["config/SEL/SEL._branchAlias"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[0][0][-1] == "muFGD+Np"
        # py[-1] == array([<STLVector ['muTPC', 'muTPC+pTPC', ..., 'muFGD+Np'] at 0x7f1d79005a90>], dtype=object)
        assert py.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


@pytest.mark.parametrize("is_forth", [False, True])
def test_28(is_forth):
    # see AwkwardForth testing: N, P
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-offsets int64
    #     output node2-data uint32
    #
    #         0 node0-offsets <- stack
    #     0 node1-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         6 stream skip
    #     0 bytestops I-> stack
    #     begin
    #     dup stream pos <>
    #     while
    #     swap 1 + swap
    #     stream !I-> stack
    #      dup node1-offsets +<- stack
    #     stream #!I-> node2-data
    #     repeat
    #     swap node0-offsets +<- stack drop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-issue-208.root")) as file:
        branch = file["config/SEL/SEL._nCutsInBranch"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[0][0][-1] == 10
        # py[-1] == array([<STLVector [8, 9, 10, 11, 10, 7, 6, 7, 7, 8, 11, 10] at 0x7f409846fa00>], dtype=object)
        assert py.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


def test_29():
    # see AwkwardForth testing: A, B, C, D, E, J, M
    # This one CannotBeAwkward.
    with uproot.open(skhep_testdata.data_path("uproot-issue213.root")) as file:
        branch = file["T/eventPack/fGenInfo"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        py = branch.array(interp, library="np", entry_stop=2)
        assert len(py) == 2
        assert py[-1].classname == "JPetGeantEventInformation"
        assert py[-1].instance_version == 3
        assert py[-1].member("fMaxGammaNumberIndex") == 3
        assert py[-1].member("fVtxPosition").classname == "TVector3"
        assert py[-1].member("fGenGammaNum").classname == "TBits"
        # py[-1] == <JPetGeantEventInformation (version 3) at 0x7fd5bdeedac0>


@pytest.mark.parametrize("is_forth", [False, True])
def test_30(is_forth):
    # see AwkwardForth testing: L, N
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-offsets int64
    #     output node2-data uint8
    #
    #         0 node0-offsets <- stack
    #     0 node1-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         6 stream skip
    #     0 bytestops I-> stack
    #     begin
    #     dup stream pos <>
    #     while
    #     swap 1 + swap
    #     stream !B-> stack dup 255 = if drop stream !I-> stack then dup node1-offsets +<- stack stream #!B-> node2-data
    #     repeat
    #     swap node0-offsets +<- stack drop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-issue243-new.root")) as file:
        branch = file["triggerList/triggerMap/triggerMap.first"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=16)
        assert len(py[0]) == 0
        # py[-1] == array(['HLT_2j35_bmv2c2060_split_2j35_L14J15.0ETA25', 'HLT_j100_2j55_bmv2c2060_split'], dtype=object)
        assert py.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


@pytest.mark.parametrize("is_forth", [False, True])
def test_31(is_forth):
    # see AwkwardForth testing: P
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-offsets int64
    #     output node2-data uint32
    #
    #         0 node0-offsets <- stack
    #     0 node1-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         6 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     0 do
    #     stream !I-> stack
    #      dup node1-offsets +<- stack
    #     stream #!I-> node2-data
    #     loop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-issue-308.root")) as file:
        branch = file["MetaData/BranchIDLists"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[0][0][-1] == 2428801822
        # py[-1] == <STLVector [[1971320715, 1805338087, 475485005, ..., 2417357619, 128868952, 2428801822], ...] at 0x7f0a7dd4a1f0>
        assert py.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


@pytest.mark.parametrize("is_forth", [False, True])
def test_32(is_forth):
    # see AwkwardForth testing: (none?)
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-data uint8
    #
    #         0 node0-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         stream !I-> stack dup node0-offsets +<- stack stream #B-> node1-data
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-issue31.root")) as file:
        branch = file["T/data/name"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[-1] == "two"
        # py[-1] == "two"
        assert py.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


@pytest.mark.parametrize("is_forth", [False, True])
def test_33(is_forth):
    # see AwkwardForth testing: L, P, R
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-offsets int64
    #     output node2-data uint8
    #     output node3-offsets int64
    #     output node4-data float64
    #
    #         0 node0-offsets <- stack
    #     0 node1-offsets <- stack
    #     0 node3-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         12 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     6 stream skip
    #     dup 0 do
    #     stream !B-> stack dup 255 = if drop stream !I-> stack then dup node1-offsets +<- stack stream #!B-> node2-data
    #     loop
    #     6 stream skip
    #     0 do
    #     stream !I-> stack
    #      dup node3-offsets +<- stack
    #     stream #!d-> node4-data
    #     loop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-issue367b.root")) as file:
        branch = file["tree/weights"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[0][4]["1"][-1] == 1.0
        # py[-1] == <STLMap {'expskin_FluxUnisim': [0.944759093019904, 1.0890682745548674, ..., 1.1035170311451232, 0.8873957186284592], ...} at 0x7f4443068ca0>
        assert py.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


@pytest.mark.parametrize("is_forth", [False, True])
def test_34(is_forth):
    # see AwkwardForth testing: (none?)
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-data uint8
    #
    #         0 node0-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         6 stream skip
    #     stream !B-> stack dup 255 = if drop stream !I-> stack then dup node0-offsets +<- stack stream #!B-> node1-data
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-issue371.root")) as file:
        branch = file["Header/Header./Header.geant4Version"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[-1] == "$Name: geant4-10-05-patch-01 $"
        assert py.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


@pytest.mark.parametrize("is_forth", [False, True])
def test_35(is_forth):
    # see AwkwardForth testing: A, E, G, L, R
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-data int32
    #     output node2-offsets int64
    #     output node3-data uint8
    #     output node4-data int32
    #     output node5-data float64
    #
    #         0 node0-offsets <- stack
    #     0 node2-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         12 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     dup
    #     stream #!i-> node1-data
    #     0 do
    #     0 stream skip
    #     6 stream skip
    #     4 stream skip
    #     6 stream skip
    #     stream !B-> stack dup 255 = if drop stream !I-> stack then dup node2-offsets +<- stack stream #!B-> node3-data
    #     stream !i-> node4-data
    #     stream !d-> node5-data
    #     loop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-issue371.root")) as file:
        branch = file["Geant4Data/Geant4Data./Geant4Data.particles"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[0, "1", 0, "name"] == "anti_alpha"
        # py[-1] == <STLMap {-1000020040: <BDSOutputROOTGeant4Data::ParticleInfo (version 1) at 0x7fb557996df0>, ...} at 0x7fb557a012e0>
        assert py.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


@pytest.mark.parametrize("is_forth", [False, True])
def test_36(is_forth):
    # see AwkwardForth testing: L, P
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-offsets int64
    #     output node2-data uint8
    #
    #         0 node0-offsets <- stack
    #     0 node1-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         6 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     0 do
    #     stream !B-> stack dup 255 = if drop stream !I-> stack then dup node1-offsets +<- stack stream #!B-> node2-data
    #     loop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-issue371.root")) as file:
        branch = file["Model/Model./Model.samplerNamesUnique"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[0][-1] == "MP_F_99."
        # py[-1] == <STLVector ['DRIFT_0.', 'PRXSE01A.', ..., 'PRBHF_99.', 'MP_F_99.'] at 0x7f22f206df10>
        assert py.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


@pytest.mark.parametrize("is_forth", [False, True])
def test_37(is_forth):
    # see AwkwardForth testing: (none?)
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-data float64
    #     output node2-data float64
    #     output node3-data float64
    #
    #         0 node0-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         6 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     0 do
    #     0 stream skip
    #     6 stream skip
    #     10 stream skip
    #     stream !d-> node1-data
    #     stream !d-> node2-data
    #     stream !d-> node3-data
    #     loop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-issue371.root")) as file:
        branch = file["Model/Model./Model.staPos"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[0][555]["fZ"] == 100.94856572890848
        # py[-1] == array([<TVector3 (version 3) at 0x7f3385c9cbe0>, <TVector3 (version 3) at 0x7f3385c9cc10>, <TVector3 (version 3) at 0x7f3385c9cc40>, ..., <TVector3 (version 3) at 0x7f3385b0cfd0>, <TVector3 (version 3) at 0x7f3385a9e040>, <TVector3 (version 3) at 0x7f3385a9e070>], dtype=object)
        assert py.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


@pytest.mark.parametrize("is_forth", [False, True])
def test_38(is_forth):
    # see AwkwardForth testing: A, B, D, G, L, P
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-data uint8
    #     output node2-data int32
    #     output node3-offsets int64
    #     output node4-data float32
    #     output node5-offsets int64
    #     output node6-data float32
    #     output node7-offsets int64
    #     output node8-data float32
    #     output node9-data float32
    #     output node10-offsets int64
    #     output node11-data float32
    #     output node12-offsets int64
    #     output node13-data float32
    #     output node14-offsets int64
    #     output node15-data float32
    #     output node16-offsets int64
    #     output node17-data float32
    #     output node18-offsets int64
    #     output node19-data float32
    #     output node20-offsets int64
    #     output node21-data int32
    #     output node22-offsets int64
    #     output node23-data int32
    #     output node24-offsets int64
    #     output node25-data int32
    #     output node26-data int32
    #     output node27-offsets int64
    #     output node28-data int32
    #     output node29-data float32
    #     output node30-offsets int64
    #     output node31-data float32
    #     output node32-offsets int64
    #     output node33-data float32
    #     output node34-offsets int64
    #     output node35-data float32
    #     output node36-offsets int64
    #     output node37-data float32
    #     output node38-offsets int64
    #     output node39-data float32
    #     output node40-offsets int64
    #     output node41-data int32
    #     output node42-offsets int64
    #     output node43-data float32
    #     output node44-offsets int64
    #     output node45-data float32
    #     output node46-offsets int64
    #     output node47-data float32
    #     output node48-offsets int64
    #     output node49-data bool
    #     output node50-offsets int64
    #     output node51-data int32
    #     output node52-offsets int64
    #     output node53-data int32
    #     output node54-offsets int64
    #     output node55-data int32
    #
    #         0 node0-offsets <- stack
    #     0 node3-offsets <- stack
    #     0 node5-offsets <- stack
    #     0 node7-offsets <- stack
    #     0 node10-offsets <- stack
    #     0 node12-offsets <- stack
    #     0 node14-offsets <- stack
    #     0 node16-offsets <- stack
    #     0 node18-offsets <- stack
    #     0 node20-offsets <- stack
    #     0 node22-offsets <- stack
    #     0 node24-offsets <- stack
    #     0 node27-offsets <- stack
    #     0 node30-offsets <- stack
    #     0 node32-offsets <- stack
    #     0 node34-offsets <- stack
    #     0 node36-offsets <- stack
    #     0 node38-offsets <- stack
    #     0 node40-offsets <- stack
    #     0 node42-offsets <- stack
    #     0 node44-offsets <- stack
    #     0 node46-offsets <- stack
    #     0 node48-offsets <- stack
    #     0 node50-offsets <- stack
    #     0 node52-offsets <- stack
    #     0 node54-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         0 stream skip
    #     10 stream skip
    #     6 stream skip
    #     stream !B-> stack dup 255 = if drop stream !I-> stack then dup node0-offsets +<- stack stream #!B-> node1-data
    #     stream !i-> node2-data
    #     6 stream skip
    #     stream !I-> stack
    #      dup node3-offsets +<- stack
    #     stream #!f-> node4-data
    #     6 stream skip
    #     stream !I-> stack
    #      dup node5-offsets +<- stack
    #     stream #!f-> node6-data
    #     6 stream skip
    #     stream !I-> stack
    #      dup node7-offsets +<- stack
    #     stream #!f-> node8-data
    #     stream !f-> node9-data
    #     6 stream skip
    #     stream !I-> stack
    #      dup node10-offsets +<- stack
    #     stream #!f-> node11-data
    #     6 stream skip
    #     stream !I-> stack
    #      dup node12-offsets +<- stack
    #     stream #!f-> node13-data
    #     6 stream skip
    #     stream !I-> stack
    #      dup node14-offsets +<- stack
    #     stream #!f-> node15-data
    #     6 stream skip
    #     stream !I-> stack
    #      dup node16-offsets +<- stack
    #     stream #!f-> node17-data
    #     6 stream skip
    #     stream !I-> stack
    #      dup node18-offsets +<- stack
    #     stream #!f-> node19-data
    #     6 stream skip
    #     stream !I-> stack
    #      dup node20-offsets +<- stack
    #     stream #!i-> node21-data
    #     6 stream skip
    #     stream !I-> stack
    #      dup node22-offsets +<- stack
    #     stream #!i-> node23-data
    #     6 stream skip
    #     stream !I-> stack
    #      dup node24-offsets +<- stack
    #     stream #!i-> node25-data
    #     stream !i-> node26-data
    #     6 stream skip
    #     stream !I-> stack
    #      dup node27-offsets +<- stack
    #     stream #!i-> node28-data
    #     stream !f-> node29-data
    #     6 stream skip
    #     stream !I-> stack
    #      dup node30-offsets +<- stack
    #     stream #!f-> node31-data
    #     6 stream skip
    #     stream !I-> stack
    #      dup node32-offsets +<- stack
    #     stream #!f-> node33-data
    #     6 stream skip
    #     stream !I-> stack
    #      dup node34-offsets +<- stack
    #     stream #!f-> node35-data
    #     6 stream skip
    #     stream !I-> stack
    #      dup node36-offsets +<- stack
    #     stream #!f-> node37-data
    #     6 stream skip
    #     stream !I-> stack
    #      dup node38-offsets +<- stack
    #     stream #!f-> node39-data
    #     6 stream skip
    #     stream !I-> stack
    #      dup node40-offsets +<- stack
    #     stream #!i-> node41-data
    #     6 stream skip
    #     stream !I-> stack
    #      dup node42-offsets +<- stack
    #     stream #!f-> node43-data
    #     6 stream skip
    #     stream !I-> stack
    #      dup node44-offsets +<- stack
    #     stream #!f-> node45-data
    #     6 stream skip
    #     stream !I-> stack
    #      dup node46-offsets +<- stack
    #     stream #!f-> node47-data
    #     6 stream skip
    #     stream !I-> stack
    #      dup node48-offsets +<- stack
    #     stream #!?-> node49-data
    #     6 stream skip
    #     stream !I-> stack
    #      dup node50-offsets +<- stack
    #     stream #!i-> node51-data
    #     6 stream skip
    #     stream !I-> stack
    #      dup node52-offsets +<- stack
    #     stream #!i-> node53-data
    #     6 stream skip
    #     stream !I-> stack
    #      dup node54-offsets +<- stack
    #     stream #!i-> node55-data
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-issue371.root")) as file:
        branch = file["Event/PRBHF_46."]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[0]["zp"][0] == pytest.approx(0.9999998807907104)
        # py[-1] == <BDSOutputROOTEventSampler<float> (version 4) at 0x7f9be2f7b2e0>
        assert py.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


@pytest.mark.parametrize("is_forth", [False, True])
def test_39(is_forth):
    # see AwkwardForth testing: A, E, G, L, R
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-data int32
    #     output node2-offsets int64
    #     output node3-data uint8
    #     output node4-data int32
    #     output node5-data float64
    #
    #         0 node0-offsets <- stack
    #     0 node2-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         12 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     dup
    #     stream #!i-> node1-data
    #     0 do
    #     0 stream skip
    #     6 stream skip
    #     4 stream skip
    #     6 stream skip
    #     stream !B-> stack dup 255 = if drop stream !I-> stack then dup node2-offsets +<- stack stream #!B-> node3-data
    #     stream !i-> node4-data
    #     stream !d-> node5-data
    #     loop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-issue399.root")) as file:
        branch = file["Geant4Data/Geant4Data./Geant4Data.particles"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[-1][1]["0"] == -1000020030
        # py[-1] == <STLMap {-1000020040: <BDSOutputROOTGeant4Data::ParticleInfo (version 1) at 0x7f6d05752220>, ...} at 0x7f6d057397f0>
        assert py.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


@pytest.mark.parametrize("is_forth", [False, True])
def test_40(is_forth):
    # see AwkwardForth testing: (none?)
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-data int32
    #
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         stream !I-> node0-data
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-issue-407.root")) as file:
        branch = file["tree/branch"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[0]["fDatime"] == 1749155840
        # py[-1] == <TDatime at 0x7f79c7368430>
        assert py.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


@pytest.mark.parametrize("is_forth", [False, True])
def test_41(is_forth):
    # see AwkwardForth testing: P
    # (never finishes producing AwkwardForth code)
    with uproot.open(skhep_testdata.data_path("uproot-issue468.root")) as file:
        branch = file["Event/Trajectory./Trajectory.XYZ"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert len(py[0]) == 0
        # py[-1] == <STLVector [] at 0x7feac87629a0>
        assert py.layout.form == interp.awkward_form(branch.file)


@pytest.mark.parametrize("is_forth", [False, True])
def test_42(is_forth):
    # see AwkwardForth testing: P
    # (never finishes producing AwkwardForth code)
    with uproot.open(skhep_testdata.data_path("uproot-issue468.root")) as file:
        branch = file["Event/Trajectory./Trajectory.energyDeposit"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert len(py[0]) == 0
        # py[-1] == <STLVector [] at 0x7feac87629a0>
        assert py.layout.form == interp.awkward_form(branch.file)


@pytest.mark.parametrize("is_forth", [False, True])
def test_43(is_forth):
    # see AwkwardForth testing: P
    # (never finishes producing AwkwardForth code)
    with uproot.open(skhep_testdata.data_path("uproot-issue468.root")) as file:
        branch = file["Event/Trajectory./Trajectory.ionA"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert len(py[0]) == 0
        # py[-1] == <STLVector [] at 0x7f90c6543af0>
        assert py.layout.form == interp.awkward_form(branch.file)


@pytest.mark.parametrize("is_forth", [False, True])
def test_44(is_forth):
    # see AwkwardForth testing: A, C, D, F, G, P
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-data bool
    #     output node1-data bool
    #     output node2-offsets int64
    #     output node3-data uint32
    #     output node4-data bool
    #     output node5-offsets int64
    #     output node6-offsets int64
    #     output node7-data bool
    #
    #         0 node2-offsets <- stack
    #     variable var_N
    #     0 node5-offsets <- stack
    #     0 node6-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         0 stream skip
    #     stream !?-> node0-data
    #     10 dup node2-offsets +<- stack
    #      stream #!?-> node1-data
    #     stream !I-> stack dup var_N ! node3-data <- stack
    #     1 stream skip
    #      var_N @ dup node5-offsets +<- stack
    #      stream #!?-> node4-data
    #     6 stream skip
    #     stream !I-> stack
    #      dup node6-offsets +<- stack
    #     stream #!?-> node7-data
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-issue46.root")) as file:
        branch = file["tree/evt"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[0]["ArrayBool"][-1] == True
        # py[-1] == <Event (version 1) at 0x7f1e3aef2dc0>
        assert py.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


def test_45():
    # see AwkwardForth testing: A, B, E, G, J, L, M, P
    # This one CannotBeAwkward.
    with uproot.open(skhep_testdata.data_path("uproot-issue485.root")) as file:
        branch = file["MCTrack/global"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        py = branch.array(interp, library="np", entry_stop=2)
        assert [len(x) for x in py] == [145, 142]
        track = py[-1][0]
        assert track.classname == "allpix::MCTrack"
        assert track.member("particle_id_") == 2212
        assert track.member("origin_g4_vol_name_") == "World"
        assert track.member("initial_kin_E_") == 120.0
        assert track.member("final_kin_E_") == pytest.approx(116.63632162357003)
        assert track.member("start_point_").member("fCoordinates").member(
            "fZ"
        ) == pytest.approx(-10.0)
        # py[-1] == <STLVector [<allpix::MCTrack (version 2) at 0x7f88b9a8ad90>, ...] at 0x7f88b9a8ad00>


def test_46():
    # see AwkwardForth testing: A, B, E, J, M, P
    # This one CannotBeAwkward.
    with uproot.open(skhep_testdata.data_path("uproot-issue485.root")) as file:
        branch = file["MCParticle/detector1"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        py = branch.array(interp, library="np", entry_stop=2)
        assert [len(x) for x in py] == [121, 116]
        particle = py[-1][0]
        assert particle.classname == "allpix::MCParticle"
        assert particle.member("particle_id_") == 2212
        assert particle.member("time_") == pytest.approx(0.07104118009984536)
        assert particle.member("local_start_point_").member("fCoordinates").member(
            "fZ"
        ) == pytest.approx(-0.15)
        # py[-1] == <STLVector [<allpix::MCParticle (version 6) at 0x7f94bc223760>, ...] at 0x7f94bc223550>


@pytest.mark.parametrize("is_forth", [False, True])
def test_47(is_forth):
    # see AwkwardForth testing: A, E, G, L, R
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-data int32
    #     output node2-offsets int64
    #     output node3-data uint8
    #     output node4-data int32
    #     output node5-data float64
    #
    #         0 node0-offsets <- stack
    #     0 node2-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         12 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     dup
    #     stream #!i-> node1-data
    #     0 do
    #     0 stream skip
    #     6 stream skip
    #     4 stream skip
    #     6 stream skip
    #     stream !B-> stack dup 255 = if drop stream !I-> stack then dup node2-offsets +<- stack stream #!B-> node3-data
    #     stream !i-> node4-data
    #     stream !d-> node5-data
    #     loop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-issue494.root")) as file:
        branch = file["Geant4Data/Geant4Data./Geant4Data.particles"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[-1][-1]["1"]["mass"] == pytest.approx(3.727379)
        # py[-1] == <STLMap {-1000020040: <BDSOutputROOTGeant4Data::ParticleInfo (version 1) at 0x7f53a43c2220>, ...} at 0x7f53a44278b0>
        assert py.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


def test_48():
    # see AwkwardForth testing: R
    # This one CannotBeAwkward.
    with uproot.open(skhep_testdata.data_path("uproot-issue494.root")) as file:
        branch = file["Geant4Data/Geant4Data./Geant4Data.ions"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        py = branch.array(interp, library="np")
        assert len(py) == 1
        assert len(py[0]) == 0
        # py[-1] == <STLMap {} at 0x7fda831357f0>


def test_49():
    # see AwkwardForth testing: A, B, E, J, M, P
    # This one CannotBeAwkward.
    with uproot.open(skhep_testdata.data_path("uproot-issue498.root")) as file:
        branch = file["MCParticle/timepix"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        py = branch.array(interp, library="np", entry_stop=2)
        assert py[0][0].member("particle_id_") == 22
        # py[-1] == <STLVector [<allpix::MCParticle (version 6) at 0x7f0697bf1820>] at 0x7f0697bf1a00>


def test_50():
    # see AwkwardForth testing: A, B, E, G, J, M, P
    # This one CannotBeAwkward.
    with uproot.open(skhep_testdata.data_path("uproot-issue498.root")) as file:
        branch = file["PixelHit/timepix"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        py = branch.array(interp, library="np", entry_stop=2)
        assert len(py[0]) == 0
        assert len(py[1]) == 0
        # py[-1] == <STLVector [] at 0x7f769cbf0bb0>


@pytest.mark.parametrize("is_forth", [False, True])
def test_51(is_forth):
    # see AwkwardForth testing: L, N
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-offsets int64
    #     output node2-data uint8
    #
    #         0 node0-offsets <- stack
    #     0 node1-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         6 stream skip
    #     0 bytestops I-> stack
    #     begin
    #     dup stream pos <>
    #     while
    #     swap 1 + swap
    #     stream !B-> stack dup 255 = if drop stream !I-> stack then dup node1-offsets +<- stack stream #!B-> node2-data
    #     repeat
    #     swap node0-offsets +<- stack drop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-issue510b.root")) as file:
        branch = file["EDepSimEvents/Event/Primaries/Primaries.GeneratorName"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[-1][0] == "GENIE:fixed@density-fixed"
        assert py.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


@pytest.mark.parametrize("is_forth", [False, True])
def test_52(is_forth):
    # see AwkwardForth testing: A, N
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node2-offsets int64
    #     output node3-data uint8
    #     output node4-data int32
    #     output node5-offsets int64
    #     output node6-data int32
    #
    #         0 node0-offsets <- stack
    #     0 node2-offsets <- stack
    #     0 node5-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         0 bytestops I-> stack
    #     begin
    #     dup stream pos <>
    #     while
    #     swap 1 + swap
    #     6 stream skip
    #     10 stream skip
    #     stream !B-> stack dup 255 = if drop stream !I-> stack then dup node2-offsets +<- stack stream #!B-> node3-data
    #     stream !I-> stack dup node4-data <- stack
    #     6 stream skip
    #     dup node5-offsets +<- stack stream #!I-> node6-data
    #     repeat
    #     swap node0-offsets +<- stack drop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-issue513.root")) as file:
        branch = file["Delphes/EFlowPhoton/EFlowPhoton.Particles"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[1][-1]["refs"][-1] == 2223
        # py[-1] == array([<TRefArray [2407] at 0x7f7888dff0d0>, <TRefArray [2411] at 0x7f7888dee100>, <TRefArray [2405] at 0x7f7888dff130>, <TRefArray [2406] at 0x7f7888dff1f0>, <TRefArray [2523] at 0x7f7888dff280>, <TRefArray [2352] at 0x7f7888dff310>, <TRefArray [2192] at 0x7f7888dff3a0>, <TRefArray [2425] at 0x7f7888dff430>, <TRefArray [2340] at 0x7f7888dff4c0>, <TRefArray [2533] at 0x7f7888dff550>, <TRefArray [2264] at 0x7f7888dff5e0>, <TRefArray [2263] at 0x7f7888dff670>, <TRefArray [2396] at 0x7f7888dff700>, <TRefArray [2519] at 0x7f7888dff790>, <TRefArray [2044] at 0x7f7888dff820>, <TRefArray [2273] at 0x7f7888dff8b0>, <TRefArray [2270] at 0x7f7888dff940>, <TRefArray [2388] at 0x7f7888dff9d0>, <TRefArray [2473] at 0x7f7888dffa60>, <TRefArray [2272] at 0x7f7888dffaf0>, <TRefArray [2475] at 0x7f7888dffb80>, <TRefArray [2212] at 0x7f7888dffc10>, <TRefArray [2220] at 0x7f7888dffca0>, <TRefArray [2358] at 0x7f7888dffd30>, <TRefArray [2472] at 0x7f7888dffdc0>, <TRefArray [2359] at 0x7f7888dffe50>, <TRefArray [2360] at 0x7f7888dffee0>, <TRefArray [2201] at 0x7f7888dfff70>, <TRefArray [2362] at 0x7f7888e02040>, <TRefArray [2537] at 0x7f7888e020d0>, <TRefArray [2230] at 0x7f7888e02160>, <TRefArray [2488] at 0x7f7888e021f0>, <TRefArray [2307] at 0x7f7888e02280>, <TRefArray [2570] at 0x7f7888e02310>, <TRefArray [2569] at 0x7f7888e023a0>, <TRefArray [2515] at 0x7f7888e02430>, <TRefArray [2423] at 0x7f7888e024c0>, <TRefArray [2571] at 0x7f7888e02550>, <TRefArray [2578] at 0x7f7888e025e0>, <TRefArray [2386] at 0x7f7888e02670>, <TRefArray [2579] at 0x7f7888e02700>, <TRefArray [2580] at 0x7f7888e02790>, <TRefArray [2556] at 0x7f7888e02820>, <TRefArray [2555] at 0x7f7888e028b0>, <TRefArray [2355] at 0x7f7888e02940>, <TRefArray [2222, 2223] at 0x7f7888e029d0>], dtype=object)
        assert py.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


@pytest.mark.parametrize("is_forth", [False, True])
def test_53(is_forth):
    # see AwkwardForth testing: (none?)
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-data float64
    #     output node2-data float64
    #     output node3-data float64
    #     output node4-data float64
    #
    #         0 node0-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         0 bytestops I-> stack
    #     begin
    #     dup stream pos <>
    #     while
    #     swap 1 + swap
    #     0 stream skip
    #     6 stream skip
    #     10 stream skip
    #     0 stream skip
    #     6 stream skip
    #     10 stream skip
    #     stream !d-> node1-data
    #     stream !d-> node2-data
    #     stream !d-> node3-data
    #     stream !d-> node4-data
    #     repeat
    #     swap node0-offsets +<- stack drop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-issue513.root")) as file:
        branch = file["Delphes/Jet/Jet.SoftDroppedSubJet2"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[0][-1]["fE"] == 0.0
        # py[-1] == array([<TLorentzVector (version 4) at 0x7f6a396bf2b0>, <TLorentzVector (version 4) at 0x7f6a396bf250>, <TLorentzVector (version 4) at 0x7f6a396bf1f0>, <TLorentzVector (version 4) at 0x7f6a396bf190>, <TLorentzVector (version 4) at 0x7f6a396bf130>, <TLorentzVector (version 4) at 0x7f6a396bf0d0>], dtype=object)
        assert py.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


@pytest.mark.parametrize("is_forth", [False, True])
def test_54(is_forth):
    # see AwkwardForth testing: (none?)
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-data float64
    #     output node2-data float64
    #     output node3-data float64
    #     output node4-data float64
    #
    #         0 node0-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         0 bytestops I-> stack
    #     begin
    #     dup stream pos <>
    #     while
    #     swap 1 + swap
    #     0 stream skip
    #     6 stream skip
    #     10 stream skip
    #     0 stream skip
    #     6 stream skip
    #     10 stream skip
    #     stream !d-> node1-data
    #     stream !d-> node2-data
    #     stream !d-> node3-data
    #     stream !d-> node4-data
    #     repeat
    #     swap 5 / node0-offsets +<- stack drop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-issue513.root")) as file:
        branch = file["Delphes/Jet/Jet.TrimmedP4[5]"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[0][-1][4]["fE"] == 0.0
        # py[-1] == array([[<TLorentzVector (version 4) at 0x7f6a02cb3220>, <TLorentzVector (version 4) at 0x7f6a02cb3280>, <TLorentzVector (version 4) at 0x7f6a02cb3fd0>, <TLorentzVector (version 4) at 0x7f6a02cb3f70>, <TLorentzVector (version 4) at 0x7f6a02cb3f10>], [<TLorentzVector (version 4) at 0x7f6a02cb3eb0>, <TLorentzVector (version 4) at 0x7f6a02cb3e50>, <TLorentzVector (version 4) at 0x7f6a02cb3df0>, <TLorentzVector (version 4) at 0x7f6a02cb3d90>, <TLorentzVector (version 4) at 0x7f6a02cb3d30>], [<TLorentzVector (version 4) at 0x7f6a02cb3cd0>, <TLorentzVector (version 4) at 0x7f6a02cb3c70>, <TLorentzVector (version 4) at 0x7f6a02cb3c10>, <TLorentzVector (version 4) at 0x7f6a02cb3bb0>, <TLorentzVector (version 4) at 0x7f6a02cb3b50>], [<TLorentzVector (version 4) at 0x7f6a02cb3af0>, <TLorentzVector (version 4) at 0x7f6a02cb3a90>, <TLorentzVector (version 4) at 0x7f6a02cb3a30>, <TLorentzVector (version 4) at 0x7f6a02cb39d0>, <TLorentzVector (version 4) at 0x7f6a02cb3970>], [<TLorentzVector (version 4) at 0x7f6a02cb3910>, <TLorentzVector (version 4) at 0x7f6a02cb38b0>, <TLorentzVector (version 4) at 0x7f6a02cb3850>, <TLorentzVector (version 4) at 0x7f6a02cb34f0>, <TLorentzVector (version 4) at 0x7f6a02cb3580>], [<TLorentzVector (version 4) at 0x7f6a02cb3430>, <TLorentzVector (version 4) at 0x7f6a02cb33d0>, <TLorentzVector (version 4) at 0x7f6a02cb3370>, <TLorentzVector (version 4) at 0x7f6a02cb32e0>, <TLorentzVector (version 4) at 0x7f6a02cba910>]], dtype=object)
        assert py.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


@pytest.mark.parametrize("is_forth", [False, True])
def test_55(is_forth):
    # see AwkwardForth testing: P
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-offsets int64
    #     output node2-data float32
    #
    #         0 node0-offsets <- stack
    #     0 node1-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         6 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     0 do
    #     stream !I-> stack
    #      dup node1-offsets +<- stack
    #     stream #!f-> node2-data
    #     loop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-issue519.root")) as file:
        branch = file["testtree/testbranch"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[-1][1][5] == pytest.approx(0.346174418926239)
        # py[-1] == <STLVector [[0.15334472, 0.10603446, 0.004459681, 0.0222604, 0.38413692, 0.39519757], ...] at 0x7f38c4650eb0>
        assert py.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


@pytest.mark.parametrize("is_forth", [False, True])
def test_56(is_forth):
    # see AwkwardForth testing: A, C, D, E, F, G, J, L, P
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-data uint8
    #     output node2-data int16
    #     output node3-data int32
    #     output node4-data int64
    #     output node5-data uint16
    #     output node6-data uint32
    #     output node7-data uint64
    #     output node8-data float32
    #     output node9-data float64
    #     output node10-offsets int64
    #     output node11-data uint8
    #     output node12-data int32
    #     output node13-data float64
    #     output node14-data int32
    #     output node15-data int16
    #     output node16-offsets int64
    #     output node17-data int32
    #     output node18-offsets int64
    #     output node19-data int64
    #     output node20-offsets int64
    #     output node21-data uint16
    #     output node22-offsets int64
    #     output node23-data uint32
    #     output node24-offsets int64
    #     output node25-data uint64
    #     output node26-offsets int64
    #     output node27-data float32
    #     output node28-offsets int64
    #     output node29-data float64
    #     output node30-offsets int64
    #     output node31-data uint32
    #     output node32-data int16
    #     output node33-offsets int64
    #     output node34-data int32
    #     output node35-offsets int64
    #     output node36-data int64
    #     output node37-offsets int64
    #     output node38-data uint16
    #     output node39-offsets int64
    #     output node40-data uint32
    #     output node41-offsets int64
    #     output node42-data uint64
    #     output node43-offsets int64
    #     output node44-data float32
    #     output node45-offsets int64
    #     output node46-data float64
    #     output node47-offsets int64
    #     output node48-offsets int64
    #     output node49-data uint8
    #     output node50-offsets int64
    #     output node51-data int16
    #     output node52-offsets int64
    #     output node53-data int32
    #     output node54-offsets int64
    #     output node55-data int64
    #     output node56-offsets int64
    #     output node57-data uint16
    #     output node58-offsets int64
    #     output node59-data uint32
    #     output node60-offsets int64
    #     output node61-data uint64
    #     output node62-offsets int64
    #     output node63-data float32
    #     output node64-offsets int64
    #     output node65-data float64
    #     output node66-offsets int64
    #     output node67-offsets int64
    #     output node68-data uint8
    #     output node69-offsets int64
    #     output node70-data uint8
    #
    #         0 node0-offsets <- stack
    #     0 node10-offsets <- stack
    #     0 node16-offsets <- stack
    #     0 node18-offsets <- stack
    #     0 node20-offsets <- stack
    #     0 node22-offsets <- stack
    #     0 node24-offsets <- stack
    #     0 node26-offsets <- stack
    #     0 node28-offsets <- stack
    #     0 node30-offsets <- stack
    #     variable var_N
    #     0 node33-offsets <- stack
    #     0 node35-offsets <- stack
    #     0 node37-offsets <- stack
    #     0 node39-offsets <- stack
    #     0 node41-offsets <- stack
    #     0 node43-offsets <- stack
    #     0 node45-offsets <- stack
    #     0 node47-offsets <- stack
    #     0 node48-offsets <- stack
    #     0 node50-offsets <- stack
    #     0 node52-offsets <- stack
    #     0 node54-offsets <- stack
    #     0 node56-offsets <- stack
    #     0 node58-offsets <- stack
    #     0 node60-offsets <- stack
    #     0 node62-offsets <- stack
    #     0 node64-offsets <- stack
    #     0 node66-offsets <- stack
    #     0 node67-offsets <- stack
    #     0 node69-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         0 stream skip
    #      stream !B-> stack dup 255 = if drop stream !I-> stack then dup node0-offsets +<- stack stream #!B-> node1-data
    #     stream !h-> node2-data
    #     stream !i-> node3-data
    #     stream !q-> node4-data
    #     stream !H-> node5-data
    #     stream !I-> node6-data
    #     stream !Q-> node7-data
    #     stream !f-> node8-data
    #     stream !d-> node9-data
    #      stream !B-> stack dup 255 = if drop stream !I-> stack then dup node10-offsets +<- stack stream #!B-> node11-data
    #     0 stream skip
    #     6 stream skip
    #     4 stream skip
    #     stream !i-> node12-data
    #     stream !d-> node13-data
    #     stream !i-> node14-data
    #     10 dup node16-offsets +<- stack
    #      stream #!h-> node15-data
    #     10 dup node18-offsets +<- stack
    #      stream #!i-> node17-data
    #     10 dup node20-offsets +<- stack
    #      stream #!q-> node19-data
    #     10 dup node22-offsets +<- stack
    #      stream #!H-> node21-data
    #     10 dup node24-offsets +<- stack
    #      stream #!I-> node23-data
    #     10 dup node26-offsets +<- stack
    #      stream #!Q-> node25-data
    #     10 dup node28-offsets +<- stack
    #      stream #!f-> node27-data
    #     10 dup node30-offsets +<- stack
    #      stream #!d-> node29-data
    #     stream !I-> stack dup var_N ! node31-data <- stack
    #     1 stream skip
    #      var_N @ dup node33-offsets +<- stack
    #      stream #!h-> node32-data
    #     1 stream skip
    #      var_N @ dup node35-offsets +<- stack
    #      stream #!i-> node34-data
    #     1 stream skip
    #      var_N @ dup node37-offsets +<- stack
    #      stream #!q-> node36-data
    #     1 stream skip
    #      var_N @ dup node39-offsets +<- stack
    #      stream #!H-> node38-data
    #     1 stream skip
    #      var_N @ dup node41-offsets +<- stack
    #      stream #!I-> node40-data
    #     1 stream skip
    #      var_N @ dup node43-offsets +<- stack
    #      stream #!Q-> node42-data
    #     1 stream skip
    #      var_N @ dup node45-offsets +<- stack
    #      stream #!f-> node44-data
    #     1 stream skip
    #      var_N @ dup node47-offsets +<- stack
    #      stream #!d-> node46-data
    #     6 stream skip
    #     stream !B-> stack dup 255 = if drop stream !I-> stack then dup node48-offsets +<- stack stream #!B-> node49-data
    #     6 stream skip
    #     stream !I-> stack
    #      dup node50-offsets +<- stack
    #     stream #!h-> node51-data
    #     6 stream skip
    #     stream !I-> stack
    #      dup node52-offsets +<- stack
    #     stream #!i-> node53-data
    #     6 stream skip
    #     stream !I-> stack
    #      dup node54-offsets +<- stack
    #     stream #!q-> node55-data
    #     6 stream skip
    #     stream !I-> stack
    #      dup node56-offsets +<- stack
    #     stream #!H-> node57-data
    #     6 stream skip
    #     stream !I-> stack
    #      dup node58-offsets +<- stack
    #     stream #!I-> node59-data
    #     6 stream skip
    #     stream !I-> stack
    #      dup node60-offsets +<- stack
    #     stream #!Q-> node61-data
    #     6 stream skip
    #     stream !I-> stack
    #      dup node62-offsets +<- stack
    #     stream #!f-> node63-data
    #     6 stream skip
    #     stream !I-> stack
    #      dup node64-offsets +<- stack
    #     stream #!d-> node65-data
    #     6 stream skip
    #     stream !I-> stack
    #      dup node66-offsets +<- stack
    #     0 do
    #     stream !B-> stack dup 255 = if drop stream !I-> stack then dup node67-offsets +<- stack stream #!B-> node68-data
    #     loop
    #      stream !B-> stack dup 255 = if drop stream !I-> stack then dup node69-offsets +<- stack stream #!B-> node70-data
    #
    #         loop
    with uproot.open(
        skhep_testdata.data_path("uproot-small-evnt-tree-nosplit.root")
    ) as file:
        branch = file["tree/evt"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[1]["SliceU64"][0] == 1
        # py[-1] == <Event (version 1) at 0x7fecdbf61dc0>
        assert py.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


@pytest.mark.parametrize("is_forth", [False, True])
def test_57(is_forth):
    # see AwkwardForth testing: L, P
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-offsets int64
    #     output node2-data uint8
    #
    #         0 node0-offsets <- stack
    #     0 node1-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         6 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     0 do
    #     stream !B-> stack dup 255 = if drop stream !I-> stack then dup node1-offsets +<- stack stream #!B-> node2-data
    #     loop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root")) as file:
        branch = file["tree/vector_string"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[-1][1] == "two"
        if is_forth:
            assert interp._complete_forth_code is not None
        # py[-1] == <STLVector ['one', 'two'] at 0x7fdeeb3f8d90>
        assert py.layout.form == interp.awkward_form(branch.file)


@pytest.mark.parametrize("is_forth", [False, True])
def test_58(is_forth):
    # see AwkwardForth testing: L, P
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-offsets int64
    #     output node2-data uint8
    #
    #         0 node0-offsets <- stack
    #     0 node1-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         6 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     0 do
    #     stream !B-> stack dup 255 = if drop stream !I-> stack then dup node1-offsets +<- stack stream #!B-> node2-data
    #     loop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root")) as file:
        branch = file["tree/vector_tstring"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak")
        assert py[1][1] == "two"
        if is_forth:
            assert interp._complete_forth_code is not None
        # py[-1] == <STLVector ['one', 'two'] at 0x7f42edc0c0a0>
        assert py.layout.form == interp.awkward_form(branch.file)


@pytest.mark.parametrize("is_forth", [False, True])
def test_59(is_forth):
    # see AwkwardForth testing: P
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-offsets int64
    #     output node2-data int32
    #
    #         0 node0-offsets <- stack
    #     0 node1-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         6 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     0 do
    #     stream !I-> stack
    #      dup node1-offsets +<- stack
    #     stream #!i-> node2-data
    #     loop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root")) as file:
        branch = file["tree/vector_vector_int32"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[-1][1][1] == 2
        if is_forth:
            assert interp._complete_forth_code is not None
        # py[-1] == <STLVector [[1], [1, 2]] at 0x7ff36af3c2b0>
        assert py.layout.form == interp.awkward_form(branch.file)


@pytest.mark.parametrize("is_forth", [False, True])
def test_60(is_forth):
    # see AwkwardForth testing: L, P
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-offsets int64
    #     output node2-offsets int64
    #     output node3-data uint8
    #
    #         0 node0-offsets <- stack
    #     0 node1-offsets <- stack
    #     0 node2-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         6 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     0 do
    #     stream !I-> stack
    #      dup node1-offsets +<- stack
    #     0 do
    #     stream !B-> stack dup 255 = if drop stream !I-> stack then dup node2-offsets +<- stack stream #!B-> node3-data
    #     loop
    #     loop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root")) as file:
        branch = file["tree/vector_vector_string"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[-1][1][1] == "two"
        if is_forth:
            assert interp._complete_forth_code is not None
        # py[-1] == <STLVector [['one'], ['one', 'two']] at 0x7fae23700eb0>
        assert py.layout.form == interp.awkward_form(branch.file)


@pytest.mark.parametrize("is_forth", [False, True])
def test_61(is_forth):
    # see AwkwardForth testing: L, P
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-offsets int64
    #     output node2-offsets int64
    #     output node3-data uint8
    #
    #         0 node0-offsets <- stack
    #     0 node1-offsets <- stack
    #     0 node2-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         6 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     0 do
    #     stream !I-> stack
    #      dup node1-offsets +<- stack
    #     0 do
    #     stream !B-> stack dup 255 = if drop stream !I-> stack then dup node2-offsets +<- stack stream #!B-> node3-data
    #     loop
    #     loop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root")) as file:
        branch = file["tree/vector_vector_tstring"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[-1][1][1] == "two"
        if is_forth:
            assert interp._complete_forth_code is not None
        # py[-1] == <STLVector [['one'], ['one', 'two']] at 0x7f06ad24c460>
        assert py.layout.form == interp.awkward_form(branch.file)


@pytest.mark.parametrize("is_forth", [False, True])
def test_62(is_forth):
    # see AwkwardForth testing: P, Q
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-offsets int64
    #     output node2-data int32
    #
    #         0 node0-offsets <- stack
    #     0 node1-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         6 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     0 do
    #     stream !I-> stack
    #     dup node1-offsets +<- stack
    #     stream #!i-> node2-data
    #     loop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root")) as file:
        branch = file["tree/vector_set_int32"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[-1][1][1] == 2
        if is_forth:
            assert interp._complete_forth_code is not None
        # py[-1] == <STLVector [{1}, {1, 2}] at 0x7f7f97e88880>
        assert py.layout.form == interp.awkward_form(branch.file)


@pytest.mark.parametrize("is_forth", [False, True])
def test_63(is_forth):
    # see AwkwardForth testing: L, P, Q
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-offsets int64
    #     output node2-offsets int64
    #     output node3-data uint8
    #
    #         0 node0-offsets <- stack
    #     0 node1-offsets <- stack
    #     0 node2-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         6 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     0 do
    #     stream !I-> stack
    #     dup node1-offsets +<- stack
    #     0 do
    #     stream !B-> stack dup 255 = if drop stream !I-> stack then dup node2-offsets +<- stack stream #!B-> node3-data
    #     loop
    #     loop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root")) as file:
        branch = file["tree/vector_set_string"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[-1][1][1] == "two"
        if is_forth:
            assert interp._complete_forth_code is not None
        # py[-1] == <STLVector [{'one'}, {'one', 'two'}] at 0x7fb29ded7370>
        assert py.layout.form == interp.awkward_form(branch.file)


@pytest.mark.parametrize("is_forth", [False, True])
def test_64(is_forth):
    # see AwkwardForth testing: Q
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-data int32
    #
    #         0 node0-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         6 stream skip
    #     stream !I-> stack
    #     dup node0-offsets +<- stack
    #     stream #!i-> node1-data
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root")) as file:
        branch = file["tree/set_int32"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[-1][1] == 2
        if is_forth:
            assert interp._complete_forth_code is not None
        # py[-1] == <STLSet {1, 2} at 0x7f47a620e310>
        assert py.layout.form == interp.awkward_form(branch.file)


@pytest.mark.parametrize("is_forth", [False, True])
def test_65(is_forth):
    # see AwkwardForth testing: L, Q
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-offsets int64
    #     output node2-data uint8
    #
    #         0 node0-offsets <- stack
    #     0 node1-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         6 stream skip
    #     stream !I-> stack
    #     dup node0-offsets +<- stack
    #     0 do
    #     stream !B-> stack dup 255 = if drop stream !I-> stack then dup node1-offsets +<- stack stream #!B-> node2-data
    #     loop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root")) as file:
        branch = file["tree/set_string"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[-1][1] == "two"
        if is_forth:
            assert interp._complete_forth_code is not None
        # py[-1] == <STLSet {'one', 'two'} at 0x7f5ffd9ff1c0>
        assert py.layout.form == interp.awkward_form(branch.file)


@pytest.mark.parametrize("is_forth", [False, True])
def test_66(is_forth):
    # see AwkwardForth testing: R
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-data int32
    #     output node2-data int16
    #
    #         0 node0-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         12 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     dup
    #     stream #!i-> node1-data
    #     stream #!h-> node2-data
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root")) as file:
        branch = file["tree/map_int32_int16"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[-1]["1"][1] == 2
        if is_forth:
            assert interp._complete_forth_code is not None
        # py[-1] == <STLMap {1: 1, 2: 2} at 0x7f23be1efbe0>
        assert py.layout.form == interp.awkward_form(branch.file)


@pytest.mark.parametrize("is_forth", [False, True])
def test_67(is_forth):
    # see AwkwardForth testing: P, R
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-data int32
    #     output node2-offsets int64
    #     output node3-data int16
    #
    #         0 node0-offsets <- stack
    #     0 node2-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         12 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     dup
    #     stream #!i-> node1-data
    #     6 stream skip
    #     0 do
    #     stream !I-> stack
    #      dup node2-offsets +<- stack
    #     stream #!h-> node3-data
    #     loop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root")) as file:
        branch = file["tree/map_int32_vector_int16"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[1][1]["1"][1] == 2
        if is_forth:
            assert interp._complete_forth_code is not None
        # py[-1] == <STLMap {1: [1], 2: [1, 2]} at 0x7f899441d2b0>
        assert py.layout.form == interp.awkward_form(branch.file)


@pytest.mark.parametrize("is_forth", [False, True])
def test_68(is_forth):
    # see AwkwardForth testing: L, P, R
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-data int32
    #     output node2-offsets int64
    #     output node3-offsets int64
    #     output node4-data uint8
    #
    #         0 node0-offsets <- stack
    #     0 node2-offsets <- stack
    #     0 node3-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         12 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     dup
    #     stream #!i-> node1-data
    #     6 stream skip
    #     0 do
    #     stream !I-> stack
    #      dup node2-offsets +<- stack
    #     0 do
    #     stream !B-> stack dup 255 = if drop stream !I-> stack then dup node3-offsets +<- stack stream #!B-> node4-data
    #     loop
    #     loop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root")) as file:
        branch = file["tree/map_int32_vector_string"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[1][1]["1"][1] == "two"
        if is_forth:
            assert interp._complete_forth_code is not None
        # py[-1] == <STLMap {1: ['one'], 2: ['one', 'two']} at 0x7fd19d3288b0>
        assert py.layout.form == interp.awkward_form(branch.file)


@pytest.mark.parametrize("is_forth", [False, True])
def test_69(is_forth):
    # see AwkwardForth testing: Q, R
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-data int32
    #     output node2-offsets int64
    #     output node3-data int16
    #
    #         0 node0-offsets <- stack
    #     0 node2-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         12 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     dup
    #     stream #!i-> node1-data
    #     6 stream skip
    #     0 do
    #     stream !I-> stack
    #     dup node2-offsets +<- stack
    #     stream #!h-> node3-data
    #     loop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root")) as file:
        branch = file["tree/map_int32_set_int16"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[1][1]["1"][1] == 2
        if is_forth:
            assert interp._complete_forth_code is not None
        # py[-1] == <STLMap {1: {1}, 2: {1, 2}} at 0x7f2b3f1b5fa0>
        assert py.layout.form == interp.awkward_form(branch.file)


@pytest.mark.parametrize("is_forth", [False, True])
def test_70(is_forth):
    # see AwkwardForth testing: L, Q, R
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-data int32
    #     output node2-offsets int64
    #     output node3-offsets int64
    #     output node4-data uint8
    #
    #         0 node0-offsets <- stack
    #     0 node2-offsets <- stack
    #     0 node3-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         12 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     dup
    #     stream #!i-> node1-data
    #     6 stream skip
    #     0 do
    #     stream !I-> stack
    #     dup node2-offsets +<- stack
    #     0 do
    #     stream !B-> stack dup 255 = if drop stream !I-> stack then dup node3-offsets +<- stack stream #!B-> node4-data
    #     loop
    #     loop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root")) as file:
        branch = file["tree/map_int32_set_string"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[1][1]["1"][1] == "two"
        if is_forth:
            assert interp._complete_forth_code is not None
        # py[-1] == <STLMap {1: {'one'}, 2: {'one', 'two'}} at 0x7f4718b237c0>
        assert py.layout.form == interp.awkward_form(branch.file)


@pytest.mark.parametrize("is_forth", [False, True])
def test_71(is_forth):
    # see AwkwardForth testing: L, R
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-offsets int64
    #     output node2-data uint8
    #     output node3-data int16
    #
    #         0 node0-offsets <- stack
    #     0 node1-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         12 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     6 stream skip
    #     dup 0 do
    #     stream !B-> stack dup 255 = if drop stream !I-> stack then dup node1-offsets +<- stack stream #!B-> node2-data
    #     loop
    #     stream #!h-> node3-data
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root")) as file:
        branch = file["tree/map_string_int16"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[1][1]["1"] == 2
        if is_forth:
            assert interp._complete_forth_code is not None
        # py[-1] == <STLMap {'one': 1, 'two': 2} at 0x7f4179bed1f0>
        assert py.layout.form == interp.awkward_form(branch.file)


@pytest.mark.parametrize("is_forth", [False, True])
def test_72(is_forth):
    # see AwkwardForth testing: L, P, R
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-offsets int64
    #     output node2-data uint8
    #     output node3-offsets int64
    #     output node4-data int16
    #
    #         0 node0-offsets <- stack
    #     0 node1-offsets <- stack
    #     0 node3-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         12 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     6 stream skip
    #     dup 0 do
    #     stream !B-> stack dup 255 = if drop stream !I-> stack then dup node1-offsets +<- stack stream #!B-> node2-data
    #     loop
    #     6 stream skip
    #     0 do
    #     stream !I-> stack
    #      dup node3-offsets +<- stack
    #     stream #!h-> node4-data
    #     loop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root")) as file:
        branch = file["tree/map_string_vector_int16"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[1][1]["1"][1] == 2
        if is_forth:
            assert interp._complete_forth_code is not None
        # py[-1] == <STLMap {'one': [1], 'two': [1, 2]} at 0x7f26376e9b20>
        assert py.layout.form == interp.awkward_form(branch.file)


@pytest.mark.parametrize("is_forth", [False, True])
def test_73(is_forth):
    # see AwkwardForth testing: L, P, R
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-offsets int64
    #     output node2-data uint8
    #     output node3-offsets int64
    #     output node4-offsets int64
    #     output node5-data uint8
    #
    #         0 node0-offsets <- stack
    #     0 node1-offsets <- stack
    #     0 node3-offsets <- stack
    #     0 node4-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         12 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     6 stream skip
    #     dup 0 do
    #     stream !B-> stack dup 255 = if drop stream !I-> stack then dup node1-offsets +<- stack stream #!B-> node2-data
    #     loop
    #     6 stream skip
    #     0 do
    #     stream !I-> stack
    #      dup node3-offsets +<- stack
    #     0 do
    #     stream !B-> stack dup 255 = if drop stream !I-> stack then dup node4-offsets +<- stack stream #!B-> node5-data
    #     loop
    #     loop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root")) as file:
        branch = file["tree/map_string_vector_string"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[1][1]["1"][1] == "two"
        if is_forth:
            assert interp._complete_forth_code is not None
        # py[-1] == <STLMap {'one': ['one'], 'two': ['one', 'two']} at 0x7f3e45d34910>
        assert py.layout.form == interp.awkward_form(branch.file)


@pytest.mark.parametrize("is_forth", [False, True])
def test_74(is_forth):
    # see AwkwardForth testing: L, Q, R
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-offsets int64
    #     output node2-data uint8
    #     output node3-offsets int64
    #     output node4-data int16
    #
    #         0 node0-offsets <- stack
    #     0 node1-offsets <- stack
    #     0 node3-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         12 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     6 stream skip
    #     dup 0 do
    #     stream !B-> stack dup 255 = if drop stream !I-> stack then dup node1-offsets +<- stack stream #!B-> node2-data
    #     loop
    #     6 stream skip
    #     0 do
    #     stream !I-> stack
    #     dup node3-offsets +<- stack
    #     stream #!h-> node4-data
    #     loop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root")) as file:
        branch = file["tree/map_string_set_int16"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[1][1]["1"][1] == 2
        if is_forth:
            assert interp._complete_forth_code is not None
        # py[-1] == <STLMap {'one': {1}, 'two': {1, 2}} at 0x7f5a94b1c760>
        assert py.layout.form == interp.awkward_form(branch.file)


@pytest.mark.parametrize("is_forth", [False, True])
def test_75(is_forth):
    # see AwkwardForth testing: L, Q, R
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-offsets int64
    #     output node2-data uint8
    #     output node3-offsets int64
    #     output node4-offsets int64
    #     output node5-data uint8
    #
    #         0 node0-offsets <- stack
    #     0 node1-offsets <- stack
    #     0 node3-offsets <- stack
    #     0 node4-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         12 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     6 stream skip
    #     dup 0 do
    #     stream !B-> stack dup 255 = if drop stream !I-> stack then dup node1-offsets +<- stack stream #!B-> node2-data
    #     loop
    #     6 stream skip
    #     0 do
    #     stream !I-> stack
    #     dup node3-offsets +<- stack
    #     0 do
    #     stream !B-> stack dup 255 = if drop stream !I-> stack then dup node4-offsets +<- stack stream #!B-> node5-data
    #     loop
    #     loop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root")) as file:
        branch = file["tree/map_string_set_string"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        if is_forth:
            assert interp._complete_forth_code is not None
        assert py[1][1]["1"][1] == "two"
        # py[-1] == <STLMap {'one': {'one'}, 'two': {'one', 'two'}} at 0x7f1a1e95aa90>
        assert py.layout.form == interp.awkward_form(branch.file)


@pytest.mark.parametrize("is_forth", [False, True])
def test_76(is_forth):
    # see AwkwardForth testing: P, R
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-data int32
    #     output node2-offsets int64
    #     output node3-offsets int64
    #     output node4-data int16
    #
    #         0 node0-offsets <- stack
    #     0 node2-offsets <- stack
    #     0 node3-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         12 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     dup
    #     stream #!i-> node1-data
    #     6 stream skip
    #     0 do
    #     stream !I-> stack
    #      dup node2-offsets +<- stack
    #     0 do
    #     stream !I-> stack
    #      dup node3-offsets +<- stack
    #     stream #!h-> node4-data
    #     loop
    #     loop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root")) as file:
        branch = file["tree/map_int32_vector_vector_int16"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        if is_forth:
            assert interp._complete_forth_code is not None
        assert py[1][1]["1"][1][1] == 2
        # py[-1] == <STLMap {1: [[1]], 2: [[1], [1, 2]]} at 0x7fc98bd7ec10>
        assert py.layout.form == interp.awkward_form(branch.file)


@pytest.mark.parametrize("is_forth", [False, True])
def test_77(is_forth):
    # see AwkwardForth testing: P, Q, R
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-data int32
    #     output node2-offsets int64
    #     output node3-offsets int64
    #     output node4-data int16
    #
    #         0 node0-offsets <- stack
    #     0 node2-offsets <- stack
    #     0 node3-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         12 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     dup
    #     stream #!i-> node1-data
    #     6 stream skip
    #     0 do
    #     stream !I-> stack
    #      dup node2-offsets +<- stack
    #     0 do
    #     stream !I-> stack
    #     dup node3-offsets +<- stack
    #     stream #!h-> node4-data
    #     loop
    #     loop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root")) as file:
        branch = file["tree/map_int32_vector_set_int16"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[1][1]["1"][1][1] == 2
        if is_forth:
            assert interp._complete_forth_code is not None
        # py[-1] == <STLMap {1: [{1}], 2: [{1}, {1, 2}]} at 0x7fcf9b191610>
        assert py.layout.form == interp.awkward_form(branch.file)


@pytest.mark.parametrize("is_forth", [False, True])
def test_78(is_forth):
    # see AwkwardForth testing: L, R
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-offsets int64
    #     output node2-data uint8
    #     output node3-offsets int64
    #     output node4-data uint8
    #
    #         0 node0-offsets <- stack
    #     0 node1-offsets <- stack
    #     0 node3-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         12 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     6 stream skip
    #     dup 0 do
    #     stream !B-> stack dup 255 = if drop stream !I-> stack then dup node1-offsets +<- stack stream #!B-> node2-data
    #     loop
    #     6 stream skip
    #     0 do
    #     stream !B-> stack dup 255 = if drop stream !I-> stack then dup node3-offsets +<- stack stream #!B-> node4-data
    #     loop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root")) as file:
        branch = file["tree/map_string_string"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        if is_forth:
            assert interp._complete_forth_code is not None
        assert py[1, 1, "1"] == "TWO"
        # py[-1] == <STLMap {'one': 'ONE', 'two': 'TWO'} at 0x7f887b27cb20>
        assert py.layout.form == interp.awkward_form(branch.file)


@pytest.mark.parametrize("is_forth", [False, True])
def test_79(is_forth):
    # see AwkwardForth testing: L, R
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-offsets int64
    #     output node2-data uint8
    #     output node3-offsets int64
    #     output node4-data uint8
    #
    #         0 node0-offsets <- stack
    #     0 node1-offsets <- stack
    #     0 node3-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         12 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     6 stream skip
    #     dup 0 do
    #     stream !B-> stack dup 255 = if drop stream !I-> stack then dup node1-offsets +<- stack stream #!B-> node2-data
    #     loop
    #     0 do
    #     stream !B-> stack dup 255 = if drop stream !I-> stack then dup node3-offsets +<- stack stream #!B-> node4-data
    #     loop
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root")) as file:
        branch = file["tree/map_string_tstring"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[1, 1, "1"] == "TWO"
        # py[-1] == <STLMap {'one': 'ONE', 'two': 'TWO'} at 0x7f4c4e527610>
        assert py.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None


@pytest.mark.parametrize("is_forth", [False, True])
def test_80(is_forth):
    # see AwkwardForth testing: P
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-offsets int64
    #     output node1-offsets int64
    #     output node2-data float64
    #
    #         0 node0-offsets <- stack
    #     0 node1-offsets <- stack
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         6 stream skip
    #     stream !I-> stack
    #      dup node0-offsets +<- stack
    #     0 do
    #     stream !I-> stack
    #      dup node1-offsets +<- stack
    #     stream #!d-> node2-data
    #     loop
    #
    #         loop
    with uproot.open(
        skhep_testdata.data_path("uproot-vectorVectorDouble.root")
    ) as file:
        branch = file["t/x"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak")
        if is_forth:
            assert interp._complete_forth_code is not None
        assert py.tolist() == [
            [],
            [[], []],
            [[10.0], [], [10.0, 20.0]],
            [[20.0, -21.0, -22.0]],
            [[200.0], [-201.0], [202.0]],
        ]
        # py[-1] == <STLVector [[], []] at 0x7f046a6951f0>


@pytest.mark.parametrize("is_forth", [False, True])
def test_81(is_forth):
    # see AwkwardForth testing: (none?)
    # Expected AwkwardForth code:
    #     input stream
    #         input byteoffsets
    #         input bytestops
    #         output node0-data float64
    #     output node1-data float64
    #
    #
    #         0 do
    #         byteoffsets I-> stack
    #         stream seek
    #         0 stream skip
    #     6 stream skip
    #     10 stream skip
    #     stream !d-> node0-data
    #     stream !d-> node1-data
    #
    #         loop
    with uproot.open(skhep_testdata.data_path("uproot-HZZ-objects.root")) as file:
        branch = file["events/MET"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        if is_forth:
            assert interp._complete_forth_code is not None
        assert py[0]["fY"] == pytest.approx(2.5636332035064697)
        # py[-1] == <STLVector [[], []] at 0x7f046a6951f0>
        assert py.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None
