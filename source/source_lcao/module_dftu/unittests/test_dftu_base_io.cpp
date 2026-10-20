#include "gtest/gtest.h"

#include "source_pw/module_pwdft/dftu_base_io.h"

#include "source_cell/unitcell.h"
#include "source_pw/module_pwdft/dftu_base.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef __MPI
#include "mpi.h"
#endif

// Focused tests for the occupation-matrix filename helpers and gating logic
// implemented in dftu_base_io.cpp.

TEST(DFTUBaseIoTest, IonStepGateZeroFreqDisablesNumberedFiles)
{
    const DFTU_BASE::OccmatOutputCfg cfg{0, 10, 10, true, 1};
    EXPECT_FALSE(DFTU_BASE::is_ion_step_output_step(0, cfg));
    EXPECT_FALSE(DFTU_BASE::is_ion_step_output_step(1, cfg));
    EXPECT_FALSE(DFTU_BASE::is_ion_step_output_step(2, cfg));
}

TEST(DFTUBaseIoTest, IonStepGateDivisibleIonStepsOnly)
{
    const DFTU_BASE::OccmatOutputCfg cfg{2, 10, 10, true, 1};
    EXPECT_TRUE(DFTU_BASE::is_ion_step_output_step(0, cfg));
    EXPECT_FALSE(DFTU_BASE::is_ion_step_output_step(1, cfg));
    EXPECT_TRUE(DFTU_BASE::is_ion_step_output_step(2, cfg));
    EXPECT_FALSE(DFTU_BASE::is_ion_step_output_step(3, cfg));
    EXPECT_TRUE(DFTU_BASE::is_ion_step_output_step(4, cfg));
}

TEST(DFTUBaseIoTest, ElecGatePeriodicTrigger)
{
    const DFTU_BASE::OccmatOutputCfg cfg{1, 2, 10, true, 1};
    EXPECT_FALSE(DFTU_BASE::is_elec_snapshot_trigger(1, false, cfg));
    EXPECT_TRUE(DFTU_BASE::is_elec_snapshot_trigger(2, false, cfg));
    EXPECT_FALSE(DFTU_BASE::is_elec_snapshot_trigger(3, false, cfg));
    EXPECT_TRUE(DFTU_BASE::is_elec_snapshot_trigger(4, false, cfg));
}

TEST(DFTUBaseIoTest, ElecGateScfNmaxTrigger)
{
    const DFTU_BASE::OccmatOutputCfg cfg{1, 10, 10, true, 1};
    EXPECT_FALSE(DFTU_BASE::is_elec_snapshot_trigger(1, false, cfg));
    EXPECT_FALSE(DFTU_BASE::is_elec_snapshot_trigger(9, false, cfg));
    EXPECT_TRUE(DFTU_BASE::is_elec_snapshot_trigger(10, false, cfg));
}

TEST(DFTUBaseIoTest, ElecGateConvergenceTrigger)
{
    const DFTU_BASE::OccmatOutputCfg cfg{1, 10, 10, true, 1};
    EXPECT_TRUE(DFTU_BASE::is_elec_snapshot_trigger(3, true, cfg));
    EXPECT_TRUE(DFTU_BASE::is_elec_snapshot_trigger(7, true, cfg));
}

TEST(DFTUBaseIoTest, IonStepFilenameStartsFromOne)
{
    const std::string fn0 = DFTU_BASE::gen_ion_step_occ_mat_filename("OUT.ABACUS/", 0);
    EXPECT_EQ(fn0, "OUT.ABACUS/occ_matg1.txt");

    const std::string fn4 = DFTU_BASE::gen_ion_step_occ_mat_filename("OUT.ABACUS/", 4);
    EXPECT_EQ(fn4, "OUT.ABACUS/occ_matg5.txt");
}

TEST(DFTUBaseIoTest, LcaoMethodSelectsKernelStorage)
{
    EXPECT_EQ(DFTU_BASE::lcao_soc_layout(1), DFTU_BASE::SOC_LAYOUT_PAULI);
    EXPECT_EQ(DFTU_BASE::lcao_soc_layout(2), DFTU_BASE::SOC_LAYOUT_SPIN_BASIS_REAL);
}

class SocIoTestDftu : public Plus_U_Base
{
  public:
    SocIoTestDftu()
    {
        this->l_channel = {1};
    }
};

// Exercise both storage contracts, including nonzero y for the new method.
TEST(DFTUBaseIoTest, LcaoSocReadableRoundTrip)
{
    UnitCell cell;
    cell.ntype = 1;
    cell.nat = 1;
    cell.atoms = new Atom[1];
    cell.atoms[0].na = 1;
    cell.atoms[0].nwl = 1;
    cell.atoms[0].label = "X";
    cell.iat2it = {0};
    cell.itia2iat.create(1, 1);
    cell.itia2iat(0, 0) = 0;
    for (int method = 1; method <= 2; ++method)
    {
        SocIoTestDftu dftu;
        auto& data = dftu.occmat().data();
        data.resize(1);
        data[0].resize(2);
        data[0][1].resize(1);
        data[0][1][0].create(6, 6);
        std::vector<double> expected(36);
        for (int is = 0; is < 4; ++is)
        {
            for (int row = 0; row < 3; ++row)
            {
                for (int col = 0; col < 3; ++col)
                {
                    double value = 0.002 * (is + 1) * (row + col + 1);
                    if (row == col)
                    {
                        value += 0.1 * (is + 1);
                    }
                    if (method == 1)
                    {
                        data[0][1][0].c[is * 9 + row * 3 + col] = value;
                    }
                    else
                    {
                        // A symmetric real spinor matrix: both offdiagonal
                        // spin blocks must agree in the legacy representation.
                        const int spinrow = is / 2;
                        const int spincol = is % 2;
                        if (is == 2)
                        {
                            value = 0.004 * (row + col + 1);
                            if (row == col)
                            {
                                value += 0.2;
                            }
                        }
                        data[0][1][0](spinrow * 3 + row, spincol * 3 + col) = value;
                    }
                }
            }
        }
        std::copy(data[0][1][0].c, data[0][1][0].c + 36, expected.begin());
        const auto layout = DFTU_BASE::lcao_soc_layout(method);
        std::ostringstream text;
        DFTU_BASE::write_occup_m(dftu, cell, text, true, 4, 2,
                                 DFTU_BASE::OCMAT_FMT_READABLE, layout);
        if (method == 1)
        {
            EXPECT_NE(text.str().find("mag 0.63600000 0.95400000 1.27200000"), std::string::npos);
        }
        const std::string path = testing::TempDir() + "lcao_soc_roundtrip_" + std::to_string(method) + ".txt";
        {
            std::ofstream file(path);
            file << text.str();
            ASSERT_TRUE(file.good());
        }
        data[0][1][0].zero_out();
        const auto& channels = dftu.get_l_channel_vec();
        DFTU_BASE::read_occup_m(cell, dftu.occmat(), channels, path, 4, 2, layout);
        for (int k = 0; k < 36; ++k)
        {
            EXPECT_NEAR(data[0][1][0].c[k], expected[k], 1.1e-8) << "method=" << method << " index=" << k;
        }
        EXPECT_EQ(std::remove(path.c_str()), 0);
    }
    delete[] cell.atoms;
    cell.atoms = nullptr;
}

int main(int argc, char** argv)
{
#ifdef __MPI
    MPI_Init(&argc, &argv);
#endif
    testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
#ifdef __MPI
    MPI_Finalize();
#endif
    return result;
}
