#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "source_cell/cal_ux.h"
#include "source_cell/read_orb.h"
#include "source_cell/read_pp_ucell.h"
#include "source_cell/read_stru.h"
#include "source_cell/cell_tools.h"
#include "source_cell/print_cell.h"
#include "source_cell/read_atoms_helper.h"
#include "memory"
#include "source_base/global_variable.h"
#include "source_base/mathzone.h"
#include "prepare_unitcell.h"
#include "source_cell/update_cell.h"
#include <fstream>
#include <algorithm>
#include <cstdio>
#include <streambuf>
#include <valarray>
#include <vector>

Magnetism::Magnetism()
{
    this->tot_mag = 0.0;
    this->abs_mag = 0.0;
}
Magnetism::~Magnetism()
{
}

class PrintCellTest : public testing::Test
{
protected:
    std::unique_ptr<UnitCell> ucell;
    void SetUp() override
    {
        // nothing to do here, each test sets up its own ucell
    }
    void TearDown() override
    {
        // cleanup generated files
    }
};

TEST_F(PrintCellTest, PrintSTRU_nspin1)
{
    UcellTestPrepare utp = UcellTestLib["C1H2-Index"];
    ucell = utp.SetUcellInfo();
    std::string fn = "C1H2_STRU_nspin1";

    unitcell::print_stru_file(*ucell, ucell->atoms, ucell->latvec,
                              fn, "", 1, false, false, false, false, 0, ModuleBase::matrix(), false);
    std::ifstream ifs;
    ifs.open(fn);
    std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    EXPECT_THAT(str, testing::HasSubstr("ATOMIC_SPECIES"));
    EXPECT_THAT(str, testing::HasSubstr("C  12.0000 C.upf upf201"));
    EXPECT_THAT(str, testing::HasSubstr("H   1.0000 H.upf upf201"));
    EXPECT_THAT(str, testing::HasSubstr("LATTICE_CONSTANT"));
    // lat0 is always printed as one Angstrom in Bohr
    EXPECT_THAT(str, testing::HasSubstr(
        "1.8897268778 # in Bohr (= 1 Angstrom); lattice vectors below are in Angstrom"));
    EXPECT_THAT(str, testing::HasSubstr("LATTICE_VECTORS # in Angstrom"));
    EXPECT_THAT(str, testing::HasSubstr("ATOMIC_POSITIONS"));
    // positions are always emitted as Cartesian_angstrom
    EXPECT_THAT(str, testing::HasSubstr("Cartesian_angstrom # positions in Angstrom"));
    EXPECT_THAT(str, testing::HasSubstr("C #label"));
    EXPECT_THAT(str, testing::HasSubstr("0.0000 #magnetism (default, overridden by per-atom mag below)"));
    EXPECT_THAT(str, testing::HasSubstr("1 #number of atoms"));
    // single-space separated coordinates
    EXPECT_THAT(str, testing::HasSubstr("0.9999996019 0.9999996019 0.9999996019 m 1 1 1"));
    EXPECT_THAT(str, testing::HasSubstr("H #label"));
    EXPECT_THAT(str, testing::HasSubstr("0.0000 #magnetism (default, overridden by per-atom mag below)"));
    EXPECT_THAT(str, testing::HasSubstr("2 #number of atoms"));
    EXPECT_THAT(str, testing::HasSubstr("1.4999994028 1.4999994028 1.4999994028 m 0 0 0"));
    EXPECT_THAT(str, testing::HasSubstr("0.4999998009 0.4999998009 0.4999998009 m 0 0 1"));
    // No force output when force matrix is empty
    EXPECT_THAT(str, testing::Not(testing::HasSubstr(" f ")));
    ifs.close();
    remove(fn.c_str());
}

TEST_F(PrintCellTest, PrintSTRU_zero_force_computed)
{
    UcellTestPrepare utp = UcellTestLib["C1H2-Index"];
    ucell = utp.SetUcellInfo();
    std::string fn = "C1H2_STRU_zero_force";

    // has_force=true with an all-zero force matrix: genuinely computed zero
    // forces must still be emitted, unlike "not computed" which omits them.
    ModuleBase::matrix force(3, 3);
    force.zero_out();

    unitcell::print_stru_file(*ucell, ucell->atoms, ucell->latvec,
                              fn, "", 2, false, false, false, false, 0, force, true);
    std::ifstream ifs;
    ifs.open(fn);
    std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());

    EXPECT_THAT(str, testing::HasSubstr("Cartesian_angstrom # positions in Angstrom, forces in eV/Angstrom"));
    EXPECT_THAT(str, testing::HasSubstr(" f 0.000000 0.000000 0.000000"));
    ifs.close();
    remove(fn.c_str());
}

TEST_F(PrintCellTest, PrintSTRU_force_matrix_without_flag)
{
    UcellTestPrepare utp = UcellTestLib["C1H2-Index"];
    ucell = utp.SetUcellInfo();
    std::string fn = "C1H2_STRU_flag_off";

    // Force matrix is correctly sized but has_force=false: the flag takes
    // priority and no f fields are emitted.
    ModuleBase::matrix force(3, 3);
    force(0, 0) = 0.1; force(0, 1) = 0.2; force(0, 2) = 0.3;

    unitcell::print_stru_file(*ucell, ucell->atoms, ucell->latvec,
                              fn, "", 2, false, false, false, false, 0, force, false);
    std::ifstream ifs;
    ifs.open(fn);
    std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());

    EXPECT_THAT(str, testing::Not(testing::HasSubstr(" f ")));
    ifs.close();
    remove(fn.c_str());
}

TEST_F(PrintCellTest, PrintSTRU_nspin2_no_force)
{
    UcellTestPrepare utp = UcellTestLib["C1H2-Index"];
    ucell = utp.SetUcellInfo();
    std::string fn = "C1H2_STRU_nspin2";

    unitcell::print_stru_file(*ucell, ucell->atoms, ucell->latvec,
                              fn, "", 2, true, false, false, false, 0, ModuleBase::matrix(), false);
    std::ifstream ifs;
    ifs.open(fn);
    std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    EXPECT_THAT(str, testing::HasSubstr("ATOMIC_POSITIONS"));
    // positions are always Cartesian_angstrom, even without forces
    EXPECT_THAT(str, testing::HasSubstr("Cartesian_angstrom # positions in Angstrom\n"));
    EXPECT_THAT(str, testing::HasSubstr("C #label"));
    EXPECT_THAT(str, testing::HasSubstr("0.0000 #magnetism (default, overridden by per-atom mag below)"));
    EXPECT_THAT(str,
                testing::HasSubstr("0.9999996019 0.9999996019 0.9999996019 m 1 1 1 v 0.1000000000 0.1000000000 0.1000000000 mag 0.0000"));
    EXPECT_THAT(str, testing::HasSubstr("H #label"));
    EXPECT_THAT(str,
                testing::HasSubstr("1.4999994028 1.4999994028 1.4999994028 m 0 0 0 v 0.1000000000 0.1000000000 0.1000000000 mag 0.0000"));
    EXPECT_THAT(str,
                testing::HasSubstr("0.4999998009 0.4999998009 0.4999998009 m 0 0 1 v 0.1000000000 0.1000000000 0.1000000000 mag 0.0000"));
    // No force output when force matrix is empty
    EXPECT_THAT(str, testing::Not(testing::HasSubstr(" f ")));
    ifs.close();
    remove(fn.c_str());
}

TEST_F(PrintCellTest, PrintSTRU_nspin2_with_mag)
{
    UcellTestPrepare utp = UcellTestLib["C1H2-Index"];
    ucell = utp.SetUcellInfo();
    std::string fn = "C1H2_STRU_nspin2_mag";

    ucell->atoms[0].mag[0] = 1.5;   // C
    ucell->atoms[1].mag[0] = -0.5;  // H1
    ucell->atoms[1].mag[1] = 2.0;   // H2
    unitcell::print_stru_file(*ucell, ucell->atoms, ucell->latvec,
                              fn, "", 2, false, false, false, false, 0, ModuleBase::matrix(), false);
    std::ifstream ifs;
    ifs.open(fn);
    std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    EXPECT_THAT(str, testing::HasSubstr("C #label"));
    EXPECT_THAT(str, testing::HasSubstr("1.5000 #magnetism (default, overridden by per-atom mag below)"));
    EXPECT_THAT(str,
                testing::HasSubstr("0.9999996019 0.9999996019 0.9999996019 m 1 1 1 mag 1.5000"));
    EXPECT_THAT(str, testing::HasSubstr("H #label"));
    EXPECT_THAT(str, testing::HasSubstr("-0.5000 #magnetism (default, overridden by per-atom mag below)"));
    EXPECT_THAT(str,
                testing::HasSubstr("1.4999994028 1.4999994028 1.4999994028 m 0 0 0 mag -0.5000"));
    EXPECT_THAT(str,
                testing::HasSubstr("0.4999998009 0.4999998009 0.4999998009 m 0 0 1 mag 2.0000"));
    ifs.close();
    remove(fn.c_str());
}

TEST_F(PrintCellTest, PrintSTRU_nspin2_mulliken)
{
    UcellTestPrepare utp = UcellTestLib["C1H2-Index"];
    ucell = utp.SetUcellInfo();
    std::string fn = "C1H2_STRU_nspin2_mulliken";

    ucell->descriptor_file = "__unittest_numerical_descriptor__";
    ucell->orbital_fn[0] = "__unittest_orbital_fn_0__";
    ucell->orbital_fn[1] = "__unittest_orbital_fn_1__";
    ucell->atomic_magnetism.source = AtomicMagnetism::Source::mulliken;
    ucell->atomic_magnetism.ionic_step = 0;
    ucell->atomic_magnetism.scf_converged = true;
    ucell->atomic_magnetism.values = {{-1, 0.5}, {-1, 0.4}, {-1, 0.3}};
    unitcell::print_stru_file(*ucell, ucell->atoms, ucell->latvec,
                              fn, "", 2, false, true, true, true, 0, ModuleBase::matrix(), false);
    std::ifstream ifs;
    ifs.open(fn);
    std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    EXPECT_THAT(str, testing::HasSubstr("NUMERICAL_ORBITAL"));
    EXPECT_THAT(str, testing::HasSubstr("__unittest_orbital_fn_0__"));
    EXPECT_THAT(str, testing::HasSubstr("__unittest_orbital_fn_1__"));
    EXPECT_THAT(str, testing::HasSubstr("NUMERICAL_DESCRIPTOR"));
    EXPECT_THAT(str, testing::HasSubstr("__unittest_numerical_descriptor__"));
    EXPECT_THAT(str,
                testing::HasSubstr("0.9999996019 0.9999996019 0.9999996019 m 1 1 1 mag 0.5000"));
    EXPECT_THAT(str,
                testing::HasSubstr("1.4999994028 1.4999994028 1.4999994028 m 0 0 0 mag 0.4000"));
    EXPECT_THAT(str,
                testing::HasSubstr("0.4999998009 0.4999998009 0.4999998009 m 0 0 1 mag 0.3000"));
    ifs.close();
    remove(fn.c_str());
}

TEST_F(PrintCellTest, PrintSTRU_nspin4_initial_mag)
{
    UcellTestPrepare utp = UcellTestLib["C1H2-Index"];
    ucell = utp.SetUcellInfo();
    std::string fn = "C1H2_STRU_nspin4";

    ucell->atoms[0].m_loc_[0].set(1.0, 0.0, 0.0);   // C
    ucell->atoms[1].m_loc_[0].set(0.0, 1.0, 0.0);   // H1
    ucell->atoms[1].m_loc_[1].set(0.0, 0.0, 1.0);   // H2
    unitcell::print_stru_file(*ucell, ucell->atoms, ucell->latvec,
                              fn, "", 4, false, false, false, false, 0, ModuleBase::matrix(), false);
    std::ifstream ifs;
    ifs.open(fn);
    std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    EXPECT_THAT(str, testing::HasSubstr("C #label"));
    EXPECT_THAT(str, testing::HasSubstr("1.0000 #magnetism (default, overridden by per-atom mag below)"));
    EXPECT_THAT(str,
                testing::HasSubstr("0.9999996019 0.9999996019 0.9999996019 m 1 1 1 mag 1.0000 0.0000 0.0000"));
    EXPECT_THAT(str, testing::HasSubstr("H #label"));
    EXPECT_THAT(str, testing::HasSubstr("1.0000 #magnetism (default, overridden by per-atom mag below)"));
    EXPECT_THAT(str,
                testing::HasSubstr("1.4999994028 1.4999994028 1.4999994028 m 0 0 0 mag 0.0000 1.0000 0.0000"));
    EXPECT_THAT(str,
                testing::HasSubstr("0.4999998009 0.4999998009 0.4999998009 m 0 0 1 mag 0.0000 0.0000 1.0000"));
    ifs.close();
    remove(fn.c_str());
}

TEST_F(PrintCellTest, PrintSTRU_nspin4_mulliken)
{
    UcellTestPrepare utp = UcellTestLib["C1H2-Index"];
    ucell = utp.SetUcellInfo();
    std::string fn = "C1H2_STRU_nspin4_mulliken";

    ucell->atomic_magnetism.source = AtomicMagnetism::Source::mulliken;
    ucell->atomic_magnetism.ionic_step = 0;
    ucell->atomic_magnetism.scf_converged = true;
    ucell->atomic_magnetism.values = {{-1, 0.5, 0.1, 0.2}, {-1, 0.4, 0.3, 0.4}, {-1, 0.3, 0.5, 0.6}};
    unitcell::print_stru_file(*ucell, ucell->atoms, ucell->latvec,
                              fn, "", 4, false, true, false, false, 0, ModuleBase::matrix(), false);
    std::ifstream ifs;
    ifs.open(fn);
    std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    EXPECT_THAT(str,
                testing::HasSubstr("0.9999996019 0.9999996019 0.9999996019 m 1 1 1 mag 0.5000 0.1000 0.2000"));
    EXPECT_THAT(str,
                testing::HasSubstr("1.4999994028 1.4999994028 1.4999994028 m 0 0 0 mag 0.4000 0.3000 0.4000"));
    EXPECT_THAT(str,
                testing::HasSubstr("0.4999998009 0.4999998009 0.4999998009 m 0 0 1 mag 0.3000 0.5000 0.6000"));
    ifs.close();
    remove(fn.c_str());
}

TEST_F(PrintCellTest, PrintSTRU_with_force)
{
    UcellTestPrepare utp = UcellTestLib["C1H2-Index"];
    ucell = utp.SetUcellInfo();
    std::string fn = "C1H2_STRU_force";

    // Create force matrix: nat=3, nc=3 (internal unit Ry/Bohr)
    ModuleBase::matrix force(3, 3);
    force(0, 0) = 0.1; force(0, 1) = 0.2; force(0, 2) = 0.3;    // C
    force(1, 0) = -0.1; force(1, 1) = -0.2; force(1, 2) = -0.3; // H1
    force(2, 0) = 0.05; force(2, 1) = 0.15; force(2, 2) = -0.25; // H2

    unitcell::print_stru_file(*ucell, ucell->atoms, ucell->latvec,
                              fn, "", 2, false, false, false, false, 0, force, true);
    std::ifstream ifs;
    ifs.open(fn);
    std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());

    // Positions are always Cartesian_angstrom; forces add the f fields
    EXPECT_THAT(str, testing::HasSubstr("Cartesian_angstrom # positions in Angstrom, forces in eV/Angstrom"));

    // Internal Cartesian tau is in units of lat0 (Bohr); expected Angstrom = tau * lat0 * BOHR_TO_A
    const double pos_conv = ucell->lat0 * ModuleBase::BOHR_TO_A;
    // Internal force is Ry/Bohr; expected eV/Angstrom = f * Ry_to_eV / BOHR_TO_A
    const double force_conv = ModuleBase::Ry_to_eV / ModuleBase::BOHR_TO_A;

    auto fmt3 = [](double a, double b, double c) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "%.10f %.10f %.10f", a, b, c);
        return std::string(buf);
    };
    auto fmtf = [](double a, double b, double c) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), " f %.6f %.6f %.6f", a, b, c);
        return std::string(buf);
    };

    // C (iat=0): m 1 1 1
    EXPECT_THAT(str, testing::HasSubstr(
        fmt3(ucell->atoms[0].tau[0].x * pos_conv,
             ucell->atoms[0].tau[0].y * pos_conv,
             ucell->atoms[0].tau[0].z * pos_conv) + " m 1 1 1"));
    EXPECT_THAT(str, testing::HasSubstr(
        fmtf(force(0, 0) * force_conv, force(0, 1) * force_conv, force(0, 2) * force_conv)));

    // H1 (iat=1): m 0 0 0
    EXPECT_THAT(str, testing::HasSubstr(
        fmt3(ucell->atoms[1].tau[0].x * pos_conv,
             ucell->atoms[1].tau[0].y * pos_conv,
             ucell->atoms[1].tau[0].z * pos_conv) + " m 0 0 0"));
    EXPECT_THAT(str, testing::HasSubstr(
        fmtf(force(1, 0) * force_conv, force(1, 1) * force_conv, force(1, 2) * force_conv)));

    // H2 (iat=2): m 0 0 1
    EXPECT_THAT(str, testing::HasSubstr(
        fmt3(ucell->atoms[1].tau[1].x * pos_conv,
             ucell->atoms[1].tau[1].y * pos_conv,
             ucell->atoms[1].tau[1].z * pos_conv) + " m 0 0 1"));
    EXPECT_THAT(str, testing::HasSubstr(
        fmtf(force(2, 0) * force_conv, force(2, 1) * force_conv, force(2, 2) * force_conv)));

    ifs.close();
    remove(fn.c_str());
}

TEST_F(PrintCellTest, PrintCell)
{
    UcellTestPrepare utp = UcellTestLib["C1H2-Index"];
    ucell = utp.SetUcellInfo();
    std::ofstream ofs;
    ofs.open("print_cell_test.log");
    unitcell::print_cell(*ucell, ofs);
    ofs.close();
    std::ifstream ifs;
    ifs.open("print_cell_test.log");
    std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    EXPECT_THAT(str, testing::HasSubstr("latName = bcc"));
    EXPECT_THAT(str, testing::HasSubstr("ntype = 2"));
    EXPECT_THAT(str, testing::HasSubstr("nat = 3"));
    EXPECT_THAT(str, testing::HasSubstr("GGT :"));
    EXPECT_THAT(str, testing::HasSubstr("omega = 6748.33"));
    remove("print_cell_test.log");
}
TEST_F(PrintCellTest, PrintTauDirect)
{
    UcellTestPrepare utp = UcellTestLib["C1H2-Index"];
    ucell = utp.SetUcellInfo();
    EXPECT_EQ(ucell->Coordinate, "Direct");

    // open a file
    std::ofstream ofs("print_tau_direct");
    unitcell::print_tau(ucell->atoms,ucell->Coordinate,ucell->ntype,ucell->lat0,ofs);
    ofs.close();
 
    // readin the data
    std::ifstream ifs;
    ifs.open("print_tau_direct");
    std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    EXPECT_THAT(str, testing::HasSubstr("DIRECT COORDINATES"));
    EXPECT_THAT(str, testing::HasSubstr("    C     0.100000000000     0.100000000000     0.100000000000  0.0000"));
    EXPECT_THAT(str, testing::HasSubstr("    H     0.150000000000     0.150000000000     0.150000000000  0.0000")); 
    ifs.close();

    remove("print_tau_direct");
}

TEST_F(PrintCellTest, PrintTauCartesian)
{
    UcellTestPrepare utp = UcellTestLib["C1H2-Cartesian"];
    ucell = utp.SetUcellInfo();
    EXPECT_EQ(ucell->Coordinate, "Cartesian");

    // open a file
    std::ofstream ofs("print_tau_Cartesian");
    unitcell::print_tau(ucell->atoms,ucell->Coordinate,ucell->ntype,ucell->lat0,ofs);
    ofs.close();

    // readin the data
    std::ifstream ifs;
    ifs.open("print_tau_Cartesian");
    std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    EXPECT_THAT(str, testing::HasSubstr("CARTESIAN COORDINATES"));
    EXPECT_THAT(str, testing::HasSubstr("    C     1.000000000000     1.000000000000     1.000000000000  0.0000"));
    EXPECT_THAT(str, testing::HasSubstr("    H     1.500000000000     1.500000000000     1.500000000000  0.0000"));
    ifs.close();

    // remove the file
    remove("print_tau_Cartesian");
}

// The same writer consumes snapshots from both PW and LCAO.
TEST_F(PrintCellTest, OnsiteSnapshotWritesActualMomentsForConstrainedAtoms)
{
    UcellTestPrepare utp = UcellTestLib["C1H2-Index"];
    ucell = utp.SetUcellInfo();
    ucell->atomic_magnetism.source = AtomicMagnetism::Source::onsite;
    ucell->atomic_magnetism.ionic_step = 2;
    ucell->atomic_magnetism.scf_converged = true;
    ucell->atomic_magnetism.values = {{0, 0.1, 0.2, 2.3}, {0, 0.4, 0.5, 0.6}, {0, 0.7, 0.8, 0.9}};
    ucell->atoms[0].m_loc_[0].set(1.123456789, 2.0, 3.0);
    ucell->atoms[0].constrain[0].set(1, 0, 1);
    ucell->atoms[0].lambda[0].set(0.0, 0.0, 0.0);
    const std::string filename = testing::TempDir() + "onsite_snapshot.STRU";
    const ModuleBase::matrix force;
    unitcell::print_stru_file(*ucell, ucell->atoms, ucell->latvec,
                             filename, "", 4, false, true, false, false, 0, force, false);
    std::ifstream file(filename);
    const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    EXPECT_THAT(text, testing::HasSubstr("source=onsite, step=3, SCF_converged=true"));
    EXPECT_THAT(text, testing::HasSubstr("mag 0.1000 0.2000 2.3000 sc 1 0 1"));
    EXPECT_THAT(text, testing::HasSubstr("computed_mag 0.1000000000 0.2000000000 2.3000000000"));
    EXPECT_THAT(text, testing::HasSubstr("mag 0.4000 0.5000 0.6000"));
    EXPECT_DOUBLE_EQ(ucell->atoms[0].m_loc_[0].z, 3.0);
    EXPECT_THAT(text, testing::HasSubstr("constraint_target 1.1234567890 2.0000000000 3.0000000000"));
    EXPECT_DOUBLE_EQ(ucell->atoms[0].m_loc_[0].x, 1.123456789);
    // Reading the output returns actual moments; target comments are informational.
    const auto atom_line = text.rfind('\n', text.find(" m ")) + 1;
    file.clear();
    file.seekg(atom_line);
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    file >> x >> y >> z;
    Atom parsed;
    parsed.vel.resize(1);
    parsed.mag.resize(1);
    parsed.m_loc_.resize(1);
    parsed.angle1.resize(1);
    parsed.angle2.resize(1);
    parsed.lambda.resize(1);
    parsed.constrain.resize(1);
    ModuleBase::Vector3<int> movement;
    bool vector_mag = false;
    bool angle_mag = false;
    bool zero_element_mag = false;
    const bool parsed_ok = unitcell::parse_atom_properties(file, parsed, 0, movement,
                                                         vector_mag, angle_mag, zero_element_mag);
    EXPECT_TRUE(parsed_ok);
    EXPECT_DOUBLE_EQ(parsed.m_loc_[0].x, 0.1);
    EXPECT_DOUBLE_EQ(parsed.m_loc_[0].z, 2.3);
    EXPECT_EQ(parsed.constrain[0].x, 1);
    EXPECT_EQ(parsed.constrain[0].y, 0);
    EXPECT_EQ(parsed.constrain[0].z, 1);
    file.close();
    std::remove(filename.c_str());
}

TEST_F(PrintCellTest, CollinearSnapshotWritesActualConstrainedMoment)
{
    UcellTestPrepare utp = UcellTestLib["C1H2-Index"];
    ucell = utp.SetUcellInfo();
    ucell->atomic_magnetism.source = AtomicMagnetism::Source::mulliken;
    ucell->atomic_magnetism.ionic_step = 0;
    ucell->atomic_magnetism.scf_converged = true;
    ucell->atomic_magnetism.values = {{0, -0.7}, {0, 0.5}, {0, 0.6}};
    ucell->atoms[0].mag[0] = 2.5;
    ucell->atoms[0].constrain[0].set(0, 0, 1);
    ucell->atoms[0].lambda[0].set(0.0, 0.0, 0.0);
    const std::string filename = testing::TempDir() + "constrained_collinear_snapshot.STRU";
    const ModuleBase::matrix force;
    unitcell::print_stru_file(*ucell, ucell->atoms, ucell->latvec,
                             filename, "", 2, false, true, false, false, 0, force, false);
    std::ifstream file(filename);
    const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    EXPECT_THAT(text, testing::HasSubstr("mag -0.7000 sc 0 0 1"));
    EXPECT_THAT(text, testing::HasSubstr("constraint_target 2.5000000000"));
    EXPECT_DOUBLE_EQ(ucell->atoms[0].mag[0], 2.5);
    EXPECT_EQ(ucell->atoms[0].constrain[0].z, 1);
    file.close();
    std::remove(filename.c_str());
}

TEST_F(PrintCellTest, InvalidAndUnevaluatedSnapshotsFallBackExplicitly)
{
    UcellTestPrepare utp = UcellTestLib["C1H2-Index"];
    ucell = utp.SetUcellInfo();
    ucell->atoms[0].mag[0] = 2.5;
    const std::string filename = testing::TempDir() + "invalid_snapshot.STRU";
    const ModuleBase::matrix force;
    for (int scenario = 0; scenario < 3; ++scenario)
    {
        ucell->atomic_magnetism.source = AtomicMagnetism::Source::onsite;
        ucell->atomic_magnetism.ionic_step = 0;
        ucell->atomic_magnetism.values = {{0, 2.3}, {0, 0.5}, {0, 0.6}};
        if (scenario == 0)
        {
            ucell->atomic_magnetism.values.pop_back(); // incomplete atom set
        }
        else if (scenario == 1)
        {
            ucell->atomic_magnetism.values[0].resize(1); // wrong spin layout
        }
        else
        {
            ucell->atomic_magnetism = AtomicMagnetism{}; // proposed, unevaluated geometry
        }
        unitcell::print_stru_file(*ucell, ucell->atoms, ucell->latvec,
                                 filename, "", 2, false, true, false, false, 0, force, false);
        std::ifstream file(filename);
        const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        EXPECT_THAT(text, testing::HasSubstr("computed result unavailable for this geometry"));
        EXPECT_THAT(text, testing::HasSubstr("mag 2.5000 # initial_mag 2.5000000000"));
        EXPECT_THAT(text, testing::Not(testing::HasSubstr("; computed_mag")));
        file.close();
    }
    std::remove(filename.c_str());
}

TEST_F(PrintCellTest, UnconvergedSnapshotIsNotLabelledConverged)
{
    UcellTestPrepare utp = UcellTestLib["C1H2-Index"];
    ucell = utp.SetUcellInfo();
    ucell->atomic_magnetism.source = AtomicMagnetism::Source::mulliken;
    ucell->atomic_magnetism.ionic_step = 4;
    ucell->atomic_magnetism.values = {{0, -0.7}, {0, 0.5}, {0, 0.6}};
    const std::string filename = testing::TempDir() + "unconverged_snapshot.STRU";
    const ModuleBase::matrix force;
    const std::string header = "# version\n# time\n# step\n# stress\n# stress\n# stress\n# NOTE: evaluated\n";
    unitcell::print_stru_file(*ucell, ucell->atoms, ucell->latvec,
                             filename, header, 2, false, true, false, false, 0, force, false);
    std::ifstream file(filename);
    const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    const auto species = text.find("ATOMIC_SPECIES");
    const auto header_end = text.begin() + species;
    EXPECT_EQ(std::count(text.begin(), header_end, '\n'), 7);
    EXPECT_THAT(text, testing::HasSubstr("source=Mulliken, step=5, SCF_converged=false"));
    EXPECT_THAT(text, testing::HasSubstr("mag -0.7000"));
    file.close();
    std::remove(filename.c_str());
}
