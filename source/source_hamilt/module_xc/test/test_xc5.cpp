#include "../xc_functional.h"
#include "../libxc_abacus.h"
#include "gtest/gtest.h"
#include "xctest.h"
#include "../exx_info.h"
#include "xc3_mock.h"
#include "source_base/matrix.h"
#include "source_cell/cal_ux.h"
#include "../../../source_base/parallel_reduce.h"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>

/************************************************
*  unit test of functionals
***********************************************/

// For more information of the functions, check the comment of xc_functional.h
// Three functions are tested:
// v_xc, the unified interface of LDA and GGA functionals
// v_xc_libxc, called by v_xc, when we use functionals from LIBXC
// v_xc_meta, unified interface of mGGA functionals

class XCTest_VXC : public XCTest
{
    protected:

        double et1 = 0, vt1 = 0;
        ModuleBase::matrix v1;

        double et2 = 0, vt2 = 0;
        ModuleBase::matrix v2;

        void SetUp()
        {
            // Define variables for parameters
            int nspin1 = 1;
            int nspin2 = 2;
            bool domag = false;
            bool domag_z = false;

            ModulePW::PW_Basis rhopw;
            UnitCell ucell;
            Charge chr;

            rhopw.nrxx = 5;
            rhopw.npw = 5;
            rhopw.nmaxgr = 5;
            rhopw.gcar = new ModuleBase::Vector3<double> [5];
            rhopw.nxyz = 1;

            ucell.tpiba = 1;
            ucell.magnet.lsign_ = true;
            unitcell::cal_ux(ucell, 4);
            ucell.omega = 1;

            chr.rhopw = &(rhopw);
            chr.rho = new double*[4];
            chr.rho[0] = new double[5];
            chr.rho[1] = new double[5];
            chr.rho[2] = new double[5];
            chr.rho[3] = new double[5];
            chr.rhog = new std::complex<double>*[2];
            chr.rhog[0] = new std::complex<double>[5];
            chr.rhog[1] = new std::complex<double>[5];

            chr.rho_core = new double[5];
            chr.rhog_core = new std::complex<double>[5];

            for(int i=0;i<5;i++)
            {
                chr.rho[0][i] = double(i);
                chr.rho[1][i] = 0.1*double(i);
                chr.rho[2][i] = chr.rho[0][i];
                chr.rho[3][i] = chr.rho[1][i];
                chr.rhog[0][i] = chr.rho[0][i];
                chr.rhog[1][i] = chr.rho[1][i];
                chr.rho_core[i] = 0;
                chr.rhog_core[i] = 0;
                rhopw.gcar[i]= 1;
            }

            XC_Functional::set_xc_type("PBE");

            const double hybrid_alpha = XC_Functional::get_hybrid_alpha();
            const double hse_omega = XC_Functional::get_hse_omega();
            std::tuple<double, double, ModuleBase::matrix> etxc_vtxc_v
                = XC_Functional::v_xc(rhopw.nrxx,&chr,&ucell,nspin1,domag,domag_z,0, hybrid_alpha, hse_omega);
            et1 = std::get<0>(etxc_vtxc_v);
            vt1 = std::get<1>(etxc_vtxc_v);
            v1  = std::get<2>(etxc_vtxc_v);

            etxc_vtxc_v
                = XC_Functional::v_xc(rhopw.nrxx,&chr,&ucell,nspin2,domag,domag_z,0, hybrid_alpha, hse_omega);
            et2 = std::get<0>(etxc_vtxc_v);
            vt2 = std::get<1>(etxc_vtxc_v);
            v2  = std::get<2>(etxc_vtxc_v);

        }
};

TEST_F(XCTest_VXC, set_xc_type)
{

    EXPECT_NEAR(et1,-22.58755058,1.0e-8);
    EXPECT_NEAR(vt1,-29.58544157,1.0e-8);
    EXPECT_NEAR(v1(0,0),0,1.0e-8);
    EXPECT_NEAR(v1(0,1),-2.10436858,1.0e-8);
    EXPECT_NEAR(v1(0,2),-2.635713084,1.0e-8);
    EXPECT_NEAR(v1(0,3),-3.005351752,1.0e-8);
    EXPECT_NEAR(v1(0,4),-3.298397892,1.0e-8);

    EXPECT_NEAR(et2,-28.97838368,1.0e-8);
    EXPECT_NEAR(vt2,-38.15420234,1.0e-8);
    EXPECT_NEAR(v2(0,0),0,1.0e-8);
    EXPECT_NEAR(v2(0,1),-2.560885436,1.0e-8);
    EXPECT_NEAR(v2(0,2),-3.219339115,1.0e-8);
    EXPECT_NEAR(v2(0,3),-3.678772816,1.0e-8);
    EXPECT_NEAR(v2(0,4),-4.043604077,1.0e-8);
    EXPECT_NEAR(v2(1,0),0,1.0e-8);
    EXPECT_NEAR(v2(1,1),-1.394281236,1.0e-8);
    EXPECT_NEAR(v2(1,2),-1.739033356,1.0e-8);
    EXPECT_NEAR(v2(1,3),-1.97506482,1.0e-8);
    EXPECT_NEAR(v2(1,4),-2.160374198,1.0e-8);

}

class XCTest_VXC_Libxc : public XCTest
{
    protected:

        double et1 = 0, vt1 = 0;
        ModuleBase::matrix v1;

        double et2 = 0, vt2 = 0;
        ModuleBase::matrix v2;

        void SetUp()
        {
            // Define variables for parameters
            int nspin1 = 1;
            int nspin2 = 2;
            bool domag = false;
            bool domag_z = false;

            ModulePW::PW_Basis rhopw;
            UnitCell ucell;
            Charge chr;

            rhopw.nrxx = 5;
            rhopw.npw = 5;
            rhopw.nmaxgr = 5;
            rhopw.gcar = new ModuleBase::Vector3<double> [5];
            rhopw.nxyz = 1;

            ucell.tpiba = 1;
            ucell.magnet.lsign_ = true;
            unitcell::cal_ux(ucell, 4);
            ucell.omega = 1;

            chr.rhopw = &(rhopw);
            chr.rho = new double*[4];
            chr.rho[0] = new double[5];
            chr.rho[1] = new double[5];
            chr.rho[2] = new double[5];
            chr.rho[3] = new double[5];
            chr.rhog = new std::complex<double>*[2];
            chr.rhog[0] = new std::complex<double>[5];
            chr.rhog[1] = new std::complex<double>[5];

            chr.rho_core = new double[5];
            chr.rhog_core = new std::complex<double>[5];

            for(int i=0;i<5;i++)
            {
                chr.rho[0][i] = double(i);
                chr.rho[1][i] = 0.1*double(i);
                chr.rho[2][i] = chr.rho[0][i];
                chr.rho[3][i] = chr.rho[1][i];
                chr.rhog[0][i] = chr.rho[0][i];
                chr.rhog[1][i] = chr.rho[1][i];
                chr.rho_core[i] = 0;
                chr.rhog_core[i] = 0;
                rhopw.gcar[i]= 1;
            }

            XC_Functional::set_xc_type("GGA_X_PBE+GGA_C_PBE");

            const double hybrid_alpha = XC_Functional::get_hybrid_alpha();
            const double hse_omega = XC_Functional::get_hse_omega();
            std::tuple<double, double, ModuleBase::matrix> etxc_vtxc_v
                = XC_Functional::v_xc(rhopw.nrxx,&chr,&ucell,nspin1,domag,domag_z,0, hybrid_alpha, hse_omega);
            et1 = std::get<0>(etxc_vtxc_v);
            vt1 = std::get<1>(etxc_vtxc_v);
            v1  = std::get<2>(etxc_vtxc_v);

            etxc_vtxc_v
                = XC_Functional::v_xc(rhopw.nrxx,&chr,&ucell,nspin2,domag,domag_z,0, hybrid_alpha, hse_omega);
            et2 = std::get<0>(etxc_vtxc_v);
            vt2 = std::get<1>(etxc_vtxc_v);
            v2  = std::get<2>(etxc_vtxc_v);

        }
};

TEST_F(XCTest_VXC_Libxc, set_xc_type)
{

    EXPECT_NEAR(et1,-22.58754423,1.0e-8);
    EXPECT_NEAR(vt1,-29.58543393,1.0e-8);
    EXPECT_NEAR(v1(0,0),0,1.0e-8);
    EXPECT_NEAR(v1(0,1),-2.104367948,1.0e-8);
    EXPECT_NEAR(v1(0,2),-2.635712365,1.0e-8);
    EXPECT_NEAR(v1(0,3),-3.005350979,1.0e-8);
    EXPECT_NEAR(v1(0,4),-3.298397079,1.0e-8);

    EXPECT_NEAR(et2,-28.97838189,1.0e-8);
    EXPECT_NEAR(vt2,-38.1541987,1.0e-8);
    EXPECT_NEAR(v2(0,0),0,1.0e-8);
    EXPECT_NEAR(v2(0,1),-2.560885532,1.0e-8);
    EXPECT_NEAR(v2(0,2),-3.219339294,1.0e-8);
    EXPECT_NEAR(v2(0,3),-3.678773042,1.0e-8);
    EXPECT_NEAR(v2(0,4),-4.043604335,1.0e-8);
    EXPECT_NEAR(v2(1,0),0,1.0e-8);
    EXPECT_NEAR(v2(1,1),-1.394276473,1.0e-8);
    EXPECT_NEAR(v2(1,2),-1.739027899,1.0e-8);
    EXPECT_NEAR(v2(1,3),-1.975058937,1.0e-8);
    EXPECT_NEAR(v2(1,4),-2.160368003,1.0e-8);

}

class XCTest_VXC_meta : public XCTest
{
    protected:

        double et1 = 0, vt1 = 0;
        ModuleBase::matrix v1,vtau1;

        double et2 = 0, vt2 = 0;
        ModuleBase::matrix v2,vtau2;

        void SetUp()
        {
            // Define variables for parameters
            int nspin1 = 1;
            int nspin2 = 2;

            ModulePW::PW_Basis rhopw;
            UnitCell ucell;
            Charge chr;

            rhopw.nrxx = 5;
            rhopw.npw = 5;
            rhopw.nmaxgr = 5;
            rhopw.gcar = new ModuleBase::Vector3<double> [5];
            rhopw.nxyz = 1;

            ucell.tpiba = 1;
            ucell.magnet.lsign_ = true;
            unitcell::cal_ux(ucell, 4);
            ucell.omega = 1;

            chr.rhopw = &(rhopw);
            chr.rho = new double*[2];
            chr.rho[0] = new double[5];
            chr.rho[1] = new double[5];
            chr.rhog = new std::complex<double>*[2];
            chr.rhog[0] = new std::complex<double>[5];
            chr.rhog[1] = new std::complex<double>[5];

            chr.rho_core = new double[5];
            chr.rhog_core = new std::complex<double>[5];

            for(int i=0;i<5;i++)
            {
                chr.rho[0][i] = double(i);
                chr.rho[1][i] = 0.1*double(i);
                chr.rhog[0][i] = chr.rho[0][i];
                chr.rhog[1][i] = chr.rho[1][i];
                chr.rho_core[i] = 0;
                chr.rhog_core[i] = 0;
                rhopw.gcar[i]= 1;
            }

            chr.kin_r = new double*[2];
            chr.kin_r[0] = new double[5];
            chr.kin_r[1] = new double[5];
            chr.kin_r[0][0] = 0;
            chr.kin_r[0][1] = 0.02403590412;
            chr.kin_r[0][2] = 0.01672229351;
            chr.kin_r[0][3] = 0.01340429824;
            chr.kin_r[0][4] = 0.01141731056;
            chr.kin_r[1][0] = 0.5;
            chr.kin_r[1][1] = 0.52403590412;
            chr.kin_r[1][2] = 0.51672229351;
            chr.kin_r[1][3] = 0.51340429824;
            chr.kin_r[1][4] = 0.51141731056;

            XC_Functional::set_xc_type("SCAN");

            const double hybrid_alpha = XC_Functional::get_hybrid_alpha();
            const double hse_omega = XC_Functional::get_hse_omega();
            std::tuple<double, double, ModuleBase::matrix, ModuleBase::matrix> etxc_vtxc_v
                = XC_Functional_Libxc::v_xc_meta(XC_Functional::get_func_id(), rhopw.nrxx,ucell.omega,ucell.tpiba,&chr,nspin1, hybrid_alpha, hse_omega);
            et1 = std::get<0>(etxc_vtxc_v);
            vt1 = std::get<1>(etxc_vtxc_v);
            v1  = std::get<2>(etxc_vtxc_v);
            vtau1 = std::get<3>(etxc_vtxc_v);

            etxc_vtxc_v
                = XC_Functional_Libxc::v_xc_meta(XC_Functional::get_func_id(), rhopw.nrxx,ucell.omega,ucell.tpiba,&chr,nspin2, hybrid_alpha, hse_omega);
            et2 = std::get<0>(etxc_vtxc_v);
            vt2 = std::get<1>(etxc_vtxc_v);
            v2  = std::get<2>(etxc_vtxc_v);
            vtau2 = std::get<3>(etxc_vtxc_v);
        }
};

TEST_F(XCTest_VXC_meta, set_xc_type)
{

    EXPECT_NEAR(et1,-25.13065363,1.0e-8);
    EXPECT_NEAR(vt1,-33.13880774,1.0e-8);
    EXPECT_NEAR(v1(0,0),0,1.0e-8);
    EXPECT_NEAR(v1(0,1),-2.336719556,1.0e-8);
    EXPECT_NEAR(v1(0,2),-2.942649664,1.0e-8);
    EXPECT_NEAR(v1(0,3),-3.36679035,1.0e-8);
    EXPECT_NEAR(v1(0,4),-3.704104452,1.0e-8);
    EXPECT_NEAR(vtau1(0,0),0,1.0e-8);
    EXPECT_NEAR(vtau1(0,1),0.0187099814,1.0e-8);
    EXPECT_NEAR(vtau1(0,2),0.01578002561,1.0e-8);
    EXPECT_NEAR(vtau1(0,3),0.01423896928,1.0e-8);
    EXPECT_NEAR(vtau1(0,4),0.01321861589,1.0e-8);

    EXPECT_NEAR(et2,-32.72218711,1.0e-8);
    EXPECT_NEAR(vt2,-43.31358017,1.0e-8);
    EXPECT_NEAR(v2(0,0),0,1.0e-8);
    EXPECT_NEAR(v2(0,1),-2.901190807,1.0e-8);
    EXPECT_NEAR(v2(0,2),-3.662642983,1.0e-8);
    EXPECT_NEAR(v2(0,3),-4.196098173,1.0e-8);
    EXPECT_NEAR(v2(0,4),-4.620454375,1.0e-8);
    EXPECT_NEAR(v2(1,0),0,1.0e-8);
    EXPECT_NEAR(v2(1,1),-1.285513329,1.0e-8);
    EXPECT_NEAR(v2(1,2),-1.795172177,1.0e-8);
    EXPECT_NEAR(v2(1,3),-2.064035864,1.0e-8);
    EXPECT_NEAR(v2(1,4),-2.275487119,1.0e-8);
    EXPECT_NEAR(vtau2(0,0),0,1.0e-8);
    EXPECT_NEAR(vtau2(0,1),0.01677946177,1.0e-8);
    EXPECT_NEAR(vtau2(0,2),0.01410816304,1.0e-8);
    EXPECT_NEAR(vtau2(0,3),0.01263339482,1.0e-8);
    EXPECT_NEAR(vtau2(0,4),0.01165715023,1.0e-8);
    EXPECT_NEAR(vtau2(1,0),0,1.0e-8);
    EXPECT_NEAR(vtau2(1,1),0.01591158497,1.0e-8);
    EXPECT_NEAR(vtau2(1,2),0.07990709956,1.0e-8);
    EXPECT_NEAR(vtau2(1,3),0.04145463825,1.0e-8);
    EXPECT_NEAR(vtau2(1,4),0.0311787189,1.0e-8);
}

/************************************************
 *  unit tests for the gga_grad keyword (nspin=4
 *  noncollinear GGA gradient methods)
 *
 *  Method 2 differentiates the complete discrete local-spin-map and FFT
 *  gradient graph.  Its reverse is tested on a real PW grid in
 *  test_ncgga_discrete_fd.cpp.
 *  Method 3 uses the continuous B2 invariants, including transverse
 *  magnetization-gradient power and the local response of m_hat.
 ***********************************************/

namespace
{
constexpr int gga_grad_nrxx = 5;

// build a mock 4-component charge on the mocked 5-point grid.
// pattern 0: m = (0,0,mz), mz>0, so m_hat = (0,0,1) everywhere
// pattern 1: m direction varies from point to point
// pattern 2: m . ux changes sign across the grid (ux = (0,1,2) in the mock)
struct Ns4Charge
{
    ModulePW::PW_Basis rhopw;
    UnitCell ucell;
    Charge chr;

    Ns4Charge(const int pattern)
    {
        rhopw.nrxx = gga_grad_nrxx;
        rhopw.npw = gga_grad_nrxx;
        rhopw.nmaxgr = gga_grad_nrxx;
        rhopw.gcar = new ModuleBase::Vector3<double>[gga_grad_nrxx];
        rhopw.nxyz = 1;

        ucell.tpiba = 1;
        ucell.omega = 1;
        ucell.magnet.lsign_ = true;
        unitcell::cal_ux(ucell, 4);

        chr.rhopw = &(rhopw);
        chr.rho = new double*[4];
        for (int is = 0; is < 4; ++is)
        {
            chr.rho[is] = new double[gga_grad_nrxx];
        }
        chr.rhog = new std::complex<double>*[2];
        chr.rhog[0] = new std::complex<double>[gga_grad_nrxx];
        chr.rhog[1] = new std::complex<double>[gga_grad_nrxx];
        chr.rho_core = new double[gga_grad_nrxx];
        chr.rhog_core = new std::complex<double>[gga_grad_nrxx];

        for (int i = 0; i < gga_grad_nrxx; ++i)
        {
            chr.rho[0][i] = 2.0 + i;
            if (pattern == 1)
            {
                chr.rho[1][i] = 0.10 * (i + 1);
                chr.rho[2][i] = 0.05 * (gga_grad_nrxx - i);
                chr.rho[3][i] = 0.20 * (i + 1);
            }
            else if (pattern == 2)
            {
                chr.rho[1][i] = 0.0;
                chr.rho[2][i] = (i % 2 == 0) ? 0.3 : -0.3;
                chr.rho[3][i] = 0.05;
            }
            else
            {
                chr.rho[1][i] = 0.0;
                chr.rho[2][i] = 0.0;
                chr.rho[3][i] = 0.2 * (i + 1);
            }
            chr.rhog[0][i] = chr.rho[0][i];
            chr.rhog[1][i] = chr.rho[1][i];
            chr.rho_core[i] = 0;
            chr.rhog_core[i] = 0;
            rhopw.gcar[i] = 1;
        }
    }
};

TEST(GgaGradVxc, LegacyProjectedLibxcReverseIsRejected)
{
    Ns4Charge mock(0);
    const int nspin = 2;
    const std::vector<double> sgn(gga_grad_nrxx * nspin, 1.0);
    const std::vector<double> vsigma(gga_grad_nrxx * 3, 0.2);
    const std::vector<double> mag_part
        = XC_Functional_Libxc::compute_mag_part_nspin4(
            gga_grad_nrxx, &mock.chr);
    const std::tuple<std::vector<double>, std::vector<double>> rho_amag
        = XC_Functional_Libxc::convert_rho_amag_nspin4(
            nspin, gga_grad_nrxx, &mock.chr);
    const std::vector<std::vector<ModuleBase::Vector3<double>>> gdr
        = XC_Functional_Libxc::cal_gdr_sf(
            nspin,
            gga_grad_nrxx,
            std::get<0>(rho_amag),
            mag_part,
            mock.ucell.tpiba,
            &mock.chr);

    EXPECT_THROW(
        XC_Functional_Libxc::cal_dh_sf(nspin,
                                       gga_grad_nrxx,
                                       sgn,
                                       gdr,
                                       vsigma,
                                       mag_part,
                                       2,
                                       mock.ucell.tpiba,
                                       &mock.chr),
        std::domain_error);
    EXPECT_NO_THROW(
        XC_Functional_Libxc::cal_dh_sf(nspin,
                                       gga_grad_nrxx,
                                       sgn,
                                       gdr,
                                       vsigma,
                                       mag_part,
                                       3,
                                       mock.ucell.tpiba,
                                       &mock.chr));
}

struct Ns2LocalCharge
{
    ModulePW::PW_Basis rhopw;
    Charge chr;

    Ns2LocalCharge()
    {
        rhopw.nrxx = 1;
        rhopw.npw = 1;
        rhopw.nmaxgr = 1;
        rhopw.nxyz = 1;
        rhopw.gcar = new ModuleBase::Vector3<double>[1];
        rhopw.gcar[0] = 0.0;

        chr.rhopw = &rhopw;
        chr.rho = new double*[2];
        chr.rhog = new std::complex<double>*[2];
        for (int is = 0; is < 2; ++is)
        {
            chr.rho[is] = new double[1];
            chr.rhog[is] = new std::complex<double>[1];
            chr.rhog[is][0] = 0.0;
        }
        chr.rho_core = new double[1];
        chr.rhog_core = new std::complex<double>[1];
        chr.rho_core[0] = 0.0;
        chr.rhog_core[0] = 0.0;
    }
};

// run XC_Functional::v_xc for nspin=4 with noncollinear magnetism
std::tuple<double, double, ModuleBase::matrix> run_vxc_nspin4(
    const std::string& functional,
    const int pattern,
    const int gga_grad)
{
    Ns4Charge mock(pattern);
    XC_Functional::set_xc_type(functional);
    return XC_Functional::v_xc(gga_grad_nrxx,
                               &mock.chr,
                               &mock.ucell,
                               4,
                               true,
                               false,
                               gga_grad,
                               XC_Functional::get_hybrid_alpha(),
                               XC_Functional::get_hse_omega());
}

// compare two (etxc, vtxc, v) results
void expect_vxc_equal(const std::tuple<double, double, ModuleBase::matrix>& a,
                      const std::tuple<double, double, ModuleBase::matrix>& b,
                      const double tol)
{
    EXPECT_NEAR(std::get<0>(a), std::get<0>(b), tol);
    EXPECT_NEAR(std::get<1>(a), std::get<1>(b), tol);
    const ModuleBase::matrix& va = std::get<2>(a);
    const ModuleBase::matrix& vb = std::get<2>(b);
    ASSERT_EQ(va.nr, vb.nr);
    ASSERT_EQ(va.nc, vb.nc);
    for (int ir = 0; ir < va.nr; ++ir)
    {
        for (int ic = 0; ic < va.nc; ++ic)
        {
            EXPECT_NEAR(va(ir, ic), vb(ir, ic), tol);
        }
    }
}
} // namespace

// m_hat = m/|m|, zero where |m| ~ 0
TEST(GgaGradTools, ComputeMagPartNspin4)
{
    Charge chr;
    chr.rho = new double*[4];
    for (int is = 0; is < 4; ++is)
    {
        chr.rho[is] = new double[3];
    }
    chr.rho[1][0] = 3.0; chr.rho[2][0] = 0.0; chr.rho[3][0] = 4.0; // |m|=5
    chr.rho[1][1] = 0.0; chr.rho[2][1] = 0.0; chr.rho[3][1] = 0.0; // |m|=0
    chr.rho[1][2] = 1.0; chr.rho[2][2] = 2.0; chr.rho[3][2] = 2.0; // |m|=3

    const std::vector<double> mp = XC_Functional_Libxc::compute_mag_part_nspin4(3, &chr);

    EXPECT_NEAR(mp[0], 3.0 / 5.0, 1e-15);
    EXPECT_NEAR(mp[3], 0.0, 1e-15);
    EXPECT_NEAR(mp[6], 4.0 / 5.0, 1e-15);
    EXPECT_NEAR(mp[1], 0.0, 1e-15);
    EXPECT_NEAR(mp[4], 0.0, 1e-15);
    EXPECT_NEAR(mp[7], 0.0, 1e-15);
    EXPECT_NEAR(mp[2], 1.0 / 3.0, 1e-15);
    EXPECT_NEAR(mp[5], 2.0 / 3.0, 1e-15);
    EXPECT_NEAR(mp[8], 2.0 / 3.0, 1e-15);
}

// v_tot = 0.5*(v_up+v_dn), v_mu = 0.5*(v_up-v_dn)*m_hat_mu
TEST(GgaGradTools, ConvertVNspin4Sf)
{
    Ns4Charge mock(0); // m_hat = (0,0,1)
    const std::vector<double> mag_part
        = XC_Functional_Libxc::compute_mag_part_nspin4(gga_grad_nrxx, &mock.chr);

    ModuleBase::matrix v(2, gga_grad_nrxx);
    for (int ir = 0; ir < gga_grad_nrxx; ++ir)
    {
        v(0, ir) = 1.0 + ir;
        v(1, ir) = 0.5 * ir;
    }
    const ModuleBase::matrix v4
        = XC_Functional_Libxc::convert_v_nspin4_sf(gga_grad_nrxx, &mock.chr, mag_part, v);

    for (int ir = 0; ir < gga_grad_nrxx; ++ir)
    {
        EXPECT_NEAR(v4(0, ir), 0.5 * (v(0, ir) + v(1, ir)), 1e-14);
        EXPECT_NEAR(v4(1, ir), 0.0, 1e-14);
        EXPECT_NEAR(v4(2, ir), 0.0, 1e-14);
        EXPECT_NEAR(v4(3, ir), 0.5 * (v(0, ir) - v(1, ir)), 1e-14);
    }
}

// original conversion: has_mag=false leaves magnetic channels zero
TEST(GgaGradTools, ConvertVNspin4HasMag)
{
    Ns4Charge mock(0);
    std::vector<double> amag(gga_grad_nrxx);
    for (int ir = 0; ir < gga_grad_nrxx; ++ir)
    {
        amag[ir] = mock.chr.rho[3][ir];
    }
    ModuleBase::matrix v(2, gga_grad_nrxx);
    for (int ir = 0; ir < gga_grad_nrxx; ++ir)
    {
        v(0, ir) = 1.0 + ir;
        v(1, ir) = 0.5 * ir;
    }

    const ModuleBase::matrix v_nomag
        = XC_Functional_Libxc::convert_v_nspin4(gga_grad_nrxx, &mock.chr, amag, v, false);
    for (int ir = 0; ir < gga_grad_nrxx; ++ir)
    {
        EXPECT_NEAR(v_nomag(0, ir), 0.5 * (v(0, ir) + v(1, ir)), 1e-14);
        EXPECT_NEAR(v_nomag(1, ir), 0.0, 1e-14);
        EXPECT_NEAR(v_nomag(2, ir), 0.0, 1e-14);
        EXPECT_NEAR(v_nomag(3, ir), 0.0, 1e-14);
    }

    const ModuleBase::matrix v_mag
        = XC_Functional_Libxc::convert_v_nspin4(gga_grad_nrxx, &mock.chr, amag, v, true);
    for (int ir = 0; ir < gga_grad_nrxx; ++ir)
    {
        const double vs = 0.5 * (v(0, ir) - v(1, ir));
        EXPECT_NEAR(v_mag(3, ir), vs * mock.chr.rho[3][ir] / amag[ir], 1e-14);
    }
}

// In the collinear limit with a uniform magnetization direction and no
// transverse gradients, continuous B2 reduces to the projected method.
TEST(GgaGradVxc, BuiltinUniformDirectionMethodsAgree)
{
    const auto r2 = run_vxc_nspin4("PBE", 0, 2);
    const auto r3 = run_vxc_nspin4("PBE", 0, 3);
    EXPECT_EQ(std::get<2>(r2).nr, 4);
    expect_vxc_equal(r2, r3, 1e-10);
}

// gga_grad=0 keeps the original built-in algorithm and must not crash
TEST(GgaGradVxc, BuiltinOriginalAlgorithmRuns)
{
    const auto r0 = run_vxc_nspin4("PBE", 1, 0);
    EXPECT_EQ(std::get<2>(r0).nr, 4);
    EXPECT_TRUE(std::isfinite(std::get<0>(r0)));
    EXPECT_TRUE(std::isfinite(std::get<1>(r0)));
}

// noncolin_rho with lsign=true defines up/down w.r.t. the global axis ux
// through sign(m . ux); with lsign=false, up is always the local |m|
TEST(GgaGradTools, NoncolinRhoGlobalAxis)
{
    Ns4Charge mock(2); // pattern 2: m . ux changes sign across the grid
    const double* ux = mock.ucell.magnet.ux_; // (0,1,2) in the mock

    std::vector<double> rup(gga_grad_nrxx), rdn(gga_grad_nrxx), neg(gga_grad_nrxx);
    XC_Functional::noncolin_rho(
        rup.data(), rdn.data(), neg.data(), mock.chr.rho, gga_grad_nrxx, ux, true);
    for (int ir = 0; ir < gga_grad_nrxx; ++ir)
    {
        const double mx = mock.chr.rho[1][ir], my = mock.chr.rho[2][ir], mz = mock.chr.rho[3][ir];
        const double amag = std::sqrt(mx * mx + my * my + mz * mz);
        const double sign = (mx * ux[0] + my * ux[1] + mz * ux[2] > 0) ? 1.0 : -1.0;
        EXPECT_NEAR(rup[ir], 0.5 * (mock.chr.rho[0][ir] + sign * amag), 1e-14);
        EXPECT_NEAR(rdn[ir], 0.5 * (mock.chr.rho[0][ir] - sign * amag), 1e-14);
    }
    // the sign really flips on this grid, i.e. the global axis matters here
    EXPECT_NEAR(neg[0], 1.0, 1e-14);
    EXPECT_NEAR(neg[1], -1.0, 1e-14);

    XC_Functional::noncolin_rho(
        rup.data(), rdn.data(), neg.data(), mock.chr.rho, gga_grad_nrxx, ux, false);
    for (int ir = 0; ir < gga_grad_nrxx; ++ir)
    {
        const double mx = mock.chr.rho[1][ir], my = mock.chr.rho[2][ir], mz = mock.chr.rho[3][ir];
        const double amag = std::sqrt(mx * mx + my * my + mz * mz);
        EXPECT_NEAR(rup[ir], 0.5 * (mock.chr.rho[0][ir] + amag), 1e-14);
        EXPECT_NEAR(rdn[ir], 0.5 * (mock.chr.rho[0][ir] - amag), 1e-14);
    }
}

// gga_grad=1 must ignore the global magnetization direction: with lsign_=true
// it has to give the same gradcorr result as gga_grad=0 with lsign_=false
TEST(GgaGradVxc, BuiltinGgaGrad1IgnoresGlobalAxis)
{
    XC_Functional::set_xc_type("PBE");
    const double hybrid_alpha = XC_Functional::get_hybrid_alpha();
    const double hse_omega = XC_Functional::get_hse_omega();

    Ns4Charge mock_a(2); // lsign_ = true
    double et1 = 0, vt1 = 0;
    ModuleBase::matrix v1(4, gga_grad_nrxx);
    std::vector<double> dum;
    XC_Functional::gradcorr(et1, vt1, v1, &mock_a.chr, &mock_a.rhopw, &mock_a.ucell,
                            dum, false, 4, true, false, 1, hybrid_alpha, hse_omega);

    Ns4Charge mock_b(2);
    mock_b.ucell.magnet.lsign_ = false;
    double et0 = 0, vt0 = 0;
    ModuleBase::matrix v0(4, gga_grad_nrxx);
    XC_Functional::gradcorr(et0, vt0, v0, &mock_b.chr, &mock_b.rhopw, &mock_b.ucell,
                            dum, false, 4, true, false, 0, hybrid_alpha, hse_omega);

    EXPECT_NEAR(et0, et1, 1e-12);
    EXPECT_NEAR(vt0, vt1, 1e-12);
    for (int is = 0; is < 4; ++is)
    {
        for (int ir = 0; ir < gga_grad_nrxx; ++ir)
        {
            EXPECT_NEAR(v0(is, ir), v1(is, ir), 1e-12);
        }
    }
}

// The diagonal FFT mock maps every field value at a grid point to a gradient
// parallel to the same mock reciprocal vector. Consequently
//   sum_mu |grad m_mu|^2 = |sum_mu m_hat_mu grad m_mu|^2
// point by point, even when the sampled magnetization direction varies. This
// mock therefore cannot be used as evidence for transverse B2 energy.
TEST(GgaGradVxc, BuiltinDiagonalFftMockCannotTestTransverseB2Energy)
{
    Ns4Charge mock(1);
    const std::vector<double> mag_part
        = XC_Functional_Libxc::compute_mag_part_nspin4(gga_grad_nrxx, &mock.chr);
    std::vector<std::vector<ModuleBase::Vector3<double>>> grad_m(
        3, std::vector<ModuleBase::Vector3<double>>(gga_grad_nrxx));
    std::vector<std::complex<double>> reciprocal(gga_grad_nrxx);
    for (int mu = 0; mu < 3; ++mu)
    {
        mock.rhopw.real2recip(mock.chr.rho[mu + 1], reciprocal.data());
        XC_Functional::grad_rho(reciprocal.data(),
                                grad_m[mu].data(),
                                &mock.rhopw,
                                mock.ucell.tpiba);
    }

    double direction_change = 0.0;
    double transverse_power = 0.0;
    for (int mu = 0; mu < 3; ++mu)
    {
        const double difference
            = mag_part[1 + mu * gga_grad_nrxx] - mag_part[mu * gga_grad_nrxx];
        direction_change += difference * difference;
    }
    for (int ir = 0; ir < gga_grad_nrxx; ++ir)
    {
        ModuleBase::Vector3<double> grad_magnitude;
        double component_power = 0.0;
        for (int mu = 0; mu < 3; ++mu)
        {
            component_power += grad_m[mu][ir] * grad_m[mu][ir];
            grad_magnitude += mag_part[ir + mu * gga_grad_nrxx] * grad_m[mu][ir];
        }
        transverse_power += std::abs(component_power - grad_magnitude * grad_magnitude);
    }

    EXPECT_GT(direction_change, 1.0e-4);
    EXPECT_LE(transverse_power, 1.0e-12);

    const auto r2 = run_vxc_nspin4("PBE", 1, 2);
    const auto r3 = run_vxc_nspin4("PBE", 1, 3);
    EXPECT_TRUE(std::isfinite(std::get<0>(r3)));
    EXPECT_TRUE(std::isfinite(std::get<1>(r3)));
    EXPECT_NEAR(std::get<0>(r3), std::get<0>(r2), 1.0e-12);
}

// The validated continuous B2 implementation currently uses the built-in
// functional derivatives. Do not silently route method 3 through the old
// LIBXC full-divergence formula.
TEST(GgaGradVxc, LibxcContinuousB2IsRejected)
{
    const auto r2 = run_vxc_nspin4("GGA_X_PBE+GGA_C_PBE", 0, 2);
    EXPECT_EQ(std::get<2>(r2).nr, 4);
    EXPECT_THROW(run_vxc_nspin4("GGA_X_PBE+GGA_C_PBE", 0, 3), std::domain_error);
}

// The low-level LibXC entry point is public and can be called without going
// through XC_Functional::v_xc. It must enforce the same method-3 boundary.
TEST(GgaGradVxc, LibxcLowLevelContinuousB2IsRejected)
{
    Ns4Charge mock(0);
    const std::vector<int> func_ids = {XC_GGA_X_PBE, XC_GGA_C_PBE};
    const auto run_low_level = [&](const int gga_grad)
    {
        return XC_Functional_Libxc::v_xc_libxc(func_ids,
                                               gga_grad_nrxx,
                                               mock.ucell.omega,
                                               mock.ucell.tpiba,
                                               &mock.chr,
                                               4,
                                               true,
                                               false,
                                               gga_grad,
                                               nullptr,
                                               0.0,
                                               0.0);
    };

    EXPECT_NO_THROW(run_low_level(2));
    EXPECT_THROW(run_low_level(3), std::domain_error);
}

TEST(GgaGradVxc, LibxcDensityFloorDifferentiatesTheWeightedEnergy)
{
    constexpr double density_threshold = 1.0e-6;
    Ns2LocalCharge mock;
    mock.chr.rho[0][0] = 2.0 * density_threshold;
    mock.chr.rho[1][0] = 0.5 * density_threshold;
    const std::vector<int> func_ids = {XC_LDA_X};
    const std::map<int, double> scaled = {{XC_LDA_X, 0.37}};
    const std::map<int, double>* scaling_cases[] = {nullptr, &scaled};
    const auto evaluate = [&](const std::map<int, double>* scaling_factor)
    {
        return XC_Functional_Libxc::v_xc_libxc(func_ids,
                                               1,
                                               1.0,
                                               1.0,
                                               &mock.chr,
                                               2,
                                               false,
                                               false,
                                               0,
                                               scaling_factor,
                                               0.0,
                                               0.0);
    };

    for (const std::map<int, double>* scaling_factor : scaling_cases)
    {
        const auto reference = evaluate(scaling_factor);
        double analytic = std::get<2>(reference)(1, 0);
        Parallel_Reduce::reduce_pool(analytic);
        if (std::getenv("ABACUS_XC_FD_TRACE") != nullptr)
        {
            std::cout << std::setprecision(17)
                      << "XC_SANITIZER_REFERENCE case=density_floor"
                      << " scaled=" << (scaling_factor != nullptr)
                      << " energy=" << std::get<0>(reference)
                      << " vtxc=" << std::get<1>(reference)
                      << " analytic=" << analytic << std::endl;
        }
        const double original = mock.chr.rho[1][0];
        const double steps[] = {8.0e-8, 4.0e-8, 2.0e-8};
        for (const double step : steps)
        {
            mock.chr.rho[1][0] = original + step;
            const double energy_plus = std::get<0>(evaluate(scaling_factor));
            mock.chr.rho[1][0] = original - step;
            const double energy_minus = std::get<0>(evaluate(scaling_factor));
            mock.chr.rho[1][0] = original;

            const double finite_difference = (energy_plus - energy_minus) / (2.0 * step);
            if (std::getenv("ABACUS_XC_FD_TRACE") != nullptr)
            {
                std::cout << std::setprecision(17)
                          << "XC_SANITIZER_FD case=density_floor"
                          << " scaled=" << (scaling_factor != nullptr)
                          << " eps=" << step
                          << " analytic=" << analytic
                          << " finite_difference=" << finite_difference
                          << " absolute_error=" << std::abs(analytic - finite_difference)
                          << std::endl;
            }
            const double scale = std::max(1.0, std::max(std::abs(analytic),
                                                        std::abs(finite_difference)));
            EXPECT_NEAR(analytic, finite_difference, 2.0e-8 * scale)
                << "step=" << step
                << ", scaled=" << (scaling_factor != nullptr);
        }
    }
}

TEST(GgaGradVxc, LibxcNspin4NearSaturationDifferentiatesTheWeightedEnergy)
{
    constexpr double density_threshold = 1.0e-6;
    Ns4Charge mock(0);
    for (int ir = 0; ir < gga_grad_nrxx; ++ir)
    {
        mock.chr.rho[0][ir] = 0.45;
        mock.chr.rho[1][ir] = 0.0;
        mock.chr.rho[2][ir] = 0.0;
        mock.chr.rho[3][ir] = 0.45 - density_threshold;
    }
    const std::vector<int> func_ids = {XC_LDA_X};
    const int gga_grad_modes[] = {0, 2};
    for (const int gga_grad : gga_grad_modes)
    {
        const auto evaluate = [&, gga_grad]()
        {
            return XC_Functional_Libxc::v_xc_libxc(func_ids,
                                                   gga_grad_nrxx,
                                                   mock.ucell.omega,
                                                   mock.ucell.tpiba,
                                                   &mock.chr,
                                                   4,
                                                   true,
                                                   false,
                                                   gga_grad,
                                                   nullptr,
                                                   0.0,
                                                   0.0);
        };

        const auto reference = evaluate();
        const int components[] = {0, 3};
        const double steps[] = {8.0e-8, 4.0e-8, 2.0e-8};
        for (const int component : components)
        {
            double analytic = std::get<2>(reference)(component, 0);
            Parallel_Reduce::reduce_pool(analytic);
            if (std::getenv("ABACUS_XC_FD_TRACE") != nullptr)
            {
                std::cout << std::setprecision(17)
                          << "XC_SANITIZER_REFERENCE case=nspin4_near_saturation"
                          << " gga_grad=" << gga_grad
                          << " component=" << component
                          << " energy=" << std::get<0>(reference)
                          << " vtxc=" << std::get<1>(reference)
                          << " analytic=" << analytic << std::endl;
            }
            const double original = mock.chr.rho[component][0];
            for (const double step : steps)
            {
                mock.chr.rho[component][0] = original + step;
                const double energy_plus = std::get<0>(evaluate());
                mock.chr.rho[component][0] = original - step;
                const double energy_minus = std::get<0>(evaluate());
                mock.chr.rho[component][0] = original;

                const double finite_difference = (energy_plus - energy_minus) / (2.0 * step);
                if (std::getenv("ABACUS_XC_FD_TRACE") != nullptr)
                {
                    std::cout << std::setprecision(17)
                              << "XC_SANITIZER_FD case=nspin4_near_saturation"
                              << " gga_grad=" << gga_grad
                              << " component=" << component
                              << " eps=" << step
                              << " analytic=" << analytic
                              << " finite_difference=" << finite_difference
                              << " absolute_error=" << std::abs(analytic - finite_difference)
                              << std::endl;
                }
                const double scale = std::max(1.0, std::max(std::abs(analytic),
                                                            std::abs(finite_difference)));
                EXPECT_NEAR(analytic, finite_difference, 2.0e-8 * scale)
                    << "gga_grad=" << gga_grad
                    << ", component=" << component
                    << ", step=" << step;
            }
        }
    }
}

// for LIBXC, gga_grad=0 and 1 both keep the original collinear algorithm
TEST(GgaGradVxc, LibxcZeroEqualsOne)
{
    const auto r0 = run_vxc_nspin4("GGA_X_PBE+GGA_C_PBE", 1, 0);
    const auto r1 = run_vxc_nspin4("GGA_X_PBE+GGA_C_PBE", 1, 1);
    expect_vxc_equal(r0, r1, 1e-12);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
