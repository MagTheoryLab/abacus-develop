#ifndef XC_BUILTIN_PBE_MATH_H
#define XC_BUILTIN_PBE_MATH_H

#include <cmath>

#if defined(__CUDACC__)
#define ABACUS_PBE_HOST_DEVICE __host__ __device__
#else
#define ABACUS_PBE_HOST_DEVICE
#endif

namespace ModuleXC
{
namespace BuiltinPbeMath
{

ABACUS_PBE_HOST_DEVICE inline void slater_spin(
    const double rho,
    const double zeta,
    double& ex,
    double& vxup,
    double& vxdw)
{
    const double f = -1.107838149573033610;
    const double alpha = 2.0 / 3.0;
    const double third = 1.0 / 3.0;
    const double p43 = 4.0 / 3.0;

    double rho13 = pow((1.0 + zeta) * rho, third);
    const double exup = f * alpha * rho13;
    vxup = p43 * f * alpha * rho13;
    rho13 = pow((1.0 - zeta) * rho, third);
    const double exdw = f * alpha * rho13;
    vxdw = p43 * f * alpha * rho13;
    ex = 0.5 * ((1.0 + zeta) * exup + (1.0 - zeta) * exdw);
}

ABACUS_PBE_HOST_DEVICE inline void pw_spin(
    const double rs,
    const double zeta,
    double& ec,
    double& vcup,
    double& vcdw)
{
    const double a = 0.0310910;
    const double a1 = 0.213700;
    const double b1 = 7.59570;
    const double b2 = 3.58760;
    const double b3 = 1.63820;
    const double b4 = 0.492940;
    const double ap = 0.0155450;
    const double a1p = 0.205480;
    const double b1p = 14.11890;
    const double b2p = 6.19770;
    const double b3p = 3.36620;
    const double b4p = 0.625170;
    const double aa = 0.0168870;
    const double a1a = 0.111250;
    const double b1a = 10.3570;
    const double b2a = 3.62310;
    const double b3a = 0.880260;
    const double b4a = 0.496710;
    const double fz0 = 1.7099210;
    const double zeta2 = zeta * zeta;
    const double zeta3 = zeta2 * zeta;
    const double zeta4 = zeta3 * zeta;
    const double rs12 = sqrt(rs);
    const double rs32 = rs * rs12;
    const double rs2 = rs * rs;

    const double om
        = 2.0 * a * (b1 * rs12 + b2 * rs + b3 * rs32 + b4 * rs2);
    const double dom
        = 2.0 * a
          * (0.5 * b1 * rs12 + b2 * rs + 1.5 * b3 * rs32
             + 2.0 * b4 * rs2);
    const double olog = log(1.0 + 1.0 / om);
    const double epwc = -2.0 * a * (1.0 + a1 * rs) * olog;
    const double vpwc
        = -2.0 * a * (1.0 + 2.0 / 3.0 * a1 * rs) * olog
          - 2.0 / 3.0 * a * (1.0 + a1 * rs) * dom / (om * (om + 1.0));

    const double omp
        = 2.0 * ap
          * (b1p * rs12 + b2p * rs + b3p * rs32 + b4p * rs2);
    const double domp
        = 2.0 * ap
          * (0.5 * b1p * rs12 + b2p * rs + 1.5 * b3p * rs32
             + 2.0 * b4p * rs2);
    const double ologp = log(1.0 + 1.0 / omp);
    const double epwcp = -2.0 * ap * (1.0 + a1p * rs) * ologp;
    const double vpwcp
        = -2.0 * ap * (1.0 + 2.0 / 3.0 * a1p * rs) * ologp
          - 2.0 / 3.0 * ap * (1.0 + a1p * rs) * domp
                / (omp * (omp + 1.0));

    const double oma
        = 2.0 * aa
          * (b1a * rs12 + b2a * rs + b3a * rs32 + b4a * rs2);
    const double doma
        = 2.0 * aa
          * (0.5 * b1a * rs12 + b2a * rs + 1.5 * b3a * rs32
             + 2.0 * b4a * rs2);
    const double ologa = log(1.0 + 1.0 / oma);
    const double alpha = 2.0 * aa * (1.0 + a1a * rs) * ologa;
    const double vpwca
        = 2.0 * aa * (1.0 + 2.0 / 3.0 * a1a * rs) * ologa
          + 2.0 / 3.0 * aa * (1.0 + a1a * rs) * doma
                / (oma * (oma + 1.0));

    const double two43 = pow(2.0, 4.0 / 3.0);
    const double fz
        = (pow(1.0 + zeta, 4.0 / 3.0)
           + pow(1.0 - zeta, 4.0 / 3.0) - 2.0)
          / (two43 - 2.0);
    const double dfz
        = (pow(1.0 + zeta, 1.0 / 3.0)
           - pow(1.0 - zeta, 1.0 / 3.0))
          * 4.0 / (3.0 * (two43 - 2.0));

    ec = epwc + alpha * fz * (1.0 - zeta4) / fz0
         + (epwcp - epwc) * fz * zeta4;
    const double spin_derivative
        = alpha / fz0
              * (dfz * (1.0 - zeta4) - 4.0 * fz * zeta3)
          + (epwcp - epwc)
                * (dfz * zeta4 + 4.0 * fz * zeta3);
    vcup = vpwc + vpwca * fz * (1.0 - zeta4) / fz0
           + (vpwcp - vpwc) * fz * zeta4
           + spin_derivative * (1.0 - zeta);
    vcdw = vpwc + vpwca * fz * (1.0 - zeta4) / fz0
           + (vpwcp - vpwc) * fz * zeta4
           - spin_derivative * (1.0 + zeta);
}

ABACUS_PBE_HOST_DEVICE inline void pbex(
    const double rho,
    const double grho,
    const int iflag,
    double& sx,
    double& v1x,
    double& v2x)
{
    const double third = 1.0 / 3.0;
    const double c1 = 0.750 / 3.14159265358979323846;
    const double c2 = 3.0936677262801360;
    const double c5 = 4.0 * third;
    const double k[3] = {0.8040, 1.24500, 0.8040};
    const double mu[3]
        = {0.2195149727645171,
           0.2195149727645171,
           0.12345679012345679};
    const double agrho = sqrt(grho);
    const double kf = c2 * pow(rho, third);
    const double dsg = 0.5 / kf;
    const double s1 = agrho * dsg / rho;
    const double s2 = s1 * s1;
    const double ds = -c5 * s1;
    const double f1 = s2 * mu[iflag] / k[iflag];
    const double f2 = 1.0 + f1;
    const double f3 = k[iflag] / f2;
    const double fx = k[iflag] - f3;
    const double exunif = -c1 * kf;
    sx = exunif * fx;
    const double dxunif = exunif * third;
    const double dfx1 = f2 * f2;
    const double dfx = 2.0 * mu[iflag] * s1 / dfx1;
    v1x = sx + dxunif * fx + exunif * dfx * ds;
    v2x = exunif * dfx * dsg / agrho;
    sx *= rho;
}

ABACUS_PBE_HOST_DEVICE inline void pbec_spin(
    const double rho,
    const double zeta,
    const double grho,
    const int iflag,
    double& sc,
    double& v1cup,
    double& v1cdw,
    double& v2c)
{
    const double ga = 0.0310910;
    const double be[3] = {0.0, 0.06672455060314922, 0.0460000};
    const double third = 1.0 / 3.0;
    const double pi34 = 0.62035049089940;
    const double xkf = 1.9191582926775130;
    const double xks = 1.1283791670955130;
    const double rs = pi34 / pow(rho, third);
    double ec = 0.0;
    double vcup = 0.0;
    double vcdw = 0.0;
    pw_spin(rs, zeta, ec, vcup, vcdw);

    const double kf = xkf / rs;
    const double ks = xks * sqrt(kf);
    const double fz
        = 0.5 * (pow(1.0 + zeta, 2.0 / 3.0)
                 + pow(1.0 - zeta, 2.0 / 3.0));
    const double fz2 = fz * fz;
    const double fz3 = fz2 * fz;
    const double dfz
        = (pow(1.0 + zeta, -1.0 / 3.0)
           - pow(1.0 - zeta, -1.0 / 3.0))
          / 3.0;
    const double t = sqrt(grho) / (2.0 * fz * ks * rho);
    const double expe = exp(-ec / (fz3 * ga));
    const double af = be[iflag] / ga * (1.0 / (expe - 1.0));
    const double bfup = expe * (vcup - ec) / fz3;
    const double bfdw = expe * (vcdw - ec) / fz3;
    const double y = af * t * t;
    const double denominator = 1.0 + y + y * y;
    const double xy = (1.0 + y) / denominator;
    const double qy
        = y * y * (2.0 + y) / pow(denominator, 2.0);
    const double s1 = 1.0 + be[iflag] / ga * t * t * xy;
    const double h0 = fz3 * ga * log(s1);
    const double dh0up
        = be[iflag] * t * t * fz3 / s1
          * (-7.0 / 3.0 * xy
             - qy * (af * bfup / be[iflag] - 7.0 / 3.0));
    const double dh0dw
        = be[iflag] * t * t * fz3 / s1
          * (-7.0 / 3.0 * xy
             - qy * (af * bfdw / be[iflag] - 7.0 / 3.0));
    const double common_zeta_response
        = 3.0 * h0 / fz
          - be[iflag] * t * t * fz2 / s1
                * (2.0 * xy
                   - qy
                         * (3.0 * af * expe * ec
                                / (fz3 * be[iflag])
                            + 2.0));
    const double dh0zup
        = common_zeta_response * dfz * (1.0 - zeta);
    const double dh0zdw
        = -common_zeta_response * dfz * (1.0 + zeta);
    const double ddh0
        = be[iflag] * fz / (2.0 * ks * ks * rho)
          * (xy - qy) / s1;
    sc = rho * h0;
    v1cup = h0 + dh0up + dh0zup;
    v1cdw = h0 + dh0dw + dh0zdw;
    v2c = ddh0;
}

} // namespace BuiltinPbeMath
} // namespace ModuleXC

#undef ABACUS_PBE_HOST_DEVICE

#endif
