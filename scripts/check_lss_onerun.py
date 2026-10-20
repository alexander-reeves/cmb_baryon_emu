"""Check utils.predict_lss_spectra against a two-CAMB-run reference.

The reference feeds P_NL * S to CAMB's own lensing integration through
``ExternalNonLinearRatio`` (a second CMB run, as BCemu's ``BCemuCAMB`` does). For a few
cosmologies across emulator_config_lss_baryon.yaml and two feedback strengths it
compares the baryonic response C_L^phiphi / C_L^phiphi,dmo - 1, and checks that

  * the P(k) are unchanged when only the baryon parameters change,
  * pk_nl / pk_lin -> 1 on large scales, and the low-k slope of pk_lin is ~n_s,
  * the wall time per sample.

Run on a compute node (4 cores):  python scripts/check_lss_onerun.py
Exit status is non-zero if the response differs by more than --tol.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from cmb_baryon_emu import utils

COSMOS = {
    "fiducial": dict(h=0.6736, omega_b=0.02233, omega_cdm=0.120, ln10As=3.044, n_s=0.96, tau=0.0544),
    "low_h_low_wc": dict(h=0.45, omega_b=0.0216, omega_cdm=0.082, ln10As=3.6, n_s=0.92, tau=0.0544),
    "high_h_high_wc": dict(h=0.98, omega_b=0.0230, omega_cdm=0.195, ln10As=2.45, n_s=1.00, tau=0.0544),
}
BCMS = {
    "fid": dict(log10Mc=13.1, Theta_co=0.3, mu=1.0, delta=6.0, eta=0.10, deta=0.22, Nstar=0.028, fb=0.0486 / 0.306),
    "strong": dict(log10Mc=14.3, Theta_co=0.3, mu=0.5, delta=6.0, eta=0.2, deta=0.2, Nstar=0.03, fb=0.20),
}


def reference_clpp(cosmo, bcm, lmax, lpa, s_on):
    """C_L^phiphi from CAMB fed sqrt(P_NL S / P_lin) via ExternalNonLinearRatio."""
    import camb
    from camb import model as camb_model

    p = utils.cosmo_to_bcemu(cosmo)
    h = p["H0"] / 100.0

    def base():
        q = camb.CAMBparams()
        q.set_cosmology(H0=p["H0"], ombh2=p["ombh2"], omch2=p["omch2"], tau=p["tau"])
        q.InitPower.set_params(As=p["As"], ns=p["ns"])
        return q

    z_nl = np.concatenate([[0.0], np.geomspace(0.01, 5.0, 40)])
    q = base()
    q.WantCls = False
    q.set_matter_power(redshifts=list(z_nl), kmax=50.0, silent=True)
    q.NonLinear = camb_model.NonLinear_both
    q.NonLinearModel.set_params("mead2020")
    r1 = camb.get_results(q)
    k, zz, plin = r1.get_linear_matter_power_spectrum(hubble_units=False, k_hunit=False)
    _, _, pnl = r1.get_nonlinear_matter_power_spectrum(hubble_units=False, k_hunit=False)
    s = utils.baryon_suppression(bcm, k, zz, h) if s_on else 1.0
    z_hi = np.geomspace(5.5, 1000.0, 25)
    ratio = np.vstack([np.sqrt(pnl * s / plin), np.ones((len(z_hi), len(k)))])

    q2 = base()
    q2.set_for_lmax(lmax, lens_potential_accuracy=lpa)
    q2.NonLinear = camb_model.NonLinear_both
    ext = camb.nonlinear.ExternalNonLinearRatio()
    ext.set_ratio(k / h, np.concatenate([zz, z_hi]), ratio)
    q2.NonLinearModel = ext
    return camb.get_results(q2).get_lens_potential_cls(lmax=lmax, raw_cl=True)[2:, 0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lmax", type=int, default=utils.LSS_LMAX)
    ap.add_argument("--lens_potential_accuracy", type=int, default=8)
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="max |response difference| (fraction of C_L) allowed")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    report = {}
    ok = True
    utils._bcemu2025()  # load once, outside the timings
    for cname, cosmo in COSMOS.items():
        outs = {}
        for bname, bcm in BCMS.items():
            t = time.perf_counter()
            outs[bname] = utils.predict_lss_spectra(
                cosmo, bcm, lmax=args.lmax,
                lens_potential_accuracy=args.lens_potential_accuracy)
            dt = time.perf_counter() - t
            o = outs[bname]
            ref_s = reference_clpp(cosmo, bcm, args.lmax, args.lens_potential_accuracy, True)
            ref_1 = reference_clpp(cosmo, bcm, args.lmax, args.lens_potential_accuracy, False)
            resp = o["clpp"] / o["clpp_dmo"] - 1
            resp_ref = ref_s / ref_1 - 1
            ell = o["ell"]
            at = {int(l): [float(resp[ell == l][0]), float(resp_ref[ell == l][0])]
                  for l in (500, 1000, 2000, ell[-1])}
            diff = float(np.abs(resp - resp_ref).max())
            ok &= diff <= args.tol
            report[f"{cname}/{bname}"] = dict(
                t_sample_s=dt, max_abs_response_diff=diff,
                response_onerun_vs_ref_at_L=at,
                dmo_onerun_vs_ref_max=float(np.abs(o["clpp_dmo"] / ref_1 - 1).max()),
            )
            print(f"{cname:16s} {bname:7s} {dt:5.2f} s  |d response| max {diff:.2e}  "
                  f"response at L={ell[-1]}: {at[int(ell[-1])][0]:+.4f} (ref {at[int(ell[-1])][1]:+.4f})",
                  flush=True)

        a, b = outs["fid"], outs["strong"]
        same_pk = all(np.array_equal(a[n], b[n]) for n in ("pk_lin", "pk_nl", "clpp_dmo"))
        k = a["pk_k"]
        low = k < 1e-3
        boost_low = float(np.abs(a["pk_nl"][:, low] / a["pk_lin"][:, low] - 1).max())
        slope = np.log(a["pk_lin"][0, 1] / a["pk_lin"][0, 0]) / np.log(k[1] / k[0])
        finite = all(np.isfinite(a[n]).all() and (a[n] > 0).all() for n in ("pk_lin", "pk_nl", "clpp"))
        ok &= same_pk and finite and boost_low < 1e-3
        report[f"{cname}/pk"] = dict(
            pk_independent_of_baryons=same_pk, finite_positive=finite,
            max_nl_over_lin_minus_1_low_k=boost_low,
            low_k_slope=float(slope), n_s=cosmo["n_s"])
        print(f"{cname:16s} pk independent of baryons: {same_pk}; finite/positive: {finite}; "
              f"|P_nl/P_lin - 1| (k<1e-3) {boost_low:.1e}; low-k slope {slope:.4f} (n_s {cosmo['n_s']})",
              flush=True)

    report["pass"] = bool(ok)
    print("PASS" if ok else "FAIL")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
