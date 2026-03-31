import argparse
from pathlib import Path
from functools import partial

import evermore as evm
import jax
import jax.numpy as jnp
import pandas as pd
from evermore.parameters import filter as evm_filter
from evermore.parameters.transform import MinuitTransform
from flax import nnx
import everwillow as ew
import everwillow.statelib as sl
from everwillow.uncertainty import uncertainties
from everwillow.hypotest.calculators import AsymptoticCalculator
from everwillow.hypotest.distributions import Q0Asymptotic
from everwillow.hypotest.test_statistics import Q0
from everwillow.parameters.transforms import MinuitTransform as EwMinuitTransform

# Import from paramore
import paramore as pm


class ParamsHgg(nnx.Pytree):
    def __init__(
        self,
        lumi_7TeV: evm.Parameter,
        n_id: evm.Parameter,
        scale_j: evm.Parameter,
        globalscale: evm.Parameter,
        smear: evm.Parameter,
        r: evm.Parameter,
        bkg_norm: evm.Parameter,
        p1: evm.Parameter,
        p2: evm.Parameter,
        p3: evm.Parameter,
        p4: evm.Parameter,
    ):
        self.lumi_7TeV = lumi_7TeV
        self.n_id = n_id
        self.scale_j = scale_j
        self.globalscale = globalscale
        self.smear = smear
        self.r = r
        self.bkg_norm = bkg_norm
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4

        
class ParamsHtt(nnx.Pytree):
    def __init__(
        self,
        r: evm.Parameter,
        lumi_8TeV: evm.Parameter,
        scale_e: evm.Parameter,
    ):
        self.r = r
        self.lumi_8TeV = lumi_8TeV
        self.scale_e = scale_e


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--hgg", action="store_true", help="Run H→γγ fit only")
    parser.add_argument("--htt", action="store_true", help="Run H→ττ fit only")
    parser.add_argument("--comb", action="store_true", help="Run combined H→γγ + H→ττ fit")
    args = parser.parse_args()
    run_comb = args.comb or (not args.hgg and not args.htt)
    run_hgg  = args.hgg or run_comb
    run_htt  = args.htt or run_comb
    
    # stuff common to both    
    minuit_transform = MinuitTransform()

    ########
    # Hgg
    ########
    if run_hgg:
        print("Doing Hgg")
        # load data
        data_dir = Path(__file__).resolve().parent / "samples/hgg_htt_discovery"
        df = pd.read_parquet(data_dir / "cat0_7TeV.parquet")
        n_obs = df["n_obs"].values
    
        # define parameters
        lumi_7TeV = evm.NormalParameter(value=0.0, name="lumi_7TeV", lower=-7.0, upper=7.0, transform=minuit_transform)
        n_id = evm.NormalParameter(value=0.0, name="CMS_hgg_n_id", lower=-7.0, upper=7.0, transform=minuit_transform)
        scale_j = evm.NormalParameter(value=0.0, name="CMS_hgg_scale_j", lower=-7.0, upper=7.0, transform=minuit_transform)
        # globalscale and smear float as unit-normal pulls; physical value = pull * sigma
        # Combine's param constraint N(theta|0,sigma) evaluated at theta=pull*sigma gives 0.5*pull^2
        # which equals evermore's Normal(0,1) prior on a unit-normal parameter.
        # Note that globalscale should be NormalParameter, but leave it like that for now due to not yet understood issue with adding constraint for globalscale
        globalscale = evm.Parameter(value=0.0, name="CMS_hgg_globalscale", lower=-4.0, upper=4.0, transform=minuit_transform, frozen=True)
        smear = evm.NormalParameter(value=0.0, name="CMS_hgg_nuissancedeltasmearcat0", lower=-4.0, upper=4.0, transform=minuit_transform)
        r = evm.Parameter(value=1.0, name="r")
        bkg_norm = evm.Parameter(value=233.0, name="bkg_norm", lower=0.0, upper=1000.0, transform=minuit_transform)
        p1 = evm.Parameter(value=-0.4725, name="p1", lower=-10.0, upper=10.0, transform=minuit_transform)
        p2 = evm.Parameter(value=-0.0991, name="p2", lower=-10.0, upper=10.0, transform=minuit_transform)
        p3 = evm.Parameter(value=-0.4936, name="p3", lower=-10.0, upper=10.0, transform=minuit_transform)
        p4 = evm.Parameter(value=-0.000347, name="p4", lower=-10.0, upper=10.0, transform=minuit_transform)

        # Signal shape constants for ggH at MH = 125.5 GeV (from workspace)
        MH = 125.5
        MASS_LO, MASS_HI = 100.0, 180.0

        # Define the model and likelihood
        def model_hgg(params: ParamsHgg):
            signal_procs = {
                "ggH": {
                    "rate": 4961.8,
                    "xs": 15.18,
                    "eff_times_acc": 0.01119,
                    "kappa_lumi": 1.022,
                    "kappa_sj": 0.985,
                    "kappa_nid_up": 0.946,
                    "kappa_nid_down": 1.056,
                    "f_right": 0.9698,
                    "f1": 0.9610,
                    "mu1": 125.456,  "sigma1_nom": 1.130,
                    "mu2": 123.734,  "sigma2_nom": 3.811,
                    "muw": 124.977,  "sigmaw_nom": 2.537,
                    "delta_smear": 0.006173,
                },
                "VH": {
                    "rate": 5089.0,
                    "xs": 0.877,
                    "eff_times_acc": 0.0576,
                    "kappa_lumi": 1.022,
                    "kappa_sj": 0.998,
                    "kappa_nid_up": 0.941,
                    "kappa_nid_down": 1.050,
                    "f_right": 0.9493,
                    "f1": 0.9455,
                    "mu1": 125.445,  "sigma1_nom": 1.128,
                    "mu2": 123.721,  "sigma2_nom": 4.375,
                    "muw": 125.417,  "sigmaw_nom": 3.085,
                    "delta_smear": 0.006173,
                },
                "qqH": {
                    "rate": 5089.0,
                    "xs": 1.205,
                    "eff_times_acc": 0.03772,
                    "kappa_lumi": 1.022,
                    "kappa_sj": 0.938,
                    "kappa_nid_up": 0.932,
                    "kappa_nid_down": 1.050,
                    "f_right": 0.9372,
                    "f1": 0.9391,
                    "mu1": 125.469,  "sigma1_nom": 1.106,
                    "mu2": 124.237,  "sigma2_nom": 2.981,
                    "muw": 125.355,  "sigmaw_nom": 2.756,
                    "delta_smear": 0.006173,
                },
                "ttH": {
                    "rate": 5089.0,
                    "xs": 0.0853,
                    "eff_times_acc": 0.09629,
                    "kappa_lumi": 1.022,
                    "kappa_sj": 0.999,
                    "kappa_nid_up": 0.951,
                    "kappa_nid_down": 1.035,
                    "f_right": 0.9993,
                    "f1": 0.9763,
                    "mu1": 125.437,  "sigma1_nom": 1.159,
                    "mu2": 123.901,  "sigma2_nom": 9.544,
                    "muw": 126.000,  "sigmaw_nom": 4.000,
                    "delta_smear": 0.006173,
                },
            }
            BR = 0.00229

            signal_pdfs = {}
            signal_yields = {}
            # Build signal PDFs and yields for each process
            # globalscale/smear are unit-normal pulls; multiply by physical sigma to get GeV shifts
            theta_gs = params.globalscale.get_value() * 0.004717
            theta_smear = params.smear.get_value() * 0.001544
            for proc_name, proc_info in signal_procs.items():
                # Build PDFs
                mu1 = proc_info["mu1"] + theta_gs
                mu2 = proc_info["mu2"] + theta_gs
                muw = proc_info["muw"] + theta_gs
            
                delta_smear = proc_info["delta_smear"]
                def smeared_sigma(sigma_nom):
                    return jnp.sqrt(sigma_nom**2 + MH**2 * ((delta_smear + theta_smear)**2 - delta_smear**2))

                g1 = pm.Gaussian(mu=mu1, sigma=smeared_sigma(proc_info["sigma1_nom"]), lower=MASS_LO, upper=MASS_HI)
                g2 = pm.Gaussian(mu=mu2, sigma=smeared_sigma(proc_info["sigma2_nom"]), lower=MASS_LO, upper=MASS_HI)
                gw = pm.Gaussian(mu=muw, sigma=smeared_sigma(proc_info["sigmaw_nom"]), lower=MASS_LO, upper=MASS_HI)

                f_right = proc_info["f_right"]
                f1 = proc_info["f1"]
                signal_pdfs[proc_name] = pm.SumPDF(
                    pdfs=[g1, g2, gw],
                    extended_vals=[f_right * f1, f_right * (1 - f1), (1 - f_right)],
                    lower=MASS_LO,
                    upper=MASS_HI,
                )

                # Build signal yields: rate × BR × xs × eff_times_acc × ProcessNorm(r, θ)
                n_base = proc_info["rate"] * BR * proc_info["xs"] * proc_info["eff_times_acc"]
                lumi_mod = params.lumi_7TeV.scale_log_symmetric(kappa=proc_info["kappa_lumi"])
                sj_mod = params.scale_j.scale_log_symmetric(kappa=proc_info["kappa_sj"])
                nid_mod = params.n_id.scale_log_asymmetric(up=proc_info["kappa_nid_up"], down=proc_info["kappa_nid_down"])
                signal_yields[proc_name] = jnp.squeeze((lumi_mod @ sj_mod @ nid_mod)(jnp.array(params.r.get_value() * n_base)))

            # Background PDF: degree-4 Bernstein polynomial (c0=1 fixed, c_i = p_i^2)
            bkg_pdf = pm.BernsteinPolynomial(
                coefs=jnp.array([1.0, params.p1.get_value()**2, params.p2.get_value()**2, params.p3.get_value()**2, params.p4.get_value()**2]),
                lower=MASS_LO,
                upper=MASS_HI,
            )
            bkg_yield = params.bkg_norm.get_value()

            all_pdfs = list(signal_pdfs.values()) + [bkg_pdf]
            all_yields = list(signal_yields.values()) + [bkg_yield]

            sum_pdf = pm.SumPDF(
                pdfs=all_pdfs,
                extended_vals=all_yields,
                lower=MASS_LO,
                upper=MASS_HI,
            )
            n_tot = jnp.sum(jnp.asarray(all_yields))
        
            return sum_pdf, n_tot

        # test it
        #m, n = model_hgg(ParamsHgg(
        #    lumi_7TeV=lumi_7TeV,
        #    n_id=n_id,
        #    scale_j=scale_j,
        #    globalscale=globalscale,
        #    smear=smear,
        #    r=r,
        #    bkg_norm=bkg_norm,
        #    p1=p1,
        #    p2=p2,
        #    p3=p3,
        #    p4=p4,
        #))
        #print("Model test - log_prob at initial parameters:", m.prob(xs))

        @jax.jit
        def loss_hgg(dynamic, observation, args):
            graphdef, static = args

            params = nnx.merge(graphdef, dynamic, static, copy=True)

            mod, n_tot = model_hgg(params)

            # since it is a binned likelihood, we need to integrate the PDF over each bin to get the expected histogram
            bin_integrals = jax.vmap(lambda lo, hi: mod.integrate(lo, hi))(df["mass_lo"].values, df["mass_hi"].values)

            expectation = bin_integrals * n_tot

            # Poisson NLL
            log_likelihood = evm.pdf.PoissonContinuous(lamb=expectation).log_prob(observation).sum()

            # Add parameter constraints from logpdfs
            constraints = evm.loss.get_log_probs(params)
            log_likelihood += evm.util.sum_over_leaves(constraints)

            return -jnp.sum(log_likelihood)

        params_hgg = ParamsHgg(
            lumi_7TeV=lumi_7TeV,
            n_id=n_id,
            scale_j=scale_j,
            globalscale=globalscale,
            smear=smear,
            r=r,
            bkg_norm=bkg_norm,
            p1=p1,
            p2=p2,
            p3=p3,
            p4=p4,
        )
        graphdef, diffable, static = nnx.split(params_hgg, evm_filter.is_dynamic_parameter, ...)
        args = (graphdef, static)
        init_state = sl.State.from_pytree(diffable, sep="/")
        graphdef_hgg, static_hgg, args_hgg, init_state_hgg = graphdef, static, args, init_state

        # Everwillow bounds: MinuitTransform per bounded parameter (r has no bounds)
        # sep="|" avoids conflict with "/" in keys like "bkg_norm/value"
        bounds_hgg = sl.State.from_pytree({
            "lumi_7TeV/value": EwMinuitTransform(lower=-7.0, upper=7.0),
            "n_id/value": EwMinuitTransform(lower=-7.0, upper=7.0),
            "scale_j/value": EwMinuitTransform(lower=-7.0, upper=7.0),
            "smear/value": EwMinuitTransform(lower=-4.0, upper=4.0),
            "bkg_norm/value": EwMinuitTransform(lower=0.0, upper=1000.0),
            "p1/value": EwMinuitTransform(lower=-10.0, upper=10.0),
            "p2/value": EwMinuitTransform(lower=-10.0, upper=10.0),
            "p3/value": EwMinuitTransform(lower=-10.0, upper=10.0),
            "p4/value": EwMinuitTransform(lower=-10.0, upper=10.0),
        }, sep="|")

        fitresult = ew.fit(
            nll_fn=partial(loss_hgg, args=args),
            params=init_state,
            observation=n_obs,
            bounds=bounds_hgg,
            max_steps=1000,
        )

        # Extract results — fitresult.params is in physical space (ewp.wrap applied internally)
        fitted_params_hgg = nnx.merge(graphdef, fitresult.params.to_pytree(), static, copy=True)

        sigmas_hgg = uncertainties(partial(loss_hgg, args=args), fitresult.params, n_obs)
        r_sigma = sigmas_hgg["r/value"]
        lumi_7TeV_sigma = sigmas_hgg["lumi_7TeV/value"]
        n_id_sigma = sigmas_hgg["n_id/value"]
        scale_j_sigma = sigmas_hgg["scale_j/value"]
        #globalscale_sigma = sigmas_hgg["globalscale/value"]
        smear_sigma = sigmas_hgg["smear/value"]
        bkg_norm_sigma = sigmas_hgg["bkg_norm/value"]
        p1_sigma= sigmas_hgg["p1/value"]
        p2_sigma= sigmas_hgg["p2/value"]
        p3_sigma= sigmas_hgg["p3/value"]
        p4_sigma= sigmas_hgg["p4/value"]

        print(f"r = {float(fitted_params_hgg.r.get_value()):.4f} ± {float(r_sigma):.4f}")
        print(f"lumi_7TeV = {float(fitted_params_hgg.lumi_7TeV.get_value()):.4f} ± {float(lumi_7TeV_sigma):.4f}")
        print(f"n_id = {float(fitted_params_hgg.n_id.get_value()):.4f} ± {float(n_id_sigma):.4f}")
        print(f"scale_j = {float(fitted_params_hgg.scale_j.get_value()):.4f} ± {float(scale_j_sigma):.4f}")
        # reported in pull units; physical value = pull * sigma
        #print(f"globalscale (pull) = {float(fitted_params_hgg.globalscale.get_value()):.4f} ± {float(globalscale_sigma):.4f}  [{float(fitted_params_hgg.globalscale.get_value()) * 0.004717:.6f} GeV]")
        print(f"smear (pull) = {float(fitted_params_hgg.smear.get_value()):.4f} ± {float(smear_sigma):.4f}  [{float(fitted_params_hgg.smear.get_value()) * 0.001544:.6f} ± {float(smear_sigma) * 0.001544:.6f} GeV]")
        print(f"bkg_norm = {float(fitted_params_hgg.bkg_norm.get_value()):.4f} ± {float(bkg_norm_sigma):.4f}")
        print(f"p1 = {float(fitted_params_hgg.p1.get_value()):.4f} ± {float(p1_sigma):.4f}")
        print(f"p2 = {float(fitted_params_hgg.p2.get_value()):.4f} ± {float(p2_sigma):.4f}")
        print(f"p3 = {float(fitted_params_hgg.p3.get_value()):.4f} ± {float(p3_sigma):.4f}")
        print(f"p4 = {float(fitted_params_hgg.p4.get_value()):.4f} ± {float(p4_sigma):.4f}")

    ########
    # Htt
    ########
    if run_htt:
        print("Doing Htt")
        data_dir_htt = Path(__file__).resolve().parent / "samples/hgg_htt_discovery"

        df_obs_htt  = pd.read_parquet(data_dir_htt / "htt_data_obs.parquet")
        df_tmpl_htt = pd.read_parquet(data_dir_htt / "htt_templates.parquet")

        # Observed data
        n_obs_htt = jnp.array(df_obs_htt["n_obs"].values)

        # Histograms: one JAX array per process (nominal + systematics)
        hists_htt = {col: jnp.array(df_tmpl_htt[col].values) for col in df_tmpl_htt.columns if col not in ("bin_lo", "bin_hi", "bin_center")}

        def model_htt(params, hists):
            r_mod = params.r.scale()
            lumi_mod = params.lumi_8TeV.scale_log_symmetric(kappa=1.05)
            
            # VerticalTemplateMorphing = FastVerticalInterpHistPdf2
            shape_VH  = params.scale_e.morphing(up_template=hists["VH_up"],  down_template=hists["VH_down"])
            shape_qqH = params.scale_e.morphing(up_template=hists["qqH_up"], down_template=hists["qqH_down"])
            shape_ggH = params.scale_e.morphing(up_template=hists["ggH_up"], down_template=hists["ggH_down"])
            shape_Ztt = params.scale_e.morphing(up_template=hists["Ztt_up"], down_template=hists["Ztt_down"])

            # Signal: shape morphing + lumi lnN + signal strength r
            VH_exp  = (shape_VH  @ lumi_mod @ r_mod)(hists["VH"])
            qqH_exp = (shape_qqH @ lumi_mod @ r_mod)(hists["qqH"])
            ggH_exp = (shape_ggH @ lumi_mod @ r_mod)(hists["ggH"])

            # Bkg
            EWK_exp   = lumi_mod(hists["EWK"])
            Ztt_exp   = shape_Ztt(hists["Ztt"])
            Fakes_exp = hists["Fakes"]
            ttbar_exp = hists["ttbar"]
            
            return VH_exp + qqH_exp + ggH_exp + EWK_exp + Ztt_exp + Fakes_exp + ttbar_exp

        @jax.jit
        def loss_htt(dynamic, observation, args):
            graphdef, static, hists = args

            params = nnx.merge(graphdef, dynamic, static, copy=True)

            expectation = model_htt(params, hists)

            # Poisson NLL
            log_likelihood = evm.pdf.PoissonContinuous(lamb=expectation).log_prob(observation).sum()

            # Add parameter constraints from logpdfs
            constraints = evm.loss.get_log_probs(params)
            log_likelihood += evm.util.sum_over_leaves(constraints)

            return -jnp.sum(log_likelihood)

        params_htt = ParamsHtt(
            r=evm.Parameter(value=1.0, name="r"),
            lumi_8TeV=evm.NormalParameter(value=0.0, name="lumi_8TeV", lower=-7.0, upper=7.0, transform=minuit_transform),
            scale_e=evm.NormalParameter(value=0.0, name="scale_e", lower=-4.0, upper=4.0, transform=minuit_transform),
        )
        graphdef, diffable, static = nnx.split(params_htt, evm_filter.is_dynamic_parameter, ...)
        args = (graphdef, static, hists_htt)
        init_state = sl.State.from_pytree(diffable, sep="/")
        graphdef_htt, static_htt, args_htt, init_state_htt = graphdef, static, args, init_state

        # Everwillow bounds: MinuitTransform per bounded parameter (r has no bounds)
        # sep="|" avoids conflict with "/" in keys like "lumi_8TeV/value"
        bounds_htt = sl.State.from_pytree({
            "lumi_8TeV/value": EwMinuitTransform(lower=-7.0, upper=7.0),
            "scale_e/value": EwMinuitTransform(lower=-4.0, upper=4.0),
        }, sep="|")

        fitresult = ew.fit(
            nll_fn=partial(loss_htt, args=args),
            params=init_state,
            observation=n_obs_htt,
            bounds=bounds_htt,
            max_steps=150,
        )

        # Extract results — fitresult.params is in physical space (ewp.wrap applied internally)
        fitted_params_htt = nnx.merge(graphdef, fitresult.params.to_pytree(), static, copy=True)

        sigmas_htt = uncertainties(partial(loss_htt, args=args), fitresult.params, n_obs_htt)
        r_sigma = sigmas_htt["r/value"]
        lumi_8TeV_sigma = sigmas_htt["lumi_8TeV/value"]
        scale_e_sigma = sigmas_htt["scale_e/value"]

        print(f"r = {float(fitted_params_htt.r.get_value()):.4f} ± {float(r_sigma):.4f}")
        print(f"lumi_8TeV = {float(fitted_params_htt.lumi_8TeV.get_value()):.4f} ± {float(lumi_8TeV_sigma):.4f}")
        print(f"scale_e = {float(fitted_params_htt.scale_e.get_value()):.4f} ± {float(scale_e_sigma):.4f}")

########
# Comb
########
if run_comb:
    print("Doing Comb")
    # Wrap each NLL to unpack the combined observation tuple (obs_hgg, obs_htt)
    nll_hgg = lambda p, obs: partial(loss_hgg, args=args_hgg)(p, obs[0])
    nll_htt = lambda p, obs: partial(loss_htt, args=args_htt)(p, obs[1])

    combined_nll, combined_state = ew.prepare([nll_hgg, nll_htt], [init_state_hgg, init_state_htt])

    # Combined bounds: union of hgg and htt bounds (r has no bounds in either)
    # sep="|" avoids conflict with "/" in keys like "bkg_norm/value"
    bounds_comb = sl.State.from_pytree({
        "lumi_7TeV/value": EwMinuitTransform(lower=-7.0,    upper=7.0),
        "n_id/value": EwMinuitTransform(lower=-7.0,    upper=7.0),
        "scale_j/value": EwMinuitTransform(lower=-7.0,    upper=7.0),
        "smear/value": EwMinuitTransform(lower=-4.0,    upper=4.0),
        "bkg_norm/value": EwMinuitTransform(lower=0.0,     upper=1000.0),
        "p1/value": EwMinuitTransform(lower=-10.0,   upper=10.0),
        "p2/value": EwMinuitTransform(lower=-10.0,   upper=10.0),
        "p3/value": EwMinuitTransform(lower=-10.0,   upper=10.0),
        "p4/value": EwMinuitTransform(lower=-10.0,   upper=10.0),
        "lumi_8TeV/value": EwMinuitTransform(lower=-7.0,    upper=7.0),
        "scale_e/value": EwMinuitTransform(lower=-4.0,    upper=4.0),
    }, sep="|")

    fitresult = ew.fit(
        nll_fn=combined_nll,
        params=combined_state,
        observation=(n_obs, n_obs_htt),
        bounds=bounds_comb,
        max_steps=1000,
    )

    # Extract results — fitresult.params is in physical space (ewp.wrap applied internally)
    fitted_pytrees = fitresult.params.to_pytree()
    fitted_params_hgg_comb = nnx.merge(graphdef_hgg, fitted_pytrees[0], static_hgg, copy=True)
    fitted_params_htt_comb = nnx.merge(graphdef_htt, fitted_pytrees[1], static_htt, copy=True)

    sigmas_comb = uncertainties(combined_nll, fitresult.params, (n_obs, n_obs_htt))
    r_sigma = sigmas_comb["r/value"]
    lumi_7TeV_sigma = sigmas_comb["lumi_7TeV/value"]
    n_id_sigma = sigmas_comb["n_id/value"]
    scale_j_sigma = sigmas_comb["scale_j/value"]
    #globalscale_sigma = sigmas_comb["globalscale/value"]
    smear_sigma = sigmas_comb["smear/value"]
    bkg_norm_sigma = sigmas_comb["bkg_norm/value"]
    p1_sigma = sigmas_comb["p1/value"]
    p2_sigma = sigmas_comb["p2/value"]
    p3_sigma = sigmas_comb["p3/value"]
    p4_sigma = sigmas_comb["p4/value"]
    lumi_8TeV_sigma = sigmas_comb["lumi_8TeV/value"]
    scale_e_sigma = sigmas_comb["scale_e/value"]

    print(f"r = {float(fitted_params_hgg_comb.r.get_value()):.4f} ± {float(r_sigma):.4f}")
    print(f"lumi_7TeV = {float(fitted_params_hgg_comb.lumi_7TeV.get_value()):.4f} ± {float(lumi_7TeV_sigma):.4f}")
    print(f"n_id = {float(fitted_params_hgg_comb.n_id.get_value()):.4f} ± {float(n_id_sigma):.4f}")
    print(f"scale_j = {float(fitted_params_hgg_comb.scale_j.get_value()):.4f} ± {float(scale_j_sigma):.4f}")
    # reported in pull units; physical value = pull * sigma
    #print(f"globalscale (pull) = {float(fitted_params_hgg_comb.globalscale.get_value()):.4f} ± {float(globalscale_sigma):.4f}  [{float(fitted_params_hgg_comb.globalscale.get_value()) * 0.004717:.6f} GeV]")
    print(f"smear (pull) = {float(fitted_params_hgg_comb.smear.get_value()):.4f} ± {float(smear_sigma):.4f}  [{float(fitted_params_hgg_comb.smear.get_value()) * 0.001544:.6f} ± {float(smear_sigma) * 0.001544:.6f} GeV]")
    print(f"bkg_norm = {float(fitted_params_hgg_comb.bkg_norm.get_value()):.4f} ± {float(bkg_norm_sigma):.4f}")
    print(f"p1 = {float(fitted_params_hgg_comb.p1.get_value()):.4f} ± {float(p1_sigma):.4f}")
    print(f"p2 = {float(fitted_params_hgg_comb.p2.get_value()):.4f} ± {float(p2_sigma):.4f}")
    print(f"p3 = {float(fitted_params_hgg_comb.p3.get_value()):.4f} ± {float(p3_sigma):.4f}")
    print(f"p4 = {float(fitted_params_hgg_comb.p4.get_value()):.4f} ± {float(p4_sigma):.4f}")
    print(f"lumi_8TeV = {float(fitted_params_htt_comb.lumi_8TeV.get_value()):.4f} ± {float(lumi_8TeV_sigma):.4f}")
    print(f"scale_e = {float(fitted_params_htt_comb.scale_e.get_value()):.4f} ± {float(scale_e_sigma):.4f}")

    # Compute observed significance via q0 test statistic (asymptotic formulae, Cowan et al.)
    print("Significance computation")
    calc = AsymptoticCalculator(
        nll_fn=combined_nll,
        params=combined_state,
        observation=(n_obs, n_obs_htt),
        poi_key="r/value",
        test_statistic=Q0(),
        distribution=Q0Asymptotic(),
    )
    result = calc.test(0.0, bounds=bounds_comb)
    z_obs = float(calc.distribution.null_significance(result.test_stat_result))
    print(f"q0 = {float(result.q_obs):.4f}")
    print(f"Observed significance (asymptotic): Z = {z_obs:.4f} sigma  (p-value = {float(result.pnull):.4e})")