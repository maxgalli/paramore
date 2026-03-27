import argparse
from pathlib import Path
from functools import partial

import evermore as evm
import jax
import jax.numpy as jnp
import optimistix
import pandas as pd
from evermore.parameters import filter as evm_filter
from evermore.parameters.transform import MinuitTransform, unwrap, wrap
from flax import nnx
from jax.experimental import checkify
from jax.flatten_util import ravel_pytree
import matplotlib.pyplot as plt
import numpy as np
import everwillow as ew
import everwillow.statelib as sl
from everwillow.uncertainty import uncertainties, covariance_matrix, correlation_matrix

# Import from paramore
import paramore as pm

wrap_checked = checkify.checkify(wrap)


class ConstrainedParameter(evm.Parameter):
    """evm.Parameter with a custom-width Gaussian prior (for 'param' systematics in Combine)."""

    def __init__(self, *args, constraint_width: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_metadata('constraint_width', constraint_width)

    @property
    def prior(self):
        return evm.pdf.Normal(mean=jnp.array(0.0), width=jnp.array(self.constraint_width))


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
        # load and plot data to have a look
        data_dir = Path(__file__).resolve().parent / "samples/hgg_htt_discovery"
        fig_output_dir = Path(__file__).resolve().parent / "figures_hgg_htt_discovery"
        df = pd.read_parquet(data_dir / "cat0_7TeV.parquet")
        mass_centers = df["mass_center"].values
        n_obs = df["n_obs"].values
        bin_width = df["mass_hi"].values - df["mass_lo"].values
    
        fig, ax = plt.subplots()
        ax.errorbar(mass_centers, n_obs / bin_width, yerr=np.sqrt(n_obs) / bin_width, fmt="ko", label="data")
        ax.set_xlim(100, 180)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(r"$m_{\gamma\gamma}$ [GeV]")
        ax.set_ylabel("Events / GeV")
        fig.tight_layout()
        for ext in ["png", "pdf"]:
            fig.savefig(fig_output_dir / f"hgg_data.{ext}")
    
        # define parameters
        lumi_7TeV = evm.NormalParameter(value=0.0, name="lumi_7TeV", lower=-7.0, upper=7.0, transform=minuit_transform)
        n_id = evm.NormalParameter(value=0.0, name="CMS_hgg_n_id", lower=-7.0, upper=7.0, transform=minuit_transform)
        scale_j = evm.NormalParameter(value=0.0, name="CMS_hgg_scale_j", lower=-7.0, upper=7.0, transform=minuit_transform)
        globalscale = ConstrainedParameter(value=0.0, name="CMS_hgg_globalscale", lower=-0.018868, upper=0.018868, transform=minuit_transform, constraint_width=0.004717)
        smear = ConstrainedParameter(value=0.0, name="CMS_hgg_nuissancedeltasmearcat0", lower=-0.006176, upper=0.006176, transform=minuit_transform, constraint_width=0.001544)
        r = evm.Parameter(value=1.0, name="r", lower=0.0, upper=10.0, transform=minuit_transform)
        bkg_norm = evm.Parameter(value=233.0, name="bkg_norm", lower=0.0, upper=1000.0, transform=minuit_transform)
        p1 = evm.Parameter(value=-0.4725, name="p1", lower=-10.0, upper=10.0, transform=minuit_transform)
        p2 = evm.Parameter(value=-0.0991, name="p2", lower=-10.0, upper=10.0, transform=minuit_transform)
        p3 = evm.Parameter(value=-0.4936, name="p3", lower=-10.0, upper=10.0, transform=minuit_transform)
        p4 = evm.Parameter(value=-0.000347, name="p4", lower=-10.0, upper=10.0, transform=minuit_transform)

        # Build signal model for 1 production mode as example and plot
        # Signal shape constants for ggH at MH = 125.5 GeV (from workspace)
        MH = 125.5
        MASS_LO, MASS_HI = 100.0, 180.0
        F_RIGHT = 0.9698
        F1 = 0.9610
        DELTA_SMEAR = 0.006173  # GeV
        DM1, DM2, DMW = -0.044, -1.766, -0.523          # mean offsets [GeV]
        SIGMA1_NOM, SIGMA2_NOM, SIGMAW_NOM = 1.130, 3.811, 2.537  # [GeV]
        N_GGH_BASE = 1.930  # rate × BR × σ × ε·A at r=1, all θ=0

        theta_gs = globalscale.get_value()
        theta_smear = smear.get_value()

        # Means: additive shift from globalscale
        mu1 = MH + DM1 + theta_gs
        mu2 = MH + DM2 + theta_gs
        muw = MH + DMW + theta_gs

        # Sigmas: quadrature smearing formula
        def smeared_sigma(sigma_nom):
            return jnp.sqrt(sigma_nom**2 + MH**2 * ((DELTA_SMEAR + theta_smear)**2 - DELTA_SMEAR**2))

        g1 = pm.Gaussian(mu=mu1, sigma=smeared_sigma(SIGMA1_NOM), lower=MASS_LO, upper=MASS_HI)
        g2 = pm.Gaussian(mu=mu2, sigma=smeared_sigma(SIGMA2_NOM), lower=MASS_LO, upper=MASS_HI)
        gw = pm.Gaussian(mu=muw, sigma=smeared_sigma(SIGMAW_NOM), lower=MASS_LO, upper=MASS_HI)

        # Full ggH yield with ProcessNorm
        lumi_mod = lumi_7TeV.scale_log_symmetric(kappa=1.022)
        sj_mod   = scale_j.scale_log_symmetric(kappa=0.985)
        nid_mod  = n_id.scale_log_asymmetric(up=0.946, down=1.056)
        n_ggh = jnp.squeeze((lumi_mod @ sj_mod @ nid_mod)(jnp.array(r.get_value() * N_GGH_BASE)))

        signal_pdf_ggh = pm.SumPDF(
            pdfs=[g1, g2, gw],
            extended_vals=[F_RIGHT * F1 * n_ggh, F_RIGHT * (1 - F1) * n_ggh, (1 - F_RIGHT) * n_ggh],
            lower=MASS_LO,
            upper=MASS_HI,
        )

        xs = jnp.array(mass_centers)
        signal_curve = signal_pdf_ggh.prob(xs) * n_ggh * bin_width[0]

        fig, ax = plt.subplots()
        ax.plot(mass_centers, signal_curve, "r-", lw=1.5, label=rf"ggH ($\mu=1$, {float(n_ggh):.2f} evt)")
        ax.set_xlim(100, 180)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(r"$m_{\gamma\gamma}$ [GeV]")
        ax.set_ylabel("Events / GeV")
        ax.legend()
        fig.tight_layout()
        for ext in ["png", "pdf"]:
            fig.savefig(fig_output_dir / f"hgg_signal_ggh.{ext}")
        plt.close(fig)

        # Build background model and plot
        bkg_pdf = pm.BernsteinPolynomial(
            coefs=jnp.array([1.0, p1.get_value()**2, p2.get_value()**2, p3.get_value()**2, p4.get_value()**2]),  # square the coefficients as done in Combine
            lower=MASS_LO,
            upper=MASS_HI,
        )

        bkg_curve = bkg_pdf.prob(xs) * bkg_norm.get_value() * bin_width[0]

        fig, ax = plt.subplots()
        ax.plot(mass_centers, bkg_curve, "b-", lw=1.5, label=f"Background (Bernstein, {float(bkg_norm.get_value()):.0f} evt)")
        ax.set_xlim(MASS_LO, MASS_HI)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(r"$m_{\gamma\gamma}$ [GeV]")
        ax.set_ylabel("Events / GeV")
        ax.legend()
        fig.tight_layout()
        for ext in ["png", "pdf"]:
            fig.savefig(fig_output_dir / f"hgg_background.{ext}")
        plt.close(fig)

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
            theta_gs = params.globalscale.get_value()
            theta_smear = params.smear.get_value()
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
                sj_mod   = params.scale_j.scale_log_symmetric(kappa=proc_info["kappa_sj"])
                nid_mod  = params.n_id.scale_log_asymmetric(up=proc_info["kappa_nid_up"], down=proc_info["kappa_nid_down"])
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

        @nnx.jit
        def loss_hgg(dynamic, observation, args):
            graphdef, static = args

            params_unwrapped = nnx.merge(graphdef, dynamic, static, copy=True)

            errors, params_wrapped = wrap_checked(params_unwrapped)

            mod, n_tot = model_hgg(params_wrapped)

            # since it is a binned likelihood, we need to integrate the PDF over each bin to get the expected histogram
            bin_integrals = jax.vmap(lambda lo, hi: mod.integrate(lo, hi))(df["mass_lo"].values, df["mass_hi"].values)

            expectation = bin_integrals * n_tot

            # Poisson NLL
            log_likelihood = evm.pdf.PoissonContinuous(lamb=expectation).log_prob(observation).sum()

            # Add parameter constraints from logpdfs
            constraints = evm.loss.get_log_probs(params_wrapped)
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
        params_hgg_unwrapped = unwrap(params_hgg)
        graphdef, diffable, static = nnx.split(params_hgg_unwrapped, evm_filter.is_dynamic_parameter, ...)
        args = (graphdef, static)
        init_state = sl.State.from_pytree(diffable)
        graphdef_hgg, static_hgg, args_hgg, init_state_hgg = graphdef, static, args, init_state

        fitresult = ew.fit(
            nll_fn=partial(loss_hgg, args=args),
            params=init_state,
            observation=n_obs,
            max_steps=1000,
        )

        # Extract results
        fitted_hgg_unwrapped = nnx.merge(graphdef, fitresult.params.to_pytree(), static, copy=True)
        fitted_params_hgg = wrap(fitted_hgg_unwrapped)

        flat_opt, unravel = ravel_pytree(fitresult.params.to_pytree())
        cov_matrix = covariance_matrix(partial(loss_hgg, args=args), fitresult.params, n_obs)

        def param_uncertainty(selector):
            """Propagate covariance from the diffable space to a physical parameter value."""
            def value_fn(flat_params):
                params_unwrapped_local = nnx.merge(graphdef, unravel(flat_params), static, copy=True)
                _, params_wrapped = wrap_checked(params_unwrapped_local)
                return selector(params_wrapped)

            grad = jax.grad(value_fn)(flat_opt)
            return jnp.sqrt(jnp.dot(grad, cov_matrix @ grad))

        r_sigma = param_uncertainty(lambda p: p.r.get_value())

        print(f"r = {float(fitted_params_hgg.r.get_value()):.4f} ± {float(r_sigma):.4f}")

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
        print(hists_htt)

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

        @nnx.jit
        def loss_htt(dynamic, observation, args):
            graphdef, static, hists = args
        
            params_unwrapped = nnx.merge(graphdef, dynamic, static, copy=True)
        
            errors, params_wrapped = wrap_checked(params_unwrapped)

            expectation = model_htt(params_wrapped, hists)
        
            # Poisson NLL
            log_likelihood = evm.pdf.PoissonContinuous(lamb=expectation).log_prob(observation).sum()

            # Add parameter constraints from logpdfs
            constraints = evm.loss.get_log_probs(params_wrapped)
            log_likelihood += evm.util.sum_over_leaves(constraints)
        
            return -jnp.sum(log_likelihood)

        params_htt = ParamsHtt(
            r=evm.Parameter(value=1.0, name="r", lower=0.0, upper=10.0, transform=minuit_transform),
            lumi_8TeV=evm.NormalParameter(value=0.0, name="lumi_8TeV", lower=-7.0, upper=7.0, transform=minuit_transform),
            scale_e=evm.NormalParameter(value=0.0, name="scale_e", lower=-4.0, upper=4.0, transform=minuit_transform),
        )
        params_htt_unwrapped = unwrap(params_htt)
        graphdef, diffable, static = nnx.split(params_htt_unwrapped, evm_filter.is_dynamic_parameter, ...)
        args = (graphdef, static, hists_htt)
        init_state = sl.State.from_pytree(diffable)
        graphdef_htt, static_htt, args_htt, init_state_htt = graphdef, static, args, init_state

        fitresult = ew.fit(
            nll_fn=partial(loss_htt, args=args),
            params=init_state,
            observation=n_obs_htt,
            max_steps=150
        )

        # Extract results
        fitted_htt_unwrapped = nnx.merge(graphdef, fitresult.params.to_pytree(), static, copy=True)
        fitted_params_htt = wrap(fitted_htt_unwrapped)

        flat_opt, unravel = ravel_pytree(fitresult.params.to_pytree())
        cov_matrix = covariance_matrix(partial(loss_htt, args=args), fitresult.params, n_obs_htt)

        def param_uncertainty(selector):
            """Propagate covariance from the diffable space to a physical parameter value."""
            def value_fn(flat_params):
                params_unwrapped_local = nnx.merge(graphdef, unravel(flat_params), static, copy=True)
                _, params_wrapped = wrap_checked(params_unwrapped_local)
                return selector(params_wrapped)

            grad = jax.grad(value_fn)(flat_opt)
            return jnp.sqrt(jnp.dot(grad, cov_matrix @ grad))

        r_sigma = param_uncertainty(lambda p: p.r.get_value())
        lumi_8TeV_sigma = param_uncertainty(lambda p: p.lumi_8TeV.get_value())
        scale_e_sigma = param_uncertainty(lambda p: p.scale_e.get_value())

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
    fitresult = ew.fit(
        nll_fn=combined_nll,
        params=combined_state,
        observation=(n_obs, n_obs_htt),
    )

    # Extract results — to_pytree() returns (pytree_hgg, pytree_htt) for the merged state
    fitted_pytrees = fitresult.params.to_pytree()
    fitted_hgg_unwrapped = nnx.merge(graphdef_hgg, fitted_pytrees[0], static_hgg, copy=True)
    fitted_params_hgg_comb = wrap(fitted_hgg_unwrapped)
    fitted_htt_unwrapped = nnx.merge(graphdef_htt, fitted_pytrees[1], static_htt, copy=True)
    fitted_params_htt_comb = wrap(fitted_htt_unwrapped)

    # Build flat_opt using the same key ordering as covariance_matrix internally uses
    free_keys = tuple(fitresult.params.keys())
    flat_opt = jnp.array([fitresult.params[k] for k in free_keys])
    cov_matrix = covariance_matrix(combined_nll, fitresult.params, (n_obs, n_obs_htt))

    def param_uncertainty_hgg(selector):
        """Propagate covariance from the diffable space to a physical Hgg parameter value."""
        def value_fn(flat_params):
            updates = {k: flat_params[i] for i, k in enumerate(free_keys)}
            pytrees = sl.update(fitresult.params, updates=updates).to_pytree()
            params_unwrapped_local = nnx.merge(graphdef_hgg, pytrees[0], static_hgg, copy=True)
            _, params_wrapped = wrap_checked(params_unwrapped_local)
            return selector(params_wrapped)
        grad = jax.grad(value_fn)(flat_opt)
        return jnp.sqrt(jnp.dot(grad, cov_matrix @ grad))

    def param_uncertainty_htt(selector):
        """Propagate covariance from the diffable space to a physical Htt parameter value."""
        def value_fn(flat_params):
            updates = {k: flat_params[i] for i, k in enumerate(free_keys)}
            pytrees = sl.update(fitresult.params, updates=updates).to_pytree()
            params_unwrapped_local = nnx.merge(graphdef_htt, pytrees[1], static_htt, copy=True)
            _, params_wrapped = wrap_checked(params_unwrapped_local)
            return selector(params_wrapped)
        grad = jax.grad(value_fn)(flat_opt)
        return jnp.sqrt(jnp.dot(grad, cov_matrix @ grad))

    r_sigma        = param_uncertainty_hgg(lambda p: p.r.get_value())
    lumi_7TeV_sigma = param_uncertainty_hgg(lambda p: p.lumi_7TeV.get_value())
    n_id_sigma     = param_uncertainty_hgg(lambda p: p.n_id.get_value())
    scale_j_sigma  = param_uncertainty_hgg(lambda p: p.scale_j.get_value())
    globalscale_sigma = param_uncertainty_hgg(lambda p: p.globalscale.get_value())
    smear_sigma    = param_uncertainty_hgg(lambda p: p.smear.get_value())
    bkg_norm_sigma = param_uncertainty_hgg(lambda p: p.bkg_norm.get_value())
    p1_sigma       = param_uncertainty_hgg(lambda p: p.p1.get_value())
    p2_sigma       = param_uncertainty_hgg(lambda p: p.p2.get_value())
    p3_sigma       = param_uncertainty_hgg(lambda p: p.p3.get_value())
    p4_sigma       = param_uncertainty_hgg(lambda p: p.p4.get_value())
    lumi_8TeV_sigma = param_uncertainty_htt(lambda p: p.lumi_8TeV.get_value())
    scale_e_sigma  = param_uncertainty_htt(lambda p: p.scale_e.get_value())

    print(f"r = {float(fitted_params_hgg_comb.r.get_value()):.4f} ± {float(r_sigma):.4f}")
    print(f"lumi_7TeV = {float(fitted_params_hgg_comb.lumi_7TeV.get_value()):.4f} ± {float(lumi_7TeV_sigma):.4f}")
    print(f"n_id = {float(fitted_params_hgg_comb.n_id.get_value()):.4f} ± {float(n_id_sigma):.4f}")
    print(f"scale_j = {float(fitted_params_hgg_comb.scale_j.get_value()):.4f} ± {float(scale_j_sigma):.4f}")
    print(f"globalscale = {float(fitted_params_hgg_comb.globalscale.get_value()):.4f} ± {float(globalscale_sigma):.4f}")
    print(f"smear = {float(fitted_params_hgg_comb.smear.get_value()):.4f} ± {float(smear_sigma):.4f}")
    print(f"bkg_norm = {float(fitted_params_hgg_comb.bkg_norm.get_value()):.4f} ± {float(bkg_norm_sigma):.4f}")
    print(f"p1 = {float(fitted_params_hgg_comb.p1.get_value()):.4f} ± {float(p1_sigma):.4f}")
    print(f"p2 = {float(fitted_params_hgg_comb.p2.get_value()):.4f} ± {float(p2_sigma):.4f}")
    print(f"p3 = {float(fitted_params_hgg_comb.p3.get_value()):.4f} ± {float(p3_sigma):.4f}")
    print(f"p4 = {float(fitted_params_hgg_comb.p4.get_value()):.4f} ± {float(p4_sigma):.4f}")
    print(f"lumi_8TeV = {float(fitted_params_htt_comb.lumi_8TeV.get_value()):.4f} ± {float(lumi_8TeV_sigma):.4f}")
    print(f"scale_e = {float(fitted_params_htt_comb.scale_e.get_value()):.4f} ± {float(scale_e_sigma):.4f}")