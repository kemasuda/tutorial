__all__ = ["multinormalpdf", "logprobparams",
           "logprob", "logprob_map"] + ["Simulate"]

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
import matplotlib.pyplot as plt
from jax import vmap, jit


def normal(x, mu, sigma):
    return jnp.exp(-0.5*(x-mu)**2/sigma**2) / jnp.sqrt(2*jnp.pi) / sigma


def multinormalpdf(x, p):
    """PDF as a superposition of multiple normal distribution

        Args:
            x: values at which PDF is evaluated
            p: set of parameters (weights, means, SDs)

        Returns:
            array: PDF values at x

    """
    weights, mus, sigmas = p[:len(
        p)//3], p[len(p)//3:2*len(p)//3], p[2*len(p)//3:]
    sigmas = jnp.exp(sigmas)
    weights = jnp.exp(weights) / jnp.sum(jnp.exp(weights))
    pdfs = jnp.array([w*normal(x, mu, sigma)
                     for w, mu, sigma in zip(weights, mus, sigmas)])
    return jnp.sum(pdfs, axis=0)


def minusloglike(p, x):
    pdf = multinormalpdf(x, p)
    loglike = jnp.sum(jnp.log(pdf))
    return -loglike


def fit_pdf(data, n_comp=3, show=False, method='TNC'):
    """fit samples using multinormal PDF

        Args:
            data: samples

        Returns:
            array: parameter array

    """
    mu, sigma = np.mean(data), np.std(data)
    print(mu, sigma)
    def func(p): return minusloglike(p, data)
    solver = jaxopt.ScipyBoundedMinimize(fun=func, method=method)
    p0 = jnp.array(list(np.arange(n_comp)) +
                   [mu]*n_comp + [np.log(sigma/2.)]*n_comp)
    plower = [-100]*n_comp + [mu-5*sigma]*n_comp + [np.log(sigma/4.)]*n_comp
    pupper = [100]*n_comp + [mu+5*sigma]*n_comp + [np.log(sigma*4.)]*n_comp
    bounds = (plower, pupper)
    res = solver.run(p0, bounds=bounds)
    if show:
        x0 = np.linspace(mu-5*sigma, mu+5*sigma, 1000)
        pdf = multinormalpdf(x0, res.params)
        cdf = np.cumsum(pdf)
        plt.figure()
        plt.plot(x0, cdf/cdf[-1], lw=1, color='C0',
                 label='%d gaussian fit' % n_comp)
        plt.ylabel("CDF")
        plt.hist(data, bins=1000, cumulative=True, density=True, histtype='step',
                 lw=4, color='gray', alpha=0.4, label='data')
        plt.legend(loc='upper left')

        plt.figure()
        plt.plot(x0, pdf, lw=1, color='C0', label='%d gaussian fit' % n_comp)
        plt.ylabel("PDF")
        plt.hist(data, bins=50, density=True, histtype='step',
                 lw=4, color='gray', alpha=0.4, label='data')
        plt.legend(loc='upper left')
    return res.params


def logprob(x, p):
    """log-probability at x for a single set of parameters"""
    return jnp.log(multinormalpdf(x, p))


# log-probability at x for 2D parameter array (num_object, num_parameters)
logprob_map = jit(vmap(logprob, (0, 0), 0))


def logprobparams(samples, n_comp=6, show=False):
    """get parameter array for log-probability

        Args:
            samples: shape (num_objects, num_samples)

        Returns:
            2D array: (num_objects, num_parameters)

    """
    params = []
    for i in range(len(samples)):
        _params = fit_pdf(samples[i], show=show, n_comp=n_comp)
        params.append(_params)
    return jnp.array(params)


class Simulate:
    """class to simulate vsini measurements"""

    def __init__(self):
        pass

    def fisher(self, N, kappa, seed=123):
        np.random.seed(seed)
        N = int(N)
        z = np.random.rand(N)
        self.cospsi = 1. + np.log((1 - z) + z * np.exp(-2 * kappa)) / kappa
        self.sinpsi = np.sqrt(1 - self.cospsi**2)
        self.psi = np.arccos(self.cospsi)
        cosomega = np.random.randn(N)
        sinomega = np.random.randn(N)
        self.omega = np.arctan2(sinomega, cosomega)
        self.cosi1 = 2 * np.random.rand(N) - 1
        self.sini1 = np.sqrt(1 - self.cosi1**2)
        self.cosi2 = self.cosi1 * self.cospsi + \
            self.sini1 * self.sinpsi * np.cos(self.omega)
        self.sini2 = np.sqrt(1 - self.cosi2**2)

    def logvsiniratio_fisher(self, N, kappa, sigma_lnv, sigma_obserr, seed=123):
        N = int(N)
        self.fisher(N, kappa)
        np.random.seed(seed)
        self.dlnv = np.random.randn(N) * sigma_lnv
        self.dlnsini = np.log(self.sini1 / self.sini2)
        self.dlnvsini = self.dlnv + self.dlnsini
        self.dlnvsini_obs = np.random.normal(self.dlnvsini, sigma_obserr)

    def vsini_fisher(self, N, kappa, sigma_lnv, v_values, vsini_obserr, seed=123):
        N = int(N)
        self.N = N
        self.sigma_lnv = sigma_lnv
        self.vsini_obserr = vsini_obserr
        self.fisher(N, kappa)
        np.random.seed(seed)
        self.v1 = v_values
        self.dlnv = np.random.randn(N) * sigma_lnv  # true v ratio; ln(v2/v1)
        self.v2 = self.v1 * np.exp(self.dlnv)  # true v2
        self.vsini1 = self.v1 * self.sini1
        self.vsini2 = self.v2 * self.sini2
        self.vsini1_obs = np.random.normal(self.vsini1, vsini_obserr)
        self.vsini2_obs = np.random.normal(self.vsini2, vsini_obserr)

    def vsini_iso(self, N, sigma_lnv, v_values, vsini_obserr, seed=123):
        N = int(N)
        self.N = N
        self.sigma_lnv = sigma_lnv
        self.vsini_obserr = vsini_obserr
        np.random.seed(seed)
        self.cosi1 = np.random.rand(N)
        self.cosi2 = np.random.rand(N)
        self.sini1 = np.sqrt(1 - self.cosi1**2)
        self.sini2 = np.sqrt(1 - self.cosi2**2)
        self.v1 = v_values
        self.dlnv = np.random.randn(N) * sigma_lnv  # true v ratio; ln(v2/v1)
        self.v2 = self.v1 * np.exp(self.dlnv)  # true v2
        self.vsini1 = self.v1 * self.sini1
        self.vsini2 = self.v2 * self.sini2
        self.vsini1_obs = np.random.normal(self.vsini1, vsini_obserr)
        self.vsini2_obs = np.random.normal(self.vsini2, vsini_obserr)

    def get_v_samples(self, n_smp=2000, seed=123):
        np.random.seed(seed)
        vsmp1, vsmp2, cosismp1, cosismp2 = [], [], [], []
        for i in range(self.N):
            vsinismp1 = np.random.normal(
                self.vsini1_obs[i], self.vsini_obserr, size=n_smp)
            vsinismp2 = np.random.normal(
                self.vsini2_obs[i], self.vsini_obserr, size=n_smp)
            cosi = np.random.rand(n_smp)
            sini = np.sqrt(1. - cosi**2)
            v1 = vsinismp1 / sini
            cosismp1.append(cosi)
            vsmp1.append(v1)
            cosi = np.random.rand(n_smp)
            sini = np.sqrt(1. - cosi**2)
            v2 = vsinismp2 / sini
            cosismp2.append(cosi)
            vsmp2.append(v2)
        vsmp1, vsmp2, cosismp1, cosismp2 = np.array(vsmp1), np.array(
            vsmp2), np.array(cosismp1), np.array(cosismp2)
        dlnvsmp = np.log(vsmp2) - np.log(vsmp1)
        return dlnvsmp, vsmp1, vsmp2, cosismp1, cosismp2
