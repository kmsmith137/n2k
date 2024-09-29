import sys
import pickle
import itertools

import numpy as np
import scipy.signal
import scipy.special
import scipy.optimize
import matplotlib.pyplot as plt


def logspace(lo, hi, n):
    assert n >= 2
    assert 0 < lo < hi
    return np.exp(np.linspace(np.log(lo), np.log(hi), n))

    
def is_power_of_two(n):
    return (n > 0) and (n & (n-1)) == 0


def round_up_to_power_of_two(n):
    assert n > 0
    return (1 << int(np.log2(n-0.5) + 1))


def is_perfect_square(n):
    if n < 1:
        return False
    m = int(n**0.5 + 0.5)
    return n == m**2


def savefig(filename):
    print(f'Writing {filename}')
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()

    
def write_pickle(filename, x):
    print(f'Writing {filename}', file=sys.stderr)
    with open(filename, 'wb') as f:
        pickle.dump(x, f)


def read_pickle(filename, verbose=True):
    print(f'Reading {filename}', file=sys.stderr)
    with open(filename, 'rb') as f:
        return pickle.load(f)


####################################################################################################


def ipow(x, n):
    """(x ** 1000) is criminally slow in numpy -- this function is a workaround."""
    
    assert n >= 0

    if n == 0:
        return np.ones_like(x)
    
    c = None
    y = x
    
    while True:
        # Invariant: at top of loop, we want to return (c * y^n)
        # If c is None, then c == 1
        
        if (n & 1):
            if c is None:
                c = np.copy(y)
            else:
                c *= y

        if n == 1:
            return c

        if y is x:
            y = x*x
        else:
            y *= y
        
        n >>= 1


def test_ipow():
    """This test can be run from the command line with 'python -m sk_bias test'."""
    
    for n in range(1,100):
        nelts = np.random.randint(1000, 2000)
        logr = np.random.uniform(-0.1, 0.1, nelts)
        a = np.exp(logr) * np.exp(1j * np.random.uniform(0,2*np.pi,size=nelts))
        
        aslow = a**n
        afast = ipow(a, n)
        epsilon = np.max(np.abs(aslow-afast) / np.exp(n*logr))
        
        print(f'{n=} {epsilon=}')
        assert np.all(epsilon < 1.0e-12)


####################################################################################################


def fit_polynomial(xvec, yvec):
    """Super-stable and gratuitously slow!"""
    
    d = len(xvec)-1
    assert d >= 0
    assert xvec.shape == yvec.shape == (d+1,)
    assert np.all(xvec[:-1] < xvec[1:])

    mat = np.zeros((d+1,d+1))
    mult = np.zeros(d+1)
    coeffs = np.zeros(d+1)
    residual = np.copy(yvec)
    
    for i in range(d+1):
        row = xvec**i if (i > 0) else np.ones(d+1)
        mult[i] = np.dot(row,row)**(-0.5)
        mat[i,:] = row * mult[i]

        coeffs[i] = np.dot(mat[i,:], residual)
        residual -= coeffs[i] * mat[i,:]
        
    coeffs += np.linalg.solve(mat.T, residual)
    coeffs *= mult
    return coeffs


def test_fit_polynomial():
    """This test can be run from the command line with 'python -m sk_bias test'."""
    
    for iouter in range(5):
        for d in range(1, 7):
            maxsep = np.exp(np.random.uniform(-5,5))
            dx = np.random.uniform(0.1*maxsep, maxsep, size=d+1)
            xvec = np.cumsum(dx)
            xvec -= np.mean(xvec)
            xvec += np.random.uniform(-10*maxsep, 10*maxsep)
            yvec = np.random.normal(size=d+1)
            
            coeffs = fit_polynomial(xvec, yvec)
            
            zvec = np.zeros(d+1)
            for i in range(d+1):
                zvec += coeffs[i] * xvec**i

            eps = np.max(np.abs(yvec-zvec))
            print(f'test_fit_polynomial({d=}): {eps=}')


####################################################################################################


class MCTracker:
    def __init__(self):
        self.n = 0
        self.mean = 0
        self.var = 0

    
    def update(self, s):
        assert s.ndim == 1
        
        if len(s) == 0:
            return
        
        ns = len(s)
        smean = np.mean(s)
        
        ds = s - smean
        ssum2 = np.dot(ds,ds)

        no = self.n
        omean = self.mean
        osum2 = (no-1) * self.var

        n = no + ns
        mean = (no*omean + ns*smean) / n
        sum2 = ssum2 + osum2 + no*(omean-mean)**2 + ns*(smean-mean)**2

        self.n = n
        self.mean = mean
        self.var = (sum2/(n-1)) if (n > 1) else 0

        
    @classmethod
    def test(cls):
        """This test can be run from the command line with 'python -m sk_bias test'."""
        
        for iouter in range(10):
            ntot = np.random.randint(5, 11)
            x = np.random.normal(size=ntot)
        
            tracker = MCTracker()
            pos = 0
        
            while pos < ntot:
                end = min(pos + np.random.randint(1,4), ntot)
                tracker.update(x[pos:end])
                pos = end

            print(f'MCTracker.test() iteration {iouter}')
            print(f'    {tracker.n=} {len(x)=}')
            print(f'    {tracker.mean=} {np.mean(x)=}')
            print(f'    {tracker.var=} {np.var(x,ddof=1)=}')



####################################################################################################


def run_unquantized_mcs(n):
    """Checks some expressions in the notes for <SK> and Var(SK) to all orders in n.
    This function doesn't really belong in n2k, but it's convenient to sneak it in.

    This function can be run from the command line with 'python -m sk_bias run_unquantized_mcs'.
    """

    nbatch = (2**21 // n) + 1
    tracker = MCTracker()
    print(f'run_unquantized_mcs({n=})')

    for iouter in itertools.count(1):
        e = np.random.normal(size=(nbatch,n,2))
        e2 = np.sum(e**2, axis=2)
        s1 = np.sum(e2, axis=1)
        s2 = np.sum(e2**2, axis=1)
        sk = (n+1)/(n-1) * (n*s2/s1**2 - 1)

        tracker.update(sk)

        if is_perfect_square(iouter):
            print(f'   nmc={tracker.n}  mean_sk={tracker.mean}  var_sk={tracker.var}')


def run_transit_mcs(nt, ndish, brightness):
    """Checks some expressions in the notes for Var(SK) during a bright source transit.
    This function doesn't really belong in n2k, but it's convenient to sneak it in.

    This function can be run from the command line with 'python -m sk_bias run_transit_mcs'.
    """

    nbatch = (2**21 // (nt*ndish)) + 1
    tracker = MCTracker()
    print(f'run_transit_mcs({nt=}, {ndish=}, {brightness=})')

    var0 = 4./nt/(2*ndish)
    var1 = var0 * (1 + (ndish-1) * brightness**4 / (1+brightness)**4)
    
    for iouter in itertools.count(1):
        e = np.random.normal(size=(nbatch,2,ndish,nt,2))                       # noise
        e += np.random.normal(size=(nbatch,2,1,nt,2), scale=brightness**0.5)   # source
        e2 = np.sum(e**2, axis=4)     # (mc,pol,dish,t)
        s1 = np.sum(e2, axis=3)       # (mc,pol,dish)
        s2 = np.sum(e2**2, axis=3)    # (mc,pol,dish)
        
        sk = (nt+1)/(nt-1) * (nt*s2/s1**2 - 1)   # (mc,pol,dish)
        sk = np.mean(sk, axis=2)                 # (mc,pol)
        sk = np.mean(sk, axis=1)                 # (mc,)
        assert sk.shape == (nbatch,)
        
        tracker.update(sk)

        if is_perfect_square(iouter):
            print(f'   nmc={tracker.n}  mean_sk={tracker.mean}  var_sk={tracker.var}   {var0=}  {var1=}')


####################################################################################################


class Pdf:
    _edges = np.array([ 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5 ])
    _rms_min = 0.07
    _rms_max = 1.0e6
    
    def __init__(self, rms):
        """Represents a quantized, clipped Gaussian PDF.
        Take rms=None to get a 'saturating' PDF, which always takes values +/- 7.
        """

        if rms is not None:
            assert self._rms_min <= rms <= self._rms_max
            cdf_edges = scipy.special.erf(self._edges / (2**0.5 * rms))
        else:
            cdf_edges = np.zeros(7)
            
        self.rms = rms
        self.p = np.zeros(8)
        self.p[0] = cdf_edges[0]
        self.p[-1] = 1 - cdf_edges[-1]
        self.p[1:-1] = cdf_edges[1:] - cdf_edges[:-1]

        # Means <x^n>
        ix = np.arange(8, dtype=float)
        x2 = np.sum(self.p * ix**2)
        x4 = np.sum(self.p * ix**4)
        x6 = np.sum(self.p * ix**6)
        x8 = np.sum(self.p * ix**8)
        
        mean_s1 = 2*x2              # <x^2 + y^2>
        mean_s2 = 2*x4 + 2*x2*x2    # <x^4 + 2x^2y^2 + y^4>
        mean_sk = mean_s2 / mean_s1**2 - 1

        # Var(S1) = Var(x^2 + y^2)
        #         = 2 Var(x^2) 
        #
        # Var(S2) = Var(x^4 + 2x^2y^2 + y^4)
        #         = 2 Var(x^4) + 8 Cov(x^4,x^2y^2) + 4 Var(x^2y^2)
        #
        # Cov(S1,S2) = Cov(x^2 + y^2, x^4 + 2x^2y^2 + y^4)
        #            = 2 Cov(x^2,x^4) + 4 Cov(x^2,x^2y^2)
        #
        # Var(SK) = Var(S2/S1^2)
        #         = < [ (dS2)/S1^2 - 2 (dS1) (S2/S1^3) ]^2 >
        #         = Var(S2)/S1^4 - 4 Cov(S1,S2) (S2/S1^5) + 4 Var(S1) (S2^2/S1^6)

        var_s1 = 2*(x4-x2*x2)
        var_s2 = 2*(x8-x4*x4) + 8*(x6-x2*x4)*x2 + 4*(x4**2-x2**4)
        cov_s12 = 2*(x6-x2*x4) + 4*(x4-x2*x2)*x2
        var_sk = var_s2/mean_s1**4 - 4*cov_s12*mean_s2/mean_s1**5 + 4*var_s1*mean_s2**2/mean_s1**6
        
        self.mu = mean_s1
        self.b_large_n = mean_sk - 1
        self.sigma_large_n = np.sqrt(var_sk)   # sqrt(N) * sigma


    @classmethod
    def from_mu(cls, mu):
        if abs(mu-98) < 1.0e-10:
            return cls(rms=None)
        
        assert mu >= 1.0e-6
        assert mu <= 97.999
        
        log_rms_min = np.log(cls._rms_min) + 1.0e-10   # rms=0.07 gives 0 < mu < 1.0e-6
        log_rms_max = np.log(cls._rms_max) - 1.0e-10   # rms=1.0e6 gives mu > 97.999

        ret = scipy.optimize.bisect(
            lambda log_rms: np.log(cls(np.exp(log_rms)).mu) - np.log(mu),
            log_rms_min, log_rms_max)

        return cls(rms = np.exp(ret))
    

    def __str__(self):
        return f'Pdf(rms={self.rms}, mu={self.mu}, b_large_n={self.b_large_n}, sigma_large_n={self.sigma_large_n})'


    def get_p1_p2_p3(self, n):
        """Returns length-(98*n+1) vectors p1, p2, p3.
        
            p1[s] = (probability of S1=s)
            p2[s] = (probability of S1=s) * (expectation value of S2, given S1=s)
            p3[s] = (probability of S1=s) * (expectation value of S2^2, given S1=s)

        (Note: the variable names p1,p2,p3 aren't very good!)
        """
        assert n >= 2
        
        p0 = np.zeros(50)
        for i in range(8):
            p0[i**2] = self.p[i]

        p1 = np.convolve(p0, p0)
        p2 = p1 * np.arange(99)**2
        p3 = p1 * np.arange(99)**4

        nout = 98*n+1
        npad = round_up_to_power_of_two(nout)

        q1 = np.fft.rfft(p1, n=npad)
        q2 = np.fft.rfft(p2, n=npad)
        q3 = np.fft.rfft(p3, n=npad)
        
        qn1 = ipow(q1, n-2)   # p1^{n-2}
        qn2 = n * (qn1*q2)    # n p1^{n-2} p2
        qn3 = n * (qn1*q1*q3) + (n-1) * (qn2*q2)   # n p1^{n-1} p3 + n(n-1) p1^{n-2} p2^2
        qn1 *= (q1*q1)        # p1^n
        qn2 *= q1             # n p1^{n-1} p2
        
        p1 = np.fft.irfft(qn1)[:nout]
        p2 = np.fft.irfft(qn2)[:nout]
        p3 = np.fft.irfft(qn3)[:nout]

        return p1, p2, p3


    @staticmethod
    def _bias_and_sigma_from_p1_p2_p3(p1, p2, p3, min_s1=None, max_s1=None, bvec=None):
        """Returns (b,sigma), given (p1,p2,p3) and an optional bvec.

        If 'bvec' is specified, it is a length-(98*n+1) array (i.e. same shape as (p1,p2,p3))
        which is indexed as bvec[S1].

        Recall definition of SK:
           sk = (n+1)/(n-1) * (n*S2/S1^2 - 1) - b(S1)

        In this function it's convenient to factorize as follows
           x = S2 / S1^2
           y = t*x - b(S1)          where t = n*(n+1)/(n-1)
           sk = y - (n+1)/(n-1)
        """
        
        n = p1.size // 98
        assert p1.shape == p2.shape == (98*n+1,)

        if min_s1 is None:
            min_s1 = 1
        if max_s1 is None:
            max_s1 = 98*n
        
        assert 1 <= min_s1 <= max_s1 <= 98*n
        i, j = min_s1, (max_s1+1)

        p1 = p1[i:j]
        px = p2[i:j] / np.arange(i,j)**2
        px2 = p3[i:j] / np.arange(i,j)**4
        b = bvec[i:j] if (bvec is not None) else np.zeros(j-i)

        # y = tx-b
        t = n*(n+1)/(n-1)
        py = t*px - b*p1
        py2 = (t*t)*px2 - (2*t)*(b*px) + b*b*p1

        den = np.sum(p1)
        assert den >= 1.0e-12

        mean_y = np.sum(py) / den
        mean_y2 = np.sum(py2) / den

        b = mean_y - (n+1)/(n-1) - 1
        sigma = np.sqrt(mean_y2 - mean_y**2)
        return b, sigma

    
    def get_bias_and_sigma(self, n, min_s1=None, max_s1=None, bvec=None):
        """Returns b = <sk>-1 and sigma=Var(sk)^(1/2).

        If 'bvec' is specified, it is a length-(98*n+1) array (i.e. same shape as (p1,p2,p3))
        which is indexed as bvec[S1].
        
        Note:  sk = (n+1)/(n-1) * (n*S2/S1^2 - 1) - b(S1)
        """

        if n is None:
            # Ignore (min_s1, max_s1, bvec)
            return self.b_large_n, 0.0
        else:
            p1, p2, p3 = self.get_p1_p2_p3(n)
            b, sigma = self._bias_and_sigma_from_p1_p2_p3(p1, p2, p3, min_s1, max_s1, bvec)
            return b, sigma


    def simulate_sk(self, n, nmc, min_s1=None, max_s1=None, bvec=None):
        """Returns 1-d array of length <= nmc, containing SK-statistics.
        The length of the array can be < nmc, if "clipped" by [min_s1,max_s1].
        """
        
        if self.rms is not None:
            e = np.random.normal(scale=self.rms, size=(nmc,n,2))
            e = np.round(e)
            e = np.maximum(e, -7.)
            e = np.minimum(e, 7.)
        else:
            e = np.full((nmc,n,2), fill_value=7.0)

        e2 = np.sum(e**2, axis=2)
        s1 = np.sum(e2, axis=1)
        s2 = np.sum(e2**2, axis=1)

        s1 = np.array(s1 + 0.5, dtype=int)   # convert float->int
        min_s1 = min_s1 if (min_s1 is not None) else 1
        valid = (s1 >= min_s1)

        if max_s1 is not None:
            valid = np.logical_and(valid, s1 <= max_s1)

        s1 = s1[valid]
        s2 = s2[valid]
        
        assert np.all(s1 > 0)
        sk = (n+1)/(n-1) * (n*s2/s1**2 - 1)

        if bvec is not None:
            assert bvec.shape == (98*n+1,)
            sk -= bvec[s1]
        
        return sk
        

    def run_mcs(self, n, nbatch=None, min_s1=None, max_s1=None, bvec=None):
        """Called by BiasInterpolator.run_mcs()"""
        
        if nbatch is None:
            nbatch = 2**21 // n

        predicted_bias, predicted_sigma = self.get_bias_and_sigma(n, min_s1, max_s1, bvec)
        tracker = MCTracker()
        
        print(self)
        print(f'Running Monte Carlos with {n=}')

        for iouter in itertools.count(1):
            sk = self.simulate_sk(n, nbatch, min_s1, max_s1, bvec)
            tracker.update(sk - 1.0)

            if is_perfect_square(iouter):
                nmc = tracker.n
                pvalid = nmc / (iouter * nbatch)
                delta = tracker.mean - predicted_bias
                rms = np.sqrt(tracker.var) if (tracker.var > 0) else 0.0
                ivar = (1.0 / tracker.var) if (tracker.var > 0) else 0.0
                sigmas = delta * (ivar * nmc)**0.5
                print(f'    nmc={nmc}  {pvalid=}  mean_bias={tracker.mean}'
                      + f'  predicted_mean={predicted_bias}  delta={delta} ({sigmas:.04f} sigma)'
                      + f'  {rms=}  predicted_rms={predicted_sigma}  ratio={rms/predicted_sigma}')


####################################################################################################


class BiasInterpolator:
    def __init__(self, *, mu_min=1.0, mu_max=90.0, num_mu=128, n_min=64, remove_zero=False):
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.n_min = n_min
        self.remove_zero = remove_zero

        # x = log(mu)
        self.nx = num_mu
        self.dx = (np.log(mu_max) - np.log(mu_min)) / (num_mu-3)
        self.xmin = np.log(mu_min) - self.dx
        self.xmax = np.log(mu_max) + self.dx
        self.xvec = np.linspace(self.xmin, self.xmax, num_mu)

        # y = 1/n
        self.ny = 4   # cubic interpolation
        self.yvec = np.linspace(0, 1/n_min, self.ny)
        self.nlist = [ None ] + [ int(1/y + 0.5) for y in self.yvec[1:] ]
        self.yvec[1:] = 1.0 / np.array(self.nlist[1:])

        print(f'{self.yvec=}')
        print(f'{self.nlist=}')
                
        # b = bias
        self.bmat = np.zeros((self.nx, self.ny))
        self.bcoeffs = np.zeros((self.nx, self.ny))

        for i,x in enumerate(self.xvec):
            mu = np.exp(x)
            pdf = Pdf.from_mu(mu)
            for j in range(self.ny):
                n = self.nlist[j]
                # min_s1, max_s1 = self.minmax_list[j]
                self.bmat[i,j] = pdf.get_bias(n)

            self.bcoeffs[i,:] = fit_polynomial(self.yvec, self.bmat[i,:])

        self.binterp = [ scipy.interpolate.KroghInterpolator(self.xvec[i:(i+4)], self.bcoeffs[i:(i+4),:])
                         for i in range(self.nx-3) ]

            
    def interpolate(self, *, mu, n):
        x = np.log(mu)
        y = 0.0 if (n is None) else (1.0/n)

        t = (x - self.xvec[0]) / self.dx
        nx = self.nx
        
        assert t > (1 - 1.0e-10)
        assert t < (nx-2 + 1.0e-10)
        assert (n is None) or (n >= self.n_min)

        i = int(t + 0.5)
        i = max(i, 1)
        i = min(i, nx-3)

        c = self.binterp[i-1](x)
        assert len(c) == self.ny

        ret = 0
        ypow = 1
        for j in range(self.ny):
            ret += c[j] * ypow
            ypow *= y

        return ret
        
        ypow = np.ones(self.ny)
        for j in range(1, self.ny):
            ypow[j] = ypow[j-1] * y
        
        xslice = self.xvec[(i-1):(i+3)]
        # bslice = np.array([ neville(self.yvec, b, y) for b in self.bmat[(i-1):(i+3),:] ])
        bslice = np.dot(self.bcoeffs[(i-1):(i+3),:], ypow)

        return neville(xslice, bslice, x)


    def eval_exact(self, *, mu, n):
        """Only called in check_interpolation()."""
        pdf = Pdf.from_mu(mu)
        return pdf.get_bias(n)

    
    def check_interpolation(self):
        mu_vec = logspace(self.mu_min, self.mu_max, 5 * self.nx)
        
        nvec = logspace(self.n_min, 10 * self.n_min, 10*self.ny)[1:-1:2]
        nvec = np.array(nvec+0.5, dtype=int)
        print(f'{nvec=}')
        
        maxdiff = 0
        argmax = 0.0

        for mu in mu_vec:
            for n in nvec:
                b_exact = self.eval_exact(mu=mu, n=n)
                b_interp = self.interpolate(mu=mu, n=n)
                diff = np.abs(b_exact - b_interp)
                if maxdiff <= diff:
                    maxdiff = diff
                    argmax = (mu,n)

        print(f'maxdiff={maxdiff} at (mu,n)={argmax}')
    

    def get_interpolated_bvec(self, n):
        assert n is not None
        
        eps = 5.0e-11
        min_s1 = n * self.mu_min
        min_s1 = int((1-eps)*min_s1 + 1)  # round up        
        max_s1 = n * self.mu_max
        max_s1 = int((1+eps)*max_s1)      # round down
        assert 0 < min_s1 <= max_s1 <= 98*n
        
        bvec = np.zeros(98*n+1)
        for s1 in range(min_s1, max_s1+1):
            bvec[s1] = self.interpolate(mu=s1/n, n=n)

        return min_s1, max_s1, bvec
    
        
    def make_plot(self, pdf_outfile):
        rms_min = Pdf.from_mu(self.mu_min).rms
        rms_max = Pdf.from_mu(self.mu_max).rms
        rms_vec = logspace(rms_min, rms_max, 200)

        # Pairs (n, color)
        todo = [
            (64, (1, 0, 0)),
            (85, (0.66, 0, 0.33)),
            (107, (0.33, 0, 0.66)),
            (128, (0, 0, 1)),
            (256, (0, 0.33, 0.66)),
            (512, (0, 0.66, 0.33)),
            (1024, (0, 1, 0))
        ]
        
        pdf_list = [ Pdf(rms) for rms in rms_vec ]
        n_list = [ 48*(64+8*i) for i in range(9) ]
        color_list = [ (i/8,0,1-i/8) for i in range(9) ]

        for (n,color) in todo:
            min_s1, max_s1, bvec = self.get_interpolated_bvec(n)
            bias_list = [ pdf.get_bias(n, min_s1, max_s1, bvec) for pdf in pdf_list ]
            sigma_vec = np.array(bias_list) / (2/n**0.5)

            # Plot positive and negative parts
            plt.loglog(rms_vec, np.maximum(sigma_vec, 1.0e-6), color=color, ls='-', label=f'n={n}')
            plt.loglog(rms_vec, np.maximum(-sigma_vec, 1.0e-6), color=color, ls=':')

        plt.xlabel('RMS (bits)')
        plt.ylabel('SK bias after correction (sigmas)')
        plt.legend(loc='upper right')
        plt.ylim(1.0e-4, 1.0)
        savefig(pdf_outfile)


    def run_mcs(self, rms, n, nbatch=None, verbose=False):
        min_s1, max_s1, bvec = self.get_interpolated_bvec(n)

        if verbose:
            print(f'{min_s1=}')
            print(f'{max_s1=}')
            for s1 in range(min_s1, max_s1+1):
                print(f'   bvec[{s1}] = {bvec[s1]}')
        
        pdf = Pdf(rms)
        pdf.run_mcs(n, nbatch, min_s1, max_s1, bvec)


    def emit_code(self):
        nx, ny = self.nx, self.ny

        print('// Autogenerated by python -m sk_bias emit_code')
        print()
        print('namespace n2k {')
        print('namespace sk_globals {')
        print()
        print(f'double mu_min = {self.mu_min};')
        print(f'double mu_max = {self.mu_max};')
        print(f'double xmin = {self.xmin};')
        print(f'double xmax = {self.xmax};')
        print()
        print(f'int nx = {nx};')
        print(f'int ny = {ny};')
        print(f'int n_min = {self.n_min};')
        print()
        print(f'static double bias_coeffs[{nx*ny}] = {{')

        for i in range(nx):
            for j in range(ny):
                print(f'  {self.bcoeffs[i,j]}', end='')
                if (i < (nx-1)) or (j < (ny-1)):
                    print(',', end='')
                if (j == (ny-1)):
                    print()

        print('};')
        print()
        print('double *get_bias_coeffs() { return bias_coeffs; }')
        print()
        print('}}  // namespace n2k::global_sk')
