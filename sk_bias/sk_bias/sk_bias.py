import itertools
import numpy as np
import scipy.signal
import scipy.special
import scipy.optimize
import matplotlib.pyplot as plt


def logspace(lo, hi, n):
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


class Pdf:
    _edges = np.array([ 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5 ])
    _rms_min = 0.07
    _rms_max = 1.0e6
    
    def __init__(self, rms):
        """Take rms=None to get a 'saturating' PDF, which always takes values \pm 7."""

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

        ix = np.arange(8, dtype=float)
        x2 = np.sum(self.p * ix**2)
        x4 = np.sum(self.p * ix**4)
        
        self.mu = 2*x2
        self.b_large_n = (2*x4 + 2*x2*x2) / (2*x2)**2 - 2


    @classmethod
    def from_mu(cls, mu):
        if mu == 98:
            # FIXME a little fragile (should allow some roundoff error in comparison)
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
        return f'Px(rms={self.rms}, mu={self.mu}, b_large_n={self.b_large_n})'


    def analyze_bias(self, n, bvec=None, remove_zero=False):
        assert n > 1

        if bvec is not None:
            assert bvec.shape == (98*n+1,)
        
        p0 = np.zeros(50)
        for i in range(8):
            p0[i**2] = self.p[i]

        p1 = np.convolve(p0, p0)
        p2 = p1 * np.arange(99)**2

        nout = 98*n+1
        npad = round_up_to_power_of_two(nout)

        q1 = np.fft.rfft(p1, n=npad)
        q2 = np.fft.rfft(p2, n=npad)

        qn1 = ipow(q1, n-1)
        qn2 = n*qn1*q2
        qn1 *= q1
        
        p1 = np.fft.irfft(qn1)[:nout]
        p2 = np.fft.irfft(qn2)[:nout]
                
        if remove_zero:
            t = np.sum(p1[1:])
            assert t > 0.0
            
            p1[0] = p2[0] = 0.
            p1 /= t
            p2 /= t
            
        mean_s2_s1 = np.sum(p2[1:] / np.arange(1,98*n+1)**2)
        mean_sk = (n+1) / (n-1) * (n*mean_s2_s1 - 1)
        mean_sk += p1[0]   # Define SK=1 if S1=0

        if bvec is not None:
            mean_sk -= np.dot(p1, bvec)
            
        return mean_sk, p1, p2


    def get_bias(self, n, remove_zero=False):
        if n is None:
            return self.b_large_n
        else:
            mean_sk, p1, p2 = self.analyze_bias(n,remove_zero=True)
            return mean_sk - 1.0


    def simulate_sk(self, *, n, nmc, bvec=None):
        """Returns length-nmc array containing SK statistics."""
            
        if bvec is not None:
            assert bvec.shape == (98*n+1,)
            
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
        
        sk = (n+1)/(n-1) * (n*s2/np.maximum(s1,1)**2 - 1)
        sk = np.where(s1 > 0, sk, 1.0)   # Define SK=1 if S1=0

        if bvec is not None:
            s1 = np.array(s1 + 0.5, dtype=int)
            assert np.all(s1 >= 0)
            assert np.all(s1 <= 98*n)
            sk -= bvec[s1]
        
        return sk
        

    def run_mcs(self, n, nbatch=None, bvec=None):
        if nbatch is None:
            nbatch = 2**21 // n
        
        # Currently runs forever!
        mean_sk, p1, p2 = self.analyze_bias(n, bvec=bvec)
        tracker = MCTracker()
        print(self)
        print(f'Running Monte Carlos with {n=}')

        for iouter in itertools.count(1):
            sk = self.simulate_sk(n=n, nmc=nbatch, bvec=bvec)
            tracker.update(sk)

            if is_perfect_square(iouter):
                nmc = tracker.n
                delta = tracker.mean - mean_sk
                sigmas = delta / (tracker.var/nmc)**0.5
                print(f'    nmc={nmc}  mc_sk={tracker.mean}  predicted_mean={mean_sk}  delta={delta}  sigmas={sigmas}')


####################################################################################################


class MCTracker:
    def __init__(self):
        self.n = 0
        self.mean = 0
        self.var = 0

    
    def update(self, s):
        assert s.ndim == 1
        assert len(s) > 0
        
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


def neville(datax, datay, x):
    """
    Finds an interpolated value using Neville's algorithm.

    Input
      datax: input x's in a list of size n
      datay: input y's in a list of size n
      x: the x value used for interpolation

    Output
      p[0]: the polynomial of degree n
    """
    n = len(datax)
    p = n*[0]
    for k in range(n):
        for i in range(n-k):
            if k == 0:
                p[i] = datay[i]
            else:
                p[i] = ((x-datax[i+k])*p[i]+ \
                        (datax[i]-x)*p[i+1])/ \
                        (datax[i]-datax[i+k])
    return p[0]


class BiasInterpolator:
    def __init__(self, *, mu_min=1.0, mu_max=90.0, num_mu=100, n_min=64, remove_zero=False):
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

        for i,x in enumerate(self.xvec):
            mu = np.exp(x)
            pdf = Pdf.from_mu(mu)
            for j,n in enumerate(self.nlist):
                self.bmat[i,j] = pdf.get_bias(n, remove_zero=remove_zero)

            
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
        i = min(i, nx-2)

        xslice = self.xvec[(i-1):(i+3)]
        bslice = np.array([ neville(self.yvec, b, y) for b in self.bmat[(i-1):(i+3),:] ])

        return neville(xslice, bslice, x)


    def eval_exact(self, *, mu, n):
        pdf = Pdf.from_mu(mu)
        return pdf.get_bias(n, remove_zero=self.remove_zero)


    def check(self):
        print(f'{np.exp(self.xvec[-2])=}')
        print(f'{self.bmat[-2,:]=}')
        
        mu_vec = logspace(self.mu_min, self.mu_max, 5 * self.nx)
        
        nvec = logspace(self.n_min, 10 * self.n_min, 10*self.ny)[1:-1:2]
        nvec = np.array(nvec+0.5, dtype=int)
        print(f'{nvec=}')
        
        maxdiff = 0
        argmax = 0.0

        for mu in mu_vec:
        # for mu in logspace(self.mu_min, self.mu_max, (self.nx-2)):
            for n in nvec:
                b_exact = self.eval_exact(mu=mu, n=n)
                b_interp = self.interpolate(mu=mu, n=n)
                diff = np.abs(b_exact - b_interp)
                if maxdiff <= diff:
                    maxdiff = diff
                    argmax = (mu,n)

        print(f'maxdiff={maxdiff} at (mu,n)={argmax}')
        
