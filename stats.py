import numpy as np
import pandas as pd
from scipy import stats

class FrequentistTest:
    """
    Performs Frequentist statistical tests (Z-Test for proportions, Chi-Square for SRM).
    """
    
    def check_srm(self, df):
        """
        Checks for Sample Ratio Mismatch (SRM) using Chi-Square Goodness of Fit.
        Assumes a target 50/50 split.
        """
        group_counts = df['group'].value_counts()
        
        # Ensure we have both groups
        if 'A' not in group_counts or 'B' not in group_counts:
            return {'p_value': 0.0, 'srm_detected': True} # Technical fail
            
        observed = [group_counts['A'], group_counts['B']]
        n_total = sum(observed)
        expected = [n_total * 0.5, n_total * 0.5]
        
        chi2_stat, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
        
        return {
            'p_value': p_value,
            'srm_detected': p_value < 0.01 # Strict alpha for SRM usually
        }

    def analyze(self, df):
        """
        Calculates Lift, Z-Score, and P-Value (Two-sided).
        """
        # Aggregate data
        results = df.groupby('group')['converted'].agg(['count', 'sum'])
        
        # Check if we have data for both
        if 'A' not in results.index or 'B' not in results.index:
            return {'lift': 0, 'p_value': 1.0, 'significant': False}
            
        n_a = results.loc['A', 'count']
        conv_a = results.loc['A', 'sum']
        n_b = results.loc['B', 'count']
        conv_b = results.loc['B', 'sum']
        
        p_a = conv_a / n_a
        p_b = conv_b / n_b
        
        # Lift
        lift = (p_b - p_a) / p_a if p_a > 0 else 0
        
        # Z-Test for Proportions (Pooled Standard Error)
        p_pool = (conv_a + conv_b) / (n_a + n_b)
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
        
        if se == 0:
            z_score = 0
            p_value = 1.0
        else:
            z_score = (p_b - p_a) / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score))) # Two-sided
            
        return {
            'lift': lift,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'stats_a': {'n': n_a, 'cr': p_a},
            'stats_b': {'n': n_b, 'cr': p_b}
        }

class BayesianTest:
    """
    Performs Bayesian analysis using Beta-Bernoulli Conjugate Priors.
    """
    
    def analyze(self, df):
        """
        Calculates Probability of B being better than A.
        """
        results = df.groupby('group')['converted'].agg(['count', 'sum'])
        
        if 'A' not in results.index or 'B' not in results.index:
            return {'prob_b_wins': 0.5}

        # Priors (Beta(1,1) - Weak/Uniform Prior)
        alpha_prior = 1
        beta_prior = 1
        
        # Posteriors
        # A
        a_conv = results.loc['A', 'sum']
        a_n = results.loc['A', 'count']
        alpha_post_a = alpha_prior + a_conv
        beta_post_a = beta_prior + (a_n - a_conv)
        
        # B
        b_conv = results.loc['B', 'sum']
        b_n = results.loc['B', 'count']
        alpha_post_b = alpha_prior + b_conv
        beta_post_b = beta_prior + (b_n - b_conv)
        
        # Monte Carlo Simulation
        n_samples = 100000
        samples_a = np.random.beta(alpha_post_a, beta_post_a, n_samples)
        samples_b = np.random.beta(alpha_post_b, beta_post_b, n_samples)
        
        prob_b_wins = np.mean(samples_b > samples_a)
        
        return {
            'prob_b_wins': prob_b_wins,
            'posterior_a': {'alpha': alpha_post_a, 'beta': beta_post_a},
            'posterior_b': {'alpha': alpha_post_b, 'beta': beta_post_b}
        }
