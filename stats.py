import numpy as np
import pandas as pd
from scipy import stats

class FrequentistTest:
    """
    Performs Frequentist statistical tests (Z-Test for proportions, T-Test for Revenue, Chi-Square for SRM).
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

    def analyze_conversion(self, df):
        """
        Calculates Lift, Z-Score, and P-Value (Two-sided) for Conversion Rate.
        """
        # Aggregate data
        results = df.groupby('group')['converted'].agg(['count', 'sum'])
        
        # Check if we have data for both
        if 'A' not in results.index or 'B' not in results.index:
            return {'lift': 0, 'p_value': 1.0, 'significant': False, 'stats_a': {'n':0, 'cr':0}, 'stats_b': {'n':0, 'cr':0}}
            
        n_a = results.loc['A', 'count']
        conv_a = results.loc['A', 'sum']
        n_b = results.loc['B', 'count']
        conv_b = results.loc['B', 'sum']
        
        p_a = conv_a / n_a if n_a > 0 else 0
        p_b = conv_b / n_b if n_b > 0 else 0
        
        # Lift
        lift = (p_b - p_a) / p_a if p_a > 0 else 0
        
        # Z-Test for Proportions (Pooled Standard Error)
        p_pool = (conv_a + conv_b) / (n_a + n_b) if (n_a + n_b) > 0 else 0
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b)) if n_a > 0 and n_b > 0 else 0
        
        if se == 0:
            z_score = 0
            p_value = 1.0
        else:
            z_score = (p_b - p_a) / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score))) # Two-sided
            
        return {
            'metric': 'conversion',
            'lift': lift,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'stats_a': {'n': n_a, 'mean': p_a},
            'stats_b': {'n': n_b, 'mean': p_b}
        }

    def analyze_revenue(self, df):
        """
        Calculates Lift, T-Score, and P-Value (Welch's T-Test) for Revenue (RPV).
        """
        # Revenue Per Visitor (including 0s)
        rev_a = df[df['group'] == 'A']['revenue']
        rev_b = df[df['group'] == 'B']['revenue']
        
        if len(rev_a) < 2 or len(rev_b) < 2:
             return {'lift': 0, 'p_value': 1.0, 'significant': False, 'stats_a': {'n':0, 'mean':0}, 'stats_b': {'n':0, 'mean':0}}
        
        mean_a = rev_a.mean()
        mean_b = rev_b.mean()
        
        # Lift
        lift = (mean_b - mean_a) / mean_a if mean_a > 0 else 0
        
        # Welch's T-Test (Does not assume equal variance)
        t_stat, p_value = stats.ttest_ind(rev_b, rev_a, equal_var=False)
        
        return {
            'metric': 'revenue',
            'lift': lift,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'stats_a': {'n': len(rev_a), 'mean': mean_a},
            'stats_b': {'n': len(rev_b), 'mean': mean_b}
        }
        
    def get_sequential_boundary(self, n_current, n_total_planned, alpha=0.05):
        """
        Calculates O'Brien-Fleming critical value for current step.
        """
        t = n_current / n_total_planned
        if t <= 0: return 999.0
        
        # O'Brien-Fleming Approximation (Pocock bounds logic but tighter early on)
        # Critical Value ~ Z_final / sqrt(t)
        # Z_final for alpha=0.05 is 1.96
        
        # Use simpler approximation for boundaries:
        # Boundary(t) = 1.96 / sqrt(t)  (Classic OBF shape)
        # Note: This is a simplification. Real OBF spending function is more complex.
        
        z_crit = 1.96 / np.sqrt(t)
        return min(z_crit, 8.0) # Cap at 8 to avoid inf

class BayesianTest:
    """
    Performs Bayesian analysis using Beta-Bernoulli (Conversion) or Monte Carlo (Revenue).
    """
    
    def analyze_conversion(self, df):
        """
        Calculates Probability of B being better than A for Conversion.
        """
        results = df.groupby('group')['converted'].agg(['count', 'sum'])
        
        if 'A' not in results.index or 'B' not in results.index:
            return {'prob_b_wins': 0.5, 'expected_loss': 0}

        # Priors
        alpha_prior = 1
        beta_prior = 1
        
        # Posteriors
        a_conv = results.loc['A', 'sum']
        a_n = results.loc['A', 'count']
        alpha_post_a = alpha_prior + a_conv
        beta_post_a = beta_prior + (a_n - a_conv)
        
        b_conv = results.loc['B', 'sum']
        b_n = results.loc['B', 'count']
        alpha_post_b = alpha_prior + b_conv
        beta_post_b = beta_prior + (b_n - b_conv)
        
        # Monte Carlo
        n_samples = 100000
        samples_a = np.random.beta(alpha_post_a, beta_post_a, n_samples)
        samples_b = np.random.beta(alpha_post_b, beta_post_b, n_samples)
        
        prob_b_wins = np.mean(samples_b > samples_a)
        
        # Expected Loss (Lift)
        # Loss = difference between max and chosen if we choose wrong.
        # If we choose B, Loss is (A - B) where A > B.
        loss_choosing_b = np.maximum(samples_a - samples_b, 0)
        expected_loss_b = np.mean(loss_choosing_b)
        
        return {
            'prob_b_wins': prob_b_wins,
            'expected_loss': expected_loss_b,
            'posterior_a': {'alpha': alpha_post_a, 'beta': beta_post_a},
            'posterior_b': {'alpha': alpha_post_b, 'beta': beta_post_b}
        }

    def analyze_revenue(self, df):
        """
        Calculates Probability of B > A for Revenue (RPV) using Bootstrap/Monte Carlo.
        Assuming Log-Normal behavior is complex in closed form, we use bootstrapping of means.
        """
        rev_a = df[df['group'] == 'A']['revenue'].values
        rev_b = df[df['group'] == 'B']['revenue'].values
        
        if len(rev_a) < 10 or len(rev_b) < 10:
             return {'prob_b_wins': 0.5, 'expected_loss': 0}
             
        # Bootstrap means
        n_boot = 5000
        means_a = np.random.choice(rev_a, size=(n_boot, len(rev_a)), replace=True).mean(axis=1)
        means_b = np.random.choice(rev_b, size=(n_boot, len(rev_b)), replace=True).mean(axis=1)
        
        prob_b_wins = np.mean(means_b > means_a)
        
        # Expected Loss (in currency units)
        loss_choosing_b = np.maximum(means_a - means_b, 0)
        expected_loss_b = np.mean(loss_choosing_b)
        
        return {
            'prob_b_wins': prob_b_wins,
            'expected_loss': expected_loss_b,
            'samples_a': means_a, # For visualization
            'samples_b': means_b
        }
