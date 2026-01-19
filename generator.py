import numpy as np
import pandas as pd
import uuid

class SimulationGenerator:
    """
    Generates synthetic A/B test data with options for Chaos Engineering 
    (SRM, Simpson's Paradox).
    """

    def generate_data(self, n_users, baseline_rate, lift, inject_srm=False, inject_simpson=False):
        """
        Generates a DataFrame with user data.
        
        Args:
            n_users (int): Total number of users to simulate.
            baseline_rate (float): Baseline conversion rate (0.0 to 1.0).
            lift (float): Relative lift for the treatment group (e.g., 0.10 for +10%).
            inject_srm (bool): If True, introduces Sample Ratio Mismatch (30/70 split).
            inject_simpson (bool): If True, introduces Simpson's Paradox via 'device' variable.
            
        Returns:
            pd.DataFrame: Columns [user_id, group, device, converted, day_index]
        """
        
        # 1. Assign Groups
        if inject_srm:
            # SRM: Severe mismatch (e.g., 30% A, 70% B)
            p_a = 0.30
        else:
            # Normal: 50% split
            p_a = 0.50
            
        groups = np.random.choice(['A', 'B'], size=n_users, p=[p_a, 1-p_a])
        
        # Initialize DataFrame
        df = pd.DataFrame({
            'user_id': [str(uuid.uuid4()) for _ in range(n_users)],
            'group': groups,
            'day_index': np.random.randint(0, 14, size=n_users) # Random day 0-13
        })
        
        # 2. Assign Devices and Conversion Rates (Simpson's Logic)
        if inject_simpson:
            # Simpson's Paradox:
            # A gets mostly Desktop (High Conv), B gets mostly Mobile (Low Conv)
            # But B is better than A within each device.
            
            # Device Assignment Probabilities
            # A: 80% Desktop, 20% Mobile
            # B: 20% Desktop, 80% Mobile
            is_a = df['group'] == 'A'
            df.loc[is_a, 'device'] = np.random.choice(['Desktop', 'Mobile'], size=is_a.sum(), p=[0.8, 0.2])
            df.loc[~is_a, 'device'] = np.random.choice(['Desktop', 'Mobile'], size=(~is_a).sum(), p=[0.2, 0.8])
            
            # Define Base Rates per Device (Desktop converts 3x better than Mobile)
            # Center the rates around the provided baseline somewhat, roughly.
            # Let's say Mobile is Base, Desktop is Base * 3.
            # Actually, let's derive them so weights make sense? 
            # Simplification: Desktop = 20%, Mobile = 5% (adjusted by baseline scaling)
            
            base_mobile = baseline_rate * 0.5
            base_desktop = baseline_rate * 1.5
            
            # Apply Lift to B (Treatment)
            # B is better than A by 'lift' % within each device
            metrics = {
                'A_Desktop': base_desktop,
                'A_Mobile': base_mobile,
                'B_Desktop': base_desktop * (1 + lift),
                'B_Mobile': base_mobile * (1 + lift)
            }
            
        else:
            # Normal Case: Even distribution of devices, no confounding
            df['device'] = np.random.choice(['Desktop', 'Mobile'], size=n_users, p=[0.5, 0.5])
            
            metrics = {
                'A_Desktop': baseline_rate,
                'A_Mobile': baseline_rate, # Simplified: device doesn't matter by default or same
                'B_Desktop': baseline_rate * (1 + lift),
                'B_Mobile': baseline_rate * (1 + lift)
            }
            # Actually better to have natural device difference even in normal case?
            # User spec: "Simpson's Paradox: If activado, invierte los pesos... Normal: ??" 
            # We'll stick to simple uniform baseline for Normal to isolate variables.
        
        # 3. determine Conversion
        # Vectorized probability assignment
        conditions = [
            (df['group'] == 'A') & (df['device'] == 'Desktop'),
            (df['group'] == 'A') & (df['device'] == 'Mobile'),
            (df['group'] == 'B') & (df['device'] == 'Desktop'),
            (df['group'] == 'B') & (df['device'] == 'Mobile')
        ]
        
        probs = [
            metrics.get('A_Desktop', baseline_rate),
            metrics.get('A_Mobile', baseline_rate),
            metrics.get('B_Desktop', baseline_rate * (1+lift)),
            metrics.get('B_Mobile', baseline_rate * (1+lift))
        ]
        
        # Create a prob column
        df['prob'] = np.select(conditions, probs, default=baseline_rate)
        
        # Bernoulli Trial
        random_draws = np.random.rand(n_users)
        df['converted'] = (random_draws < df['prob']).astype(int)
        
        # Cleanup
        return df[['user_id', 'group', 'device', 'converted', 'day_index']]
