"""
Corporate Credit Risk Data Pipeline
-----------------------------------
This module generates synthetic corporate accounting data and computes key financial ratios used in quantitative credit risk assessment.
"""

import numpy as np
import pandas as pd


def generate_corporate_financials(n_companies: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic dataset of corporate financial statements.
    
    Parameters:
        n_companies (int): Number of corporate records to generate (default 1000).
        seed (int): Random seed for reproducible results.
        
    Returns:
        pd.DataFrame: Raw balance sheet and income statement metrics.
    """
    np.random.seed(seed)
    
    # --- Section 1: Raw Financial Accounting Metrics ---
    # Total Assets: Company sizes ranging from $1M to $100M
    total_assets = np.random.uniform(1e6, 1e8, n_companies)
    
    # Total Liabilities: Scaled as 20% to 95% of Total Assets
    total_liabilities = total_assets * np.random.uniform(0.20, 0.95, n_companies)
    
    # Current Assets: Short-term liquidity assets (10% to 50% of Assets)
    current_assets = total_assets * np.random.uniform(0.10, 0.50, n_companies)
    
    # Current Liabilities: Short-term debt obligations (20% to 70% of Liabilities)
    current_liabilities = total_liabilities * np.random.uniform(0.20, 0.70, n_companies)
    
    # Operating Earnings (EBIT): -5% to +25% Return on Assets
    ebit = total_assets * np.random.uniform(-0.05, 0.25, n_companies)
    
    # Interest Expense: Cost of servicing debt (2% to 8% of Total Liabilities)
    interest_expense = np.maximum(total_liabilities * np.random.uniform(0.02, 0.08, n_companies), 1000)
    
    # Combine raw variables into a DataFrame
    df_raw = pd.DataFrame({
        'total_assets': total_assets,
        'total_liabilities': total_liabilities,
        'current_assets': current_assets,
        'current_liabilities': current_liabilities,
        'ebit': ebit,
        'interest_expense': interest_expense
    })
    
    return df_raw


def compute_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw accounting metrics into standard financial ratios used in credit risk evaluation.
    """
    df = df.copy()
    
    # 1. Leverage Ratio (Debt / Assets): Measures overall solvency risk
    df['leverage_ratio'] = df['total_liabilities'] / df['total_assets']
    
    # 2. Liquidity Ratio (Current Assets / Current Liabilities): Short-term debt coverage
    df['liquidity_ratio'] = df['current_assets'] / df['current_liabilities']
    
    # 3. Interest Coverage Ratio (EBIT / Interest Expense): Ability to pay debt interest
    df['interest_coverage'] = df['ebit'] / df['interest_expense']
    
    # 4. Return on Assets (EBIT / Total Assets): Operational profitability metric
    df['roa'] = df['ebit'] / df['total_assets']
    
    # --- Section 2: Target Variable Generation (Default Status) ---
    # High leverage, negative EBIT, and low liquidity increase the probability of default
    risk_score = (
        1.5 * df['leverage_ratio'] 
        - 0.8 * df['liquidity_ratio'] 
        - 0.5 * np.clip(df['interest_coverage'], -2, 5) 
        - 2.0 * df['roa'] 
        + np.random.normal(0, 0.4, len(df))
    )
    
    # Top 20% highest risk scores are classified as defaulted (1 = Default, 0 = Solvent)
    default_threshold = np.percentile(risk_score, 80)
    df['default_status'] = (risk_score >= default_threshold).astype(int)
    
    return df


if __name__ == "__main__":
    # Test pipeline execution locally
    raw_data = generate_corporate_financials()
    processed_data = compute_financial_ratios(raw_data)
    print("Pipeline Execution Successful:")
    print(processed_data[['leverage_ratio', 'liquidity_ratio', 'interest_coverage', 'roa', 'default_status']].head())
