#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
from pyfinance.ols import OLS


def change_factor_frequency(
        factor_data: pd.DataFrame,
        date_list: None | list | np.ndarray = None,
        change_to: None | str = None
):
    """
Adjust Factor Data Frequency

Parameters
----------
factor_data : pd.DataFrame - MultiIndex  
    Factor data  
date_list : list or np.ndarray, optional  
    List of rebalancing dates (optional). If not provided, all dates appearing in `factor_data` will be used.  
change_to : str, optional, {'M', 'W'}  
    Frequency to convert to (optional):  
    * 'M': The last day of each month in the rebalancing date list  
    * 'W': The last day of each week in the rebalancing date list  

Returns
-------
pd.DataFrame - MultiIndex  
    Factor data with adjusted frequency  
"""

    factor_data = factor_data.copy()
    if date_list is None:
        date_list = factor_data.index.get_level_values('date').drop_duplicates().sort_values()
    if change_to is not None:
        date_df = pd.DataFrame(date_list)
        if change_to == 'M':
            date_df['mark'] = date_df['date'].dt.strftime('%Y-%m')
            date_df['day'] = date_df['date'].dt.day
        elif change_to == 'W':
            date_df['mark'] = date_df['date'].dt.strftime('%Y-%W')
            date_df['day'] = date_df['date'].dt.strftime('%w').astype('int')
        date_df = date_df[date_df['day'] == date_df.groupby('mark')['day'].transform('max')]
        date_list = date_df['date']
    factor_data = factor_data[factor_data.index.get_level_values('date').isin(date_list)]

    return factor_data


def process_outlier(
        factor_data: pd.DataFrame,
        method: str = 'winsorize',
        factor_list: None | list = None,
        winsorize_fraction: float = 0.01,
        n_sigma: float = 3,
        n_mad: float = 3
):
    """
Handle Outliers in Factor Data

Parameters
----------
factor_data : pd.DataFrame - MultiIndex  
    Factor data  
method : str, {'winsorize', 'sigma', 'mad'}, default 'winsorize'  
    Method for handling outliers (default: 'winsorize')  
    * 'winsorize': Winsorization  
    * 'sigma': n-sigma method  
    * 'mad': n-MAD method  
factor_list : list, optional  
    List of factor names to process (optional). If not provided, all factors will be processed.  
winsorize_fraction : float, optional  
    Only applicable when `method == 'winsorize'`. Specifies the quantile threshold for outliers.  
    Default is 0.01 (data beyond the 1st and 99th percentiles is considered an outlier).  
n_sigma : float, optional  
    Only applicable when `method == 'sigma'`. Specifies the number of standard deviations to define outliers.  
    Default is 3.  
n_mad : float, optional  
    Only applicable when `method == 'mad'`. Specifies the number of MAD (Median Absolute Deviation) values to define outliers.  
    Default is 3.  

Returns
-------
data : pd.DataFrame - MultiIndex  
    Factor data after outlier processing  
"""

    data = factor_data.copy()
    factor_list = list(data.columns) if factor_list is None else factor_list
    for factor in factor_list:
        if method == 'winsorize':
            data['upper'] = data.groupby('date')[factor].transform(lambda x: x.quantile(1 - winsorize_fraction))
            data['lower'] = data.groupby('date')[factor].transform(lambda x: x.quantile(winsorize_fraction))
        elif method == 'sigma':
            data['upper'] = data.groupby('date')[factor].transform(lambda x: x.mean() + n_sigma * x.std())
            data['lower'] = data.groupby('date')[factor].transform(lambda x: x.mean() - n_sigma * x.std())
        elif method == 'mad':
            data['upper'] = data.groupby('date')[factor]. \
                transform(lambda x: x.median() + n_mad * (x - x.median()).abs().median())
            data['lower'] = data.groupby('date')[factor]. \
                transform(lambda x: x.median() - n_mad * (x - x.median()).abs().median())
        data.loc[data[factor] > data['upper'], factor] = data['upper']
        data.loc[data[factor] < data['lower'], factor] = data['lower']
        data.drop(columns=['upper', 'lower'], inplace=True)
    return data


def standardize_factor(
        factor_data: pd.DataFrame,
        factor_list: None | list = None,
        suffix: str = '',
):
    """
    Cross-Sectional Standardization of Factors

    Parameters
    ----------
    factor_data : pd.DataFrame  
        Factor data  

    factor_list : list, optional  
        List of factor names to be standardized (optional). If not provided, all factors will be processed.  

    suffix : str, optional  
        Suffix for standardized factor columns (default: None).  
        * If not provided, the standardized factor data will overwrite the original factor data.  
        * If provided, the standardized factor fields will be added to `factor_data` with the specified suffix.  

    Returns
    -------
    data : pd.DataFrame - MultiIndex  
        Standardized factor data  
    """

    data = factor_data.copy()
    factor_list = list(data.columns) if factor_list is None else factor_list
    for factor in factor_list:
        data[f'{factor}{suffix}'] = data.groupby('date')[factor].transform(lambda x: (x - x.mean()) / x.std())
    return data


def combine_factors(
    factor_data: pd.DataFrame,
    factor_list: None | list = None,
    method: str = 'equal',
    standardization: bool = True,
) -> pd.Series:
    """
Factor Synthesis

Parameters
----------
factor_data : pd.DataFrame  
    Factor data  

factor_list : list[str], optional  
    List of factor names to be synthesized (optional). If not provided, all factors will be synthesized.  

method : str, {'equal'}, default 'equal'  
    Method for factor synthesis (default: 'equal').  
    * 'equal': Equal-weighted synthesis  

standardization : bool, default True  
    Whether to perform cross-sectional standardization on individual factors before synthesis.  

Returns
-------
pd.Series - MultiIndex  
    Synthesized factor data  
"""

    data = factor_data.copy()
    factor_list = list(data.columns) if factor_list is None else factor_list
    data = data[factor_list]
    if standardization:
        data = standardize_factor(data=data)
    if method == 'equal':
        return data.mean(axis=1)


def neutralize_factors(
        factor_data: pd.DataFrame,
        neutralization_list: list,
):
    """
    During neutralization, samples with missing values in risk factors will be removed.  
    Whether a risk factor is treated as quantitative or categorical (object, str) depends on its dtype.  

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex  
        Factor data  

    neutralization_list : list[pd.DataFrame]  
        List of risk factor data used for neutralization  

    Returns
    -------
    pd.DataFrame  
        Neutralized factor data  
    """
    #
    f_data=factor_data.copy()
    n_list=[i.copy() for i in neutralization_list]

    factor_list = list(f_data.columns)

    for data in [f_data, ] + n_list:
        data=data.reset_index().sort_values(['date','asset'])
    for n_data in n_list:
        f_data = pd.merge_asof(f_data, n_data, on='date', by='asset')
    f_data.set_index(['date', 'asset'], inplace=True)

    risk_list = [i for i in f_data.columns if i not in factor_list]

    df_list = []
    for date, sub_df in f_data.groupby('date'):
        df = sub_df.copy().dropna()
        if df.shape[0] == 0:
            continue
        for risk in risk_list:
            if pd.api.types.is_object_dtype(df[risk]) or pd.api.types.is_string_dtype(df[risk]):
                df = pd.merge(df, pd.get_dummies(df[risk], drop_first=True), left_index=True, right_index=True)
                df.drop(columns=[risk, ], inplace=True)
        y_list = factor_list
        x_list = [i for i in df.columns if i not in factor_list]
        for y in y_list:
            df[y] = OLS(y=df[y], x=df[x_list]).resids
        df = df[y_list]
        df_list.append(df)
    all_df = pd.concat(df_list)
    return all_df
