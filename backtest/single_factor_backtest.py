#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import quantstats.stats as qs
import matplotlib.dates as mdates
import datetime
import matplotlib.ticker as ticker

# Use seaborn style
# plt.style.use('seaborn-v0_8-ticks')
plt.style.use('default')

# # Set Chinese font to KaiTi
# # mpl.rcParams['font.sans-serif'] = ['KaiTi']
# mpl.rcParams['font.sans-serif'] = 'SimHei'  # Choose a font that contains Chinese characters

# # Solve negative sign display issue
# mpl.rcParams['axes.unicode_minus'] = False

# Specify a font that contains Chinese characters, e.g., SimHei
plt.rcParams['font.sans-serif'] = 'SimHei'
# To properly display negative signs
plt.rcParams['axes.unicode_minus'] = False

# Define color list
colorslist = ['#63be7b', '#fbfbfe', '#f8696b']
# Create color map
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
new_cmap = LinearSegmentedColormap.from_list('new_cmap', colorslist, N=800)

# Import FactorCapacity module
from factor_capacity import FactorCapacity

class SingleFactor_SinglePool_BackTest:
    def __init__(
            self,
            factor_data,
            price_data,
            benchmark_data,
            pool_data,
            factor_name,
            pool_name,
            start_date=None,
            end_date=None,
            is_daily_factor=True,
            group_data=None,
            direction=1,
    ):
        """
       Constructor for initializing a FactorCapacity class instance.

        Parameters
        
        factor_data : pd.DataFrame - MultiIndex (date, asset)
        Factor data.
        price_data : pd.DataFrame - MultiIndex (date, asset)
        Price data.
        benchmark_data : pd.DataFrame - Index (date)
        Benchmark data.
        pool_data : pd.DataFrame - MultiIndex (date, asset)
        Pool data.
        factor_name : str
        Factor name.
        pool_name : str
        Pool name.
        start_date : str or None, optional
        Start date, defaults to None.
        end_date : str or None, optional
        End date, defaults to None.
        is_daily_factor : bool, optional
        Whether it's a daily factor, defaults to True.
        group_data : pd.DataFrame, optional
        Grouping data, defaults to None.
        direction : int, optional
        Factor direction, defaults to 1.
        """

        self.factor_data = factor_data
        self.factor_data.columns = ['factor']
        self.factor_name = factor_name
        self.pool_name = pool_name
        self.price_data = price_data
        self.benchmark_data = benchmark_data
        self.pool_data = pool_data
        self.group_data = group_data
        self.is_daily_factor = is_daily_factor
        self.direction = direction
        self.start_date = pd.Timestamp(start_date) if start_date is not None else None
        self.end_date = pd.Timestamp(end_date) if end_date is not None else None

        self.factor_freq_clean_data = None
        self.daily_freq_clean_data = None
        self.grouped_factor_freq_clean_data = None

        self.factor_capacity = FactorCapacity()


    def generate_clean_data(self, quantiles: int = 5):
        """
        Assigns the following attributes to the Backtest object:
        factor_freq_clean_data - data cleaned at factor frequency
        daily_freq_clean_data - data cleaned at daily frequency
        grouped_factor_freq_clean_data - factor frequency cleaned data with industry classification

        Parameters
        ----------quantiles : int
            Divides all stocks into n groups based on factor values

        Returns

        None
        """
        factor_list = list(self.factor_data.keys())
        for factor in factor_list:
            self.get_factor_freq_clean_data(quantiles=quantiles)
            self.get_daily_freq_clean_data()
            if self.group_data is not None:
                self.get_grouped_factor_freq_clean_data()

    def get_factor_freq_clean_data(self, quantiles: int = 5):
        """
            1. Select data within the specified start and end time range.  
            - The final start time is the later of the dataset's start time and the specified start time.  
            - The final end time is the earlier of the dataset's end time and the specified end time.  

            2. Filter stocks that are in the specified stock pool.  

            3. Align the frequency of stock pool data and price data with the factor data.  

            4. Calculate the next-period stock returns and factor groupings.  

            Parameters
            ----------
            quantiles : int  
                Number of groups to divide all stocks based on factor values.  

            Returns
            -------
            factor_freq_clean_data : pd.DataFrame - MultiIndex  
                - Index: date, asset  
                - Columns: forward_return, factor, factor_quantile  
            """

        #  Filter All Data by Start and End Dates
        start_date_list = []
        end_date_list = []
        for data in [self.factor_data, self.price_data, self.pool_data, self.benchmark_data]:
            start_date_list.append(data.index.get_level_values('date').min())
            end_date_list.append(data.index.get_level_values('date').max())
        if self.start_date:
            start_date_list.append(self.start_date)
        if self.end_date:
            end_date_list.append(self.end_date)
        self.start_date = max(start_date_list)
        self.end_date = min(end_date_list)

        self.factor_data = self.factor_data[
            (self.factor_data.index.get_level_values('date') >= self.start_date) & (
                        self.factor_data.index.get_level_values('date') <= self.end_date)
            ]
        self.price_data = self.price_data[
            (self.price_data.index.get_level_values('date') >= self.start_date) & (
                        self.price_data.index.get_level_values('date') <= self.end_date)
            ]
        self.pool_data = self.pool_data[
            (self.pool_data.index.get_level_values('date') >= self.start_date) & (
                        self.pool_data.index.get_level_values('date') <= self.end_date)
            ]
        self.benchmark_data = self.benchmark_data[
            (self.benchmark_data.index.get_level_values('date') >= self.start_date) & (
                        self.benchmark_data.index.get_level_values('date') <= self.end_date)
            ]
        # Align the Frequency of Stock Pool Data with Factor Data Frequency
        self.pool_data = self.pool_data[
            self.pool_data.index.get_level_values('date').isin(self.factor_data.index.get_level_values('date'))]
        # Filter Stocks in the Stock Pool
        factor_array = self.factor_data.copy()
        factor_array = factor_array[factor_array.index.isin(self.pool_data.index)]
        # Align the Frequency of Price Data with Factor Data Frequency
        price_array = self.price_data.copy().reset_index().pivot(index='date', columns='asset', values='price')
        price_array = price_array[price_array.index.isin(factor_array.index.get_level_values('date').unique())]
        # Compute Next-Period Returns
        forward_returns = price_array.pct_change(1).shift(-1)
        forward_returns = forward_returns.stack().to_frame().rename({0: 'forward_return'}, axis=1)
        # Merge Factor Data with Future Returns
        factor_freq_clean_data = forward_returns.copy()
        factor_freq_clean_data['factor'] = factor_array
        factor_freq_clean_data = factor_freq_clean_data.dropna()

        # Generate Factor Groupings
        def quantile_calc(x, _quantiles):
            return pd.qcut(x, _quantiles, labels=False, duplicates='drop') + 1

        factor_quantile = factor_freq_clean_data.groupby('date')['factor'].transform(quantile_calc, quantiles)
        factor_quantile.name = 'factor_quantile'
        factor_freq_clean_data['factor_quantile'] = factor_quantile.dropna()
        factor_freq_clean_data = factor_freq_clean_data.dropna()

        self.factor_freq_clean_data = factor_freq_clean_data

    def get_daily_freq_clean_data(self):
        """
        For factors that are not at a daily frequency, generate a daily-frequency `clean_data`  
to facilitate the calculation of daily net value and return sequences.  

        Returns
        -------
        daily_freq_clean_data : pd.DataFrame - MultiIndex
            index : date, asset
            columns : forward_return, factor, factor_quantile
        """
        if self.is_daily_factor:
            self.daily_freq_clean_data = self.factor_freq_clean_data.copy()
        else:
            quantile_data = self.factor_freq_clean_data.copy()
            quantile_data = quantile_data.reset_index().pivot(index='date', columns='asset',
                                                              values='factor_quantile').fillna(0).stack()

            factor_data = self.factor_freq_clean_data.copy()
            factor_data = factor_data.reset_index().pivot(index='date', columns='asset', values='factor').stack()

            # Obtain Next-Day Returns
            price_array = self.price_data.copy().reset_index().pivot(index='date', columns='asset', values='price')
            # Compute Next-Period Return
            forward_returns = price_array.pct_change(1).shift(-1)
            forward_returns = forward_returns.stack().to_frame().rename({0: 'forward_return'}, axis=1)

            daily_freq_clean_data = forward_returns.copy()
            daily_freq_clean_data['factor'] = factor_data
            daily_freq_clean_data['factor_quantile'] = quantile_data
            daily_freq_clean_data = daily_freq_clean_data.sort_index(level=['asset', 'date']).groupby(['asset']).ffill()
            daily_freq_clean_data = daily_freq_clean_data.dropna(subset='factor_quantile')
            daily_freq_clean_data['factor_quantile'] = daily_freq_clean_data['factor_quantile'].astype('int')
            daily_freq_clean_data = daily_freq_clean_data[daily_freq_clean_data['factor_quantile'] != 0]

            self.daily_freq_clean_data = daily_freq_clean_data

    def get_grouped_factor_freq_clean_data(self):
        """
        Generate clean_data with Industry Information for Each Stock in Each Period

        Returns
        -------
        grouped_factor_freq_clean_data : pd.DataFrame - MultiIndex
            index : date, asset
            columns : forward_return, factor, factor_quantile, group
        """
        factor_data = self.factor_freq_clean_data.copy().reset_index().sort_values(['date', 'asset'])
        group_data = self.group_data.copy().reset_index().sort_values(['date', 'asset'])

        grouped_factor_freq_clean_data = pd.merge_asof(factor_data, group_data, on='date', by='asset').set_index(
            ['date', 'asset']).sort_index(level=['asset', 'date'])
        grouped_factor_freq_clean_data['group'] = grouped_factor_freq_clean_data.groupby('asset')[
            'group'].ffill().bfill()

        self.grouped_factor_freq_clean_data = grouped_factor_freq_clean_data

    # Factor Coverage Analysis
    def get_factor_coverage(self, group=False):
        """
        Factor Coverage Analysis  

        Retrieve the factor coverage sequence at the given factor frequency,  
        defined as the proportion of stocks with factor values in the stock pool for each period.  

        Parameters
        ----------
        group : bool, default False  
            Whether to calculate factor coverage separately for each industry.  

        Returns
        -------
        result : pd.DataFrame  
            - Index: date  
            - Columns: Factor coverage (if grouped by industry, each column represents the factor coverage for a specific industry).  
        """

        # Copy stock pool data and factor data
        pool_array = self.pool_data.copy()
        factor_array = self.factor_data.copy()

        # If grouping by industry, process the factor data accordingly
        if group:
            factor_array = factor_array.reset_index().sort_values(['date', 'asset'])
            group_array = self.group_data.copy().reset_index().sort_values(['date', 'asset'])
            factor_array = pd.merge_asof(factor_array, group_array, on='date', by='asset') \
                .set_index(['date', 'asset']).sort_index(level=['asset', 'date'])
            # Forward-fill and backward-fill missing industry group values
            factor_array['group'] = factor_array.groupby('asset')['group'].ffill().bfill()

        # Merge stock pool data with factor data
        df = pd.merge(pool_array, factor_array, on=['date', 'asset'], how='left')

        # Add a count column to track stock occurrences
        df['count'] = 1

        # Determine the grouping key (by industry or overall)
        if group:
            grouper = ['date', 'group']
        else:
            grouper = ['date']

        df['cumulative_count'] = df.groupby(grouper)['count'].cumsum()
        result = pd.DataFrame()
        result['factor_coverage_stock_count'] = df.groupby(grouper)['factor'].count()

        # Count total stocks in the pool for each period
        result['total_stock_pool_count'] = df.groupby(grouper)['cumulative_count'].max()

        # Calculate factor coverage ratio
        result['factor_coverage_ratio'] = result['factor_coverage_stock_count'] / result['total_stock_pool_count']

        # If grouping by industry, reshape the DataFrame
        if group:
            result = result[['factor_coverage_ratio']].reset_index().pivot(index='date', columns='group', values='factor_coverage_ratio')
        else:
            result = result[['factor_coverage_ratio']]

        # Store the result in the factor capacity attribute
        self.factor_capacity.factor_coverage_array = result.copy()

        return result


    

    def plot_factor_coverage(self):
        """
        Plot Factor Coverage  

        This function generates a time series plot of factor coverage, showing the proportion of stocks  
        with factor values in the stock pool for each period.

        Returns
        -------
        None
        """
        coverage = self.get_factor_coverage()
        fig, ax1 = plt.subplots(figsize=(16, 8))
        ax1.plot(coverage.index, coverage['Factor Coverage'], color='red', label='Factor Coverage')
        ax1.set_xticks(
            coverage.loc[coverage.groupby(coverage.index.year)['Factor Coverage'].cumcount() == 1].index,
            coverage.loc[coverage.groupby(coverage.index.year)['Factor Coverage'].cumcount() == 1].index.strftime("%Y")
        )
        ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax1.set_ylim(0, 1.1)

    def analyse_factor_descriptive_statistics(self, by='quantile'):
        """
        Analyze Factor Descriptive Statistics  

        Perform descriptive statistical analysis on the factor, grouped by quantile or year.

        Parameters
        ----------
        by : str, default 'quantile'  
            The grouping method for analysis. Options:  
            - 'quantile': Group by factor quantiles  
            - 'year': Group by year  

        Returns
        -------
        result : pd.DataFrame  
            - Index: Grouping category  
            - Columns: Statistical indicators  
        """
        if by == 'quantile':
            data1 = self.factor_freq_clean_data.copy()[['factor_quantile', 'factor']]
            data2 = data1.copy()
            data2['factor_quantile'] = 'Overall'
            data = pd.concat([data1, data2])
            grouper = 'factor_quantile'
        elif by == 'year':
            data1 = self.factor_freq_clean_data.copy()[['factor']]
            data2 = data1.copy()
            data1['year'] = data1.index.get_level_values('date').year
            data2['year'] = 'Overall'
            data = pd.concat([data1, data2])
            grouper = 'year'
        result = pd.DataFrame()
        result['Sample Size'] = data.groupby(grouper)['factor'].count()
        result['Mean'] = data.groupby(grouper)['factor'].mean()
        result['Standard Deviation'] = data.groupby(grouper)['factor'].std()
        result['Skewness'] = data.groupby(grouper)['factor'].skew()
        result['Kurtosis'] = data.groupby(grouper)['factor'].agg(lambda x: x.kurt())
        result['Minimum'] = data.groupby(grouper)['factor'].min()
        result['p10'] = data.groupby(grouper)['factor'].quantile(0.1)
        result['p25'] = data.groupby(grouper)['factor'].quantile(0.25)
        result['p50'] = data.groupby(grouper)['factor'].median()
        result['p75'] = data.groupby(grouper)['factor'].quantile(0.75)
        result['p90'] = data.groupby(grouper)['factor'].quantile(0.9)
        result['Maximum'] = data.groupby(grouper)['factor'].max()
        result['Median Absolute Deviation'] = data.groupby(grouper)['factor'].agg(lambda x: (x - x.median()).abs().median())
        return result


    def plot_factor_distribution(self):
        """
        Plot Factor Distribution Histogram and Density Plot

        Returns
        -------
        None
        """
        data = self.factor_freq_clean_data.copy()['factor']
        fig, ax1 = plt.subplots(figsize=(16, 8))
        ax1 = sns.histplot(data=data, kde=True)
        ax1.set_xlim(left=data.quantile(0.05), right=data.quantile(0.95))

    def get_ic(self):
        """
        Compute Factor IC Sequence

        Returns
        -------
        ic : pd.Series
            index : date
            value : Current IC value
        """
        ic = self.factor_freq_clean_data.copy()
        ic = ic[['forward_return', 'factor']].groupby(ic.index.get_level_values('date')).corr(method='spearman')
        ic = ic[ic.index.get_level_values(-1) == 'factor'].droplevel(-1)['forward_return'].rename('IC')
        self.factor_capacity.ic_array = ic
        return ic

    def analyse_ic(self):
        """
        Analyze Factor IC

        Returns
        -------
        ic_summary_table : pd.DataFrame
            index : Not meaningful for now
            columns : Analysis indicators
        """
        ic = self.get_ic()

        ic_summary_table = pd.DataFrame([{
            "IC Mean": ic.mean(),
            "IC Standard Deviation": ic.std(),
            "ICIR": ic.mean() / ic.std(),
            't-statistic': stats.ttest_1samp(ic, 0, nan_policy='omit')[0],
            'p-value': stats.ttest_1samp(ic, 0, nan_policy='omit')[1],
            'IC Skewness': ic.skew(),
            'IC Kurtosis': ic.kurt(),
            'IC Win Rate': len(ic[ic > 0]) / len(ic) if self.direction == 1 else len(ic[ic < 0]) / len(ic)
        }])
        ic_summary_table.index = [self.factor_name]

        self.factor_capacity.ic_summary = ic_summary_table

        return ic_summary_table


    def get_quantile_ic(self):
        """
        Compute the rank correlation coefficient between factor quantile and next-period return.

        Returns
        -------
        ic : pd.Series
            index : date
            value : Current quantile_IC value
        """
        ic = self.get_quantile_return_data().reset_index(level=0)
        ic = ic.groupby(ic.index.get_level_values('date')).corr(method='spearman')
        ic = ic[ic.index.get_level_values(-1) == 'factor_quantile'].droplevel(-1)['forward_return'].rename('quantile_IC')
        self.factor_capacity.quantile_ic_array = ic
        return ic

    def analyse_quantile_ic(self):
        """
        Analyze factor quantile_IC (rank correlation coefficient between factor quantile and next-period return).
        This is an original metric used to analyze the monotonicity of grouped returns.

        Returns
        -------
        ic_summary_table : pd.DataFrame
            index : Not meaningful for now
            columns : Analysis indicators
        """
        ic = self.get_quantile_ic()

        ic_summary_table = pd.DataFrame([{
            "q_IC Mean": ic.mean(),
            "q_IC Standard Deviation": ic.std(),
            "ICIR": ic.mean() / ic.std(),
            't-statistic': stats.ttest_1samp(ic, 0, nan_policy='omit')[0],
            'p-value': stats.ttest_1samp(ic, 0, nan_policy='omit')[1],
            'q_IC Skewness': ic.skew(),
            'q_IC Kurtosis': ic.kurt(),
            'q_IC Win Rate': len(ic[ic > 0]) / len(ic) if self.direction == 1 else len(ic[ic < 0]) / len(ic)
        }])
        ic_summary_table.index = [self.factor_name]
        self.factor_capacity.quantile_ic_summary = ic_summary_table
        return ic_summary_table

    def get_grouped_ic(self):
        """
        Compute factor IC sequence by industry.

        Returns
        -------
        ic : pd.DataFrame
            index : date
            columns : industries
            value : Current IC value for each industry
        """
        ic = self.grouped_factor_freq_clean_data.copy().set_index('group', append=True)
        ic = ic[['forward_return', 'factor']].groupby(
            [ic.index.get_level_values('date'), ic.index.get_level_values('group')]).corr(method='spearman')
        ic = ic[ic.index.get_level_values(-1) == 'factor'].droplevel(-1)['forward_return'].rename('IC').unstack(level=-1)
        self.factor_capacity.grouped_ic_array = ic
        return ic

    def analyse_grouped_ic(self):
        """
        Analyze Factor IC for Each Industry

        Returns
        -------
        ic_summary_table : pd.DataFrame
            - Index: Industries
            - Columns: Analysis metrics
        """
        ic = self.get_grouped_ic()

        ic_summary_table = pd.DataFrame({
            "IC Mean": ic.mean(),
            "IC Standard Deviation": ic.std(),
            "ICIR": ic.mean() / ic.std(),
            't-statistic': stats.ttest_1samp(ic, 0, nan_policy='omit')[0],
            'p-value': stats.ttest_1samp(ic, 0, nan_policy='omit')[1],
            'IC Skewness': ic.skew(),
            'IC Kurtosis': ic.kurt(),
            'IC Win Rate': [len(ic[i][ic[i] > 0]) / len(ic[i]) if self.direction == 1 else len(ic[i][ic[i] < 0]) / len(ic[i])
                            for i in ic.columns]
        })
        return ic_summary_table

    def plot_grouped_ic(self):
        """
        Plot Industry-Wise IC Bar Chart

        Returns
        -------
        None
        """
        grouped_ic = self.analyse_grouped_ic().sort_values('IC Mean', ascending=False)
        fig, ax1 = plt.subplots(figsize=(16, 8))
        ax1 = sns.barplot(data=grouped_ic, x=grouped_ic['IC Mean'], y=grouped_ic.index, color='#9e413e')
        ax1.set_xlim(-round(grouped_ic['IC Mean'].abs().max() * 1.05, 1),
                    round(grouped_ic['IC Mean'].abs().max() * 1.05, 1))
        ax1.bar_label(ax1.containers[0], fmt='%.3g')

    def analyse_ic_decay(self, max_lag=10):
        """
        Analyze IC Decay Over Time

        Parameters
        ----------
        max_lag : int, default 10
            Maximum period for IC decay calculation

        Returns
        -------
        result : pd.DataFrame
            - Index: Decay periods
            - Columns: IC and ICIR for each period
        """
        ic = self.factor_freq_clean_data.copy().sort_index(level=['asset', 'date'])
        for i in range(1, max_lag + 1):
            ic[f'forward_return_F{i}'] = ic.groupby(ic.index.get_level_values('asset'))['forward_return'].shift(-i)
        ic = ic.groupby(ic.index.get_level_values('date')).corr(method='spearman')
        ic = ic[ic.index.get_level_values(-1) == 'factor'].droplevel(-1)[
            [i for i in list(ic.columns) if 'forward' in i]]
        ic.columns = [str(i) for i in range(0, max_lag + 1)]

        result = pd.DataFrame()
        result['IC'] = ic.mean()
        result['ICIR'] = ic.mean() / ic.std()

        return result


    def plot_ic_dacay(self, max_lag=10):
        """
        Plot IC Decay Bar Chart

        Parameters
       
        max_lag : int, default 10
            Maximum period for IC decay calculation

        Returns

        None
        """

        decay = self.analyse_ic_decay(max_lag=max_lag)
        fig, ax1 = plt.subplots(figsize=(8, 8))
        ax2 = ax1.twinx()
        x = np.arange(len(decay.index))
        width = 0.35
        ax1.bar(x - width / 2, decay['IC'], width, label='IC')
        ax1.set_ylim(top=max(decay['IC'].abs()) * 1.05, bottom=-max(decay['IC'].abs()) * 1.05)
        ax1.axhline(0, color='black', linewidth=1)
        ax1.set_xticks(np.arange(len(decay.index)))
        ax2.bar(x + width / 2, decay['ICIR'], width, label='ICIR', color='red')
        ax2.set_ylim(top=max(decay['ICIR'].abs()) * 1.05, bottom=-max(decay['ICIR'].abs()) * 1.05)
        ax2.axhline(0, color='black', linewidth=1)
        ax2.set_xticks(np.arange(len(decay.index)))
        fig.legend(loc='lower center', bbox_transform=ax1.transAxes, ncol=2, bbox_to_anchor=(0, 0, 1, 1), frameon=False)

    def plot_ic(self, bar_figure=False):
        '''
        If the factor frequency is weekly or more granular, do not plot the bar chart
        '''
        ic_array = self.get_ic().to_frame()

        if bar_figure:
            fig, ax1 = plt.subplots(figsize=(8, 8))
            ax2 = ax1.twinx()
            ax1.bar(ic_array.index, ic_array['IC'], color='#2F2F2F', width=20, label='IC')
            ax2.plot(ic_array.index, ic_array['IC'].cumsum(), color='red', label='Accumulated_IC')
            ax1.set_xticks(
                ic_array.loc[ic_array.groupby(ic_array.index.year)['IC'].cumcount() == 1].index,
                ic_array.loc[ic_array.groupby(ic_array.index.year)['IC'].cumcount() == 1].index.strftime("%Y")
            )
            ax1.axhline(0, color='black', linewidth=1)
            fig.legend(loc='lower center', bbox_transform=ax1.transAxes, ncol=2, bbox_to_anchor=(0, 0, 1, 1),
                       frameon=False)
        else:
            fig, ax1 = plt.subplots(figsize=(8, 8))
            ax1.plot(ic_array.index, ic_array['IC'].cumsum(), color='red', label='Accumulated_IC')
            ax1.set_xticks(
                ic_array.loc[ic_array.groupby(ic_array.index.year)['IC'].cumcount() == 1].index,
                ic_array.loc[ic_array.groupby(ic_array.index.year)['IC'].cumcount() == 1].index.strftime("%Y")
            )
            fig.legend(loc='lower center', bbox_transform=ax1.transAxes, ncol=2, bbox_to_anchor=(0, 0, 1, 1),
                       frameon=False)

    def plot_quantile_ic(self, bar_figure=False):
        '''
        If the factor frequency is weekly or more granular, do not plot the bar chart.
        '''
        ic_array = self.get_quantile_ic().to_frame()

        if bar_figure:
            fig, ax1 = plt.subplots(figsize=(8, 8))
            ax2 = ax1.twinx()
            ax1.bar(ic_array.index, ic_array['quantile_IC'], color='#2F2F2F', width=20, label='quantile_IC')
            ax2.plot(ic_array.index, ic_array['quantile_IC'].cumsum(), color='red', label='Accumulated_IC(Right)')
            ax1.set_xticks(
                ic_array.loc[ic_array.groupby(ic_array.index.year)['quantile_IC'].cumcount() == 1].index,
                ic_array.loc[ic_array.groupby(ic_array.index.year)['quantile_IC'].cumcount() == 1].index.strftime("%Y")
            )
            ax1.axhline(0, color='black', linewidth=1)
            fig.legend(loc='lower center', bbox_transform=ax1.transAxes, ncol=2, bbox_to_anchor=(0, 0, 1, 1),
                       frameon=False)
        else:
            fig, ax1 = plt.subplots(figsize=(8, 8))
            ax1.plot(ic_array.index, ic_array['quantile_IC'].cumsum(), color='red', label='Accumulated_IC(Right)')
            ax1.set_xticks(
                ic_array.loc[ic_array.groupby(ic_array.index.year)['quantile_IC'].cumcount() == 1].index,
                ic_array.loc[ic_array.groupby(ic_array.index.year)['quantile_IC'].cumcount() == 1].index.strftime("%Y")
            )
            fig.legend(loc='lower center', bbox_transform=ax1.transAxes, ncol=2, bbox_to_anchor=(0, 0, 1, 1),
                       frameon=False)

    # Factor Autocorrelation and Turnover Rate Analysis
    def get_factor_autocorrelation(self, max_lag=5):
        """
        Generate the factor autocorrelation series

        Parameters
        ----------
        max_lag : int, default 10
            Maximum lookback period for calculating autocorrelation

        Returns
        -------
        result : pd.DataFrame
            index : date
            columns : Different lookback periods
        """
        ac_data = self.factor_freq_clean_data.copy()
        ac_data = ac_data[['factor']].reset_index().pivot(index='date', columns='asset', values='factor')
        ac_array = pd.DataFrame()
        for lag in range(1, max_lag + 1):
            for i in range(lag, len(ac_data)):
                ac_array.loc[ac_data.index[i], f'Lag {lag} Autocorrelation'] = ac_data.iloc[i].corr(
                    ac_data.iloc[i - lag], method='spearman'
                )
        self.factor_capacity.autocorrelation_array = ac_array
        return ac_array


    def analyse_factor_autocorrelation(self, max_lag=5):
        """
        Analyze Factor Autocorrelation

        Parameters
        ----------
        max_lag : int, default 5
            Maximum lag period for autocorrelation calculation

        Returns
        -------
        result : pd.DataFrame
            - Index: Different lag orders
            - Columns: Time-series mean and standard deviation of factor autocorrelation coefficients
        """
        ac_array = self.get_factor_autocorrelation(max_lag=max_lag)
        result = pd.DataFrame()
        result['Mean'] = ac_array.mean()
        result['Standard Deviation'] = ac_array.std()
        return result


    def get_factor_turnover(self, used_factor_freq=True):
        """
        Generate Factor Turnover Series

        Parameters
        ----------
        used_factor_freq : Boolean, default True
            Whether to calculate turnover frequency based on the factor's frequency 
            (Default is True when calling this method directly)

        Returns
        -------
        turnover : pd.Series - MultiIndex
            - Index: factor_quantile, date
            - Value: Turnover rate for the period
        """
        if used_factor_freq:
            turnover = self.factor_freq_clean_data.copy()
        else:
            turnover = self.daily_freq_clean_data.copy()
        
        turnover = turnover.reset_index(level=-1).set_index('factor_quantile', append=True)
        turnover = turnover.groupby(level=['date', 'factor_quantile']).agg(lambda x: set(x)).sort_index(level=[1, 0])
        
        turnover['last_asset'] = turnover.groupby(turnover.index.get_level_values('factor_quantile'))['asset'].shift(1)
        turnover['new_names'] = (turnover['asset'] - turnover['last_asset']).dropna()
        
        turnover['turnover'] = turnover['new_names'].map(lambda x: len(x) if x is not np.nan else 1) / \
                            turnover['last_asset'].map(lambda x: len(x) if x is not np.nan else 1)
        
        turnover = turnover.swaplevel('date', 'factor_quantile')['turnover']
        self.factor_capacity.turnover_array = turnover
        
        return turnover


    def analyse_factor_turnover(self, used_factor_freq=True):
        """
        Analyze Turnover Rate of Factor Quantiles

        Parameters
        ----------
        used_factor_freq : Boolean, default True
            Whether to calculate turnover frequency based on the factor's frequency 
            (Default is True when calling this method directly)

        Returns
        -------
        result : pd.DataFrame
            - Index: factor_quantile
            - Columns: Mean and Standard Deviation of Turnover Rate
        """
        turnover = self.get_factor_turnover(used_factor_freq=used_factor_freq).to_frame()
        turnover['count'] = turnover.groupby('factor_quantile').cumcount()
        turnover = turnover.loc[turnover['count'] != 0, 'turnover']
        
        result = pd.DataFrame()
        result['Mean'] = turnover.groupby('factor_quantile').mean()
        result['Standard Deviation'] = turnover.groupby('factor_quantile').std()
        
        return result

    def analyse_factor_group_distribution(self, long=True):
        """
        Analyze Industry Distribution of Long and Short Factor Portfolios

        Parameters
        ----------
        long : Boolean, default True
            By default, analyzes long portfolio; set to False for short portfolio.

        Returns
        -------
        pd.DataFrame
            - Index: Industry group
            - Columns: Mean and Standard Deviation of the proportion in the long/short portfolio.
        """
        group = self.grouped_factor_freq_clean_data.copy()
        group = group.reset_index('asset', drop=True)[['factor_quantile', 'group']].set_index('factor_quantile', append=True)
        group = group.groupby(['date', 'factor_quantile']).value_counts(normalize=True)

        max_group = group.loc[group.index.get_level_values('factor_quantile') == group.index.get_level_values('factor_quantile').max()]
        max_grouper = max_group.groupby('group')
        max_group_distribution = pd.DataFrame({
            'Mean': max_grouper.mean(),
            'Standard Deviation': max_grouper.std()
        })
        max_group_distribution.sort_values('Mean', ascending=False, inplace=True)

        min_group = group.loc[group.index.get_level_values('factor_quantile') == group.index.get_level_values('factor_quantile').min()]
        min_grouper = min_group.groupby('group')
        min_group_distribution = pd.DataFrame({
            'Mean': min_grouper.mean(),
            'Standard Deviation': min_grouper.std()
        })
        min_group_distribution.sort_values('Mean', ascending=False, inplace=True)

        if self.direction == 1:
            return max_group_distribution if long else min_group_distribution
        else:
            return min_group_distribution if long else max_group_distribution

    def analyse_factor_group_distribution_topN_per_year(self, long=True, display_num=5):
        """
        Analyze the Top N Industries in Long and Short Portfolios Each Year

        Parameters
        ----------
        long : Boolean, default True
            By default, analyzes long portfolio; set to False for short portfolio.
        display_num : int, default 5
            Number of top industries to display based on proportion.

        Returns
        -------
        pd.DataFrame
            - Index: Rank
            - Columns: Year
            - Values: Industry Name
        """
        group = self.grouped_factor_freq_clean_data.copy()
        group = group.reset_index('asset', drop=True)[['factor_quantile', 'group']].set_index('factor_quantile', append=True)
        group = group.groupby([group.index.get_level_values('date').year, 'factor_quantile']).value_counts(normalize=True)

        max_group = group.loc[group.index.get_level_values('factor_quantile') == group.index.get_level_values('factor_quantile').max()].to_frame()
        max_group = max_group.rename(columns={0: 'proportion'})
        max_group['rank'] = max_group.groupby('date')['proportion'].rank(method='first', ascending=False).astype('int')
        max_group = max_group.reset_index().pivot(index='rank', columns='date', values='group')
        max_group = max_group.head(display_num)

        min_group = group.loc[group.index.get_level_values('factor_quantile') == group.index.get_level_values('factor_quantile').min()].to_frame()
        min_group = min_group.rename(columns={0: 'proportion'})
        min_group['rank'] = min_group.groupby('date')['proportion'].rank(method='first', ascending=False).astype('int')
        min_group = min_group.reset_index().pivot(index='rank', columns='date', values='group')
        min_group = min_group.head(display_num)

        if self.direction == 1:
            return max_group if long else min_group
        else:
            return min_group if long else max_group

    def plot_factor_group_distribution(self, long=True):
        """
        Plot the Industry Distribution in Long/Short Factor Portfolios
        """
        group_distribution = self.analyse_factor_group_distribution(long=long)
        fig, ax1 = plt.subplots(figsize=(8, 8))
        ax1.barh(group_distribution.index, group_distribution['Mean'])
        ax1.invert_yaxis()


    def get_quantile_return_data(self, commission=0, used_factor_freq=True):
        """
        Calculates the next period's return for each factor quantile group.

        Parameters
        ----------
        commission : float, default 0
            Commission fee, charged on both sides. For example, if the commission is 0.0001 (1 basis point), input 1/10000.
        used_factor_freq : bool, default True
            If True, calculates returns based on factor frequency. If False, calculates returns based on daily frequency.

        Returns
        -------
        return_array : pd.DataFrame
            index : factor_quantile, date
            columns : forward_return
        """

        if used_factor_freq:
            factor_data = self.factor_freq_clean_data.copy()
        else:
            factor_data = self.daily_freq_clean_data.copy()

        grouper = ['factor_quantile']
        grouper.append(factor_data.index.get_level_values('date'))

        return_array = factor_data.groupby(grouper).agg({'forward_return': 'mean'})
        return_array['turnover'] = self.get_factor_turnover(used_factor_freq=used_factor_freq)
        return_array['forward_return'] = return_array['forward_return'] - return_array['turnover'] * commission * 2  # Charged on both sides

        return_array = return_array[['forward_return']]

        return return_array


    def get_benchmark_return_array(self, used_factor_freq=True):
        """
        Generates the return series for the benchmark portfolio.

        Parameters
        ----------
        used_factor_freq : bool, default True
            If True, calculates returns based on factor frequency. If False, calculates returns based on daily frequency.

        Returns
        -------
        benchmark_return : pd.DataFrame
            index : factor_quantile, date
            columns : forward_return
        """

        benchmark_return = self.benchmark_data.copy()
        
        if used_factor_freq:
            benchmark_return = benchmark_return[
                benchmark_return.index.isin(self.factor_freq_clean_data.index.get_level_values('date'))
            ]
        else:
            benchmark_return = benchmark_return[
                benchmark_return.index.isin(self.daily_freq_clean_data.index.get_level_values('date'))
            ]
        
        benchmark_return = benchmark_return.pct_change(1)
        
        return benchmark_return


    def get_quantile_return_array(self, commission=0, excess_return=False, used_factor_freq=True):
        """
        Generates the return series for each factor quantile group.

        Parameters
        ----------
        commission : float, default 0
            Commission fee, charged on both sides. For example, if the commission is 0.0001 (1 basis point), input 1/10000.
        excess_return : bool, default False
            If True, calculates excess return relative to the benchmark. If False, calculates absolute return.
        used_factor_freq : bool, default True
            If True, calculates returns based on factor frequency. If False, calculates returns based on daily frequency.

        Returns
        -------
        return_data : pd.DataFrame
            index : date
            columns : each quantile, benchmark portfolio, long-short portfolio
            value : period return
        """

        return_data = self.get_quantile_return_data(commission=commission, used_factor_freq=used_factor_freq) \
            .reset_index(level=0).pivot(columns='factor_quantile', values='forward_return')
        
        return_data = return_data.shift(1)
        return_data['benchmark'] = self.get_benchmark_return_array(used_factor_freq=used_factor_freq)

        max_quantile = return_data.columns[:-1].max()
        min_quantile = return_data.columns[:-1].min()

        if self.direction == 1:
            return_data['long_short'] = return_data[max_quantile] - return_data[min_quantile]
        else:
            return_data['long_short'] = return_data[min_quantile] - return_data[max_quantile]

        return_data.iloc[0] = 0

        if excess_return:
            for i in return_data.columns:
                if i != 'benchmark' and i != 'long_short':
                    return_data[i] = return_data[i] - return_data['benchmark']

        return return_data

    def get_net_value_array(self, commission=0, excess_return=False, used_factor_freq=True):
        """
        Generates the net value series for each factor quantile group.

        Parameters
        ----------
        commission : float, default 0
            Commission fee, charged on both sides. For example, if the commission is 0.0001 (1 basis point), input 1/10000.
        excess_return : bool, default False
            If True, calculates net value relative to the benchmark. If False, calculates absolute net value.
        used_factor_freq : bool, default True
            If True, calculates net values based on factor frequency. If False, calculates net values based on daily frequency.

        Returns
        -------
        return_data : pd.DataFrame
            index : date
            columns : each quantile, benchmark portfolio, long-short portfolio
            value : period net value
        """

        return_data = self.get_quantile_return_array(commission=commission, excess_return=excess_return,
                                                    used_factor_freq=used_factor_freq)

        nav_array = (return_data + 1).cumprod()
        
        return nav_array


    def get_single_net_value_array(self, nv_type, commission=0, excess_return=False, used_factor_freq=True):
        nav_array = self.get_net_value_array(commission=commission, excess_return=excess_return,
                                             used_factor_freq=used_factor_freq)

        if nv_type == 'ls':
            return nav_array['long_short']
        elif nv_type == 'l':
            if self.direction == 1:
                return nav_array[nav_array.columns[-3]]
            else:
                return nav_array[nav_array.columns[0]]
        elif nv_type == 's':
            if self.direction == 1:
                return nav_array[nav_array.columns[0]]
            else:
                return nav_array[nav_array.columns[-3]]
        elif isinstance(nv_type, int):
            return nav_array[nv_type]

    def analyse_return_array(self, commission=0):
        """
        Evaluation of factor grouping, benchmark, and long-short returns.

        Parameters
        ----------
        commission : float, default 0
            Commission, charged on both sides. For a commission of 1/10000, input 1/10000.

        Returns
        -------
        result : pd.DataFrame
            index : Factor groups, benchmark portfolio, long-short portfolio
            columns : Return evaluation metrics
        """

        f_return_array = self.get_quantile_return_array(commission=commission)
        d_return_array = self.get_quantile_return_array(commission=commission, used_factor_freq=False)
        f_excess_return_array = self.get_quantile_return_array(commission=commission,
                                                               excess_return=True).drop(['benchmark'], axis=1)
        d_excess_return_array = self.get_quantile_return_array(commission=commission, excess_return=True,
                                                               used_factor_freq=False).drop(['benchmark'], axis=1)

        result = pd.DataFrame()

        # Absolute return analysis
        result['Annualized Return'] = qs.cagr(d_return_array, periods=240)
        result['Annualized Volatility'] = qs.volatility(d_return_array, periods=240)
        result['Sharpe Ratio'] = (result['Annualized Return'] - 0.015) / result['Annualized Volatility']
        result['Max Drawdown'] = qs.max_drawdown(d_return_array)
        result['Calmar Ratio'] = -result['Annualized Return'] / result['Max Drawdown']

        # Excess return analysis
        result['Excess Annualized Return'] = qs.cagr(d_excess_return_array, periods=240)
        result['Excess Annualized Volatility'] = qs.volatility(d_excess_return_array, periods=240)
        result['Information Ratio'] = result['Excess Annualized Return'] / result['Excess Annualized Volatility']
        result['Excess Max Drawdown'] = qs.max_drawdown(d_excess_return_array)
        result['Excess Calmar Ratio'] = -result['Excess Annualized Return'] / result['Excess Max Drawdown']
        result['Win Rate vs Benchmark'] = f_excess_return_array.agg(lambda x: len(x[x > 0]) / len(x))
        whole = f_return_array.copy().iloc[1:, :-2]
        result['Win Rate vs All'] = whole.agg(lambda x: len(x[x == whole.max(axis=1)]) / len(x))
        result['Profit-Loss Ratio'] = f_excess_return_array.agg(lambda x: x[x > 0].mean() / x[x < 0].abs().mean())

        return result

    def analyse_return_briefly(self, commission=0):
        """
        Brief return analysis summary.
        """
        detailed_table = self.analyse_return_array(commission=commission)
        single_pofo = detailed_table.drop(index=['benchmark', 'long_short'])
        if self.direction == 1:
            sub_table1 = single_pofo.loc[
                single_pofo.index[[-1]], ['Annualized Return', 'Sharpe Ratio', 'Excess Annualized Return',
                                          'Information Ratio', 'Excess Max Drawdown', 'Win Rate vs Benchmark',
                                          'Profit-Loss Ratio']]
        else:
            sub_table1 = single_pofo.loc[
                single_pofo.index[[0]], ['Annualized Return', 'Sharpe Ratio', 'Excess Annualized Return',
                                         'Information Ratio', 'Excess Max Drawdown', 'Win Rate vs Benchmark',
                                         'Profit-Loss Ratio']]
        sub_table1.index = [self.factor_name]
        sub_table2 = detailed_table.loc[['long_short'], ['Annualized Return', 'Sharpe Ratio', 'Max Drawdown']]
        sub_table2.columns = ['Long-Short Annualized Return', 'Long-Short Sharpe Ratio', 'Long-Short Max Drawdown']
        sub_table2.index = [self.factor_name]
        result_table = pd.concat([sub_table1, sub_table2], axis=1)
        self.factor_capacity.return_summary = result_table
        return result_table

    def plot_annual_return_heatmap(self, commission=0, excess_return=False):
        """
        Plot an annual return heatmap.

        Parameters
        ----------
        commission : float, default 0
            Commission, charged on both sides. For a commission of 1/10000, input 1/10000.

        Returns
        -------
        result : pd.DataFrame
            index : Factor groups, benchmark portfolio, long-short portfolio
            columns : Return evaluation metrics
        """
        return_array = self.get_quantile_return_array(commission=commission, excess_return=excess_return,
                                                      used_factor_freq=False)
        return_array = return_array.groupby(return_array.index.year).agg(lambda x: (1 + x).prod() - 1)
        fig, ax1 = plt.subplots(figsize=(16, 8))
        ax1 = sns.heatmap(
            data=return_array.iloc[:, :-2].rank(axis=1, pct=True),
            cmap=new_cmap,
            annot=return_array.iloc[:, :-2],
            fmt='.2%',
            annot_kws={'color': 'black'},
            cbar=False,
        )
        fig.subplots_adjust(left=0.25, bottom=0.5)
        ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)

    def plot_quantile_annualized_return(self, commission=0, excess_return=True):
        """
        Plot annualized (excess) returns for different quantiles as a bar chart.
        """
        return_array = self.get_quantile_return_array(commission=commission, excess_return=excess_return,
                                                      used_factor_freq=False)
        result = pd.DataFrame()
        result['Annualized Return'] = qs.cagr(return_array, periods=240)
        fig, ax1 = plt.subplots(figsize=(16, 8))
        for i in result.index.drop(['benchmark', 'long_short']):
            ax1.bar(str(i), result.at[i, 'Annualized Return'], label=str(result.index))
        ax1.axhline(0, color='black', linewidth=1)

    def plot_quantile_accumulated_net_value(self, commission=0, excess_return=False):
        """
        Plot the accumulated net value of different factor quantiles as a line chart.
        """
        nav_array = self.get_net_value_array(used_factor_freq=False, commission=commission,
                                             excess_return=excess_return)
        fig, ax1 = plt.subplots(figsize=(16, 8))
        for i in nav_array.columns.drop(['benchmark', 'long_short']):
            ax1.plot(nav_array.index, nav_array[i], label=str(i))
        fig.legend(loc=2, bbox_transform=ax1.transAxes, bbox_to_anchor=(0, 0, 1, 1))

    def plot_long_short_accumulated_net_value(self, commission=0, excess_return=True):
        """
        Plot the accumulated net value of long, short, long-short, and benchmark portfolios as a line chart.
        """
        nav_array = self.get_net_value_array(used_factor_freq=False, commission=commission,
                                             excess_return=excess_return)
        fig, ax1 = plt.subplots(figsize=(16, 8))

        if excess_return:
            if self.direction == 1:
                ax1.plot(nav_array.index, nav_array[nav_array.columns[-3]], label='long_excess')
                ax1.plot(nav_array.index, nav_array[nav_array.columns[0]], label='short_excess')
            else:
                ax1.plot(nav_array.index, nav_array[nav_array.columns[0]], label='long_excess')
                ax1.plot(nav_array.index, nav_array[nav_array.columns[-3]], label='short_excess')
        else:
            if self.direction == 1:
                ax1.plot(nav_array.index, nav_array[nav_array.columns[-3]], label='long_group')
                ax1.plot(nav_array.index, nav_array[nav_array.columns[0]], label='short_group')
            else:
                ax1.plot(nav_array.index, nav_array[nav_array.columns[0]], label='long_group')
                ax1.plot(nav_array.index, nav_array[nav_array.columns[-3]], label='short_group')
        ax1.plot(nav_array.index, nav_array['long_short'], label='long_short_group')
        if not excess_return:
            ax1.plot(nav_array.index, nav_array['benchmark'], label='benchmark_group')
        fig.legend(loc=2, bbox_transform=ax1.transAxes, bbox_to_anchor=(0, 0, 1, 1))
