#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import pandas as pd

from single_factor_backtest import *
from factor_capacity import *
from typing import Union, Optional, Dict


class MultiFactor_MultiPool_BackTest:
    def __init__(
            self,
            factor_data: pd.DataFrame,  # Multi-index, columns for each factor, column names are factor names
            price_data: pd.DataFrame,  # Multi-index, column for price
            benchmark_data: dict,  # Same as above
            pool_data: dict,  # Dictionary, key is pool name, value is dataframe, multi-index, same format as single pool
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            is_daily_factor: Union[bool, Dict[str, bool]] = True,  # Boolean or dictionary, key is factor name, value is boolean
            group_data: Optional[pd.DataFrame] = None,  # Same as above
            direction: Union[int, Dict[str, int]] = 1,  # int or dictionary, key is factor name, value is int
            quantiles: int = 5,
    ):
        """
        Initialization method.

        Parameters
        ----------
        factor_data : pd.DataFrame - MultiIndex (date, asset)
            Factor data, each factor is a column, column name is the factor name.
        price_data : pd.DataFrame - MultiIndex (date, asset)
            Price data, with only one price column.
        benchmark_data : pd.DataFrame - MultiIndex (date, asset)
            Benchmark data, with only one benchmark_price column.
        pool_data : Dict[str, pd.DataFrame - MultiIndex (date, asset)]
            Stock pool data, each stock pool is a DataFrame with no columns.
        start_date : str, optional
            Start date, format: '20130101', default is None.
        end_date : str, optional
            End date, format: '20221231', default is None.
        is_daily_factor : bool or Dict[str, bool], optional
            Whether the factor is a daily factor, default is True. If bool, all factors share this flag; if dictionary, each factor can be set separately.
        group_data : Dict[str, pd.DataFrame - MultiIndex (date, asset)], optional
            Group (industry) data, default is None; DataFrame's column is the industry.
        direction : int or Dict[str, int], optional
            Factor direction, default is 1. If int, all factors share this direction; if dictionary, each factor can be set separately.
        quantiles : int, optional
            Number of backtest groups, default is 5.

        Returns
        -------
        None

        """
        self.factor_data = factor_data
        self.price_data = price_data
        self.benchmark_data = benchmark_data
        self.pool_data = pool_data

        self.start_date = pd.Timestamp(start_date) if start_date is not None else None
        self.end_date = pd.Timestamp(end_date) if end_date is not None else None

        if isinstance(is_daily_factor, bool):
            self.is_daily_factor = {factor_name: is_daily_factor for factor_name in self.factor_data.columns}
        elif isinstance(is_daily_factor, dict):
            self.is_daily_factor = is_daily_factor

        self.group_data = group_data

        if isinstance(direction, int):
            self.direction = {factor_name: direction for factor_name in self.factor_data.columns}
        elif isinstance(direction, dict):
            self.direction = direction

        self.quantiles = quantiles

        self.factor_nums = self.factor_data.shape[1]
        self.pool_nums = len(self.pool_data)

        self.single_backtest_dataframe = None

    def get_factor_list(self):
        return list(self.factor_data.columns)

    def get_pool_list(self):
        return list(self.pool_data.keys())

    def generate_single_factor_pool_object(self):
        """
        Using a factors and b stock pools passed to the instance, generate a×b single factor instances
        and complete the data cleaning of all instances
        """
        self.single_backtest_dataframe = pd.DataFrame(
            np.full(shape=(self.factor_nums, self.pool_nums), fill_value=None),
            index=self.get_factor_list(),
            columns=self.get_pool_list(),
        )
        for factor_name in self.get_factor_list():
            for pool_name in self.get_pool_list():
                sub_object = SingleFactor_SinglePool_BackTest(
                    factor_data=self.factor_data[[factor_name]],
                    price_data=self.price_data,
                    benchmark_data=self.benchmark_data[pool_name],
                    pool_data=self.pool_data[pool_name],
                    factor_name=factor_name,
                    pool_name=pool_name,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    is_daily_factor=self.is_daily_factor[factor_name],
                    group_data=self.group_data,
                    direction=self.direction[factor_name]
                )
                sub_object.generate_clean_data(quantiles=self.quantiles)
                self.single_backtest_dataframe.at[factor_name, pool_name] = sub_object

    def get_backtest(self, factor_name: str, pool_name: str) -> SingleFactor_SinglePool_BackTest:
        """
        Get the backtest object for the specified factor and stock pool

        Parameters
        ----------
        factor_name : str
            Factor name
        pool_name : str
            Stock pool name

        Returns
        -------
        Backtest
            Backtest object
        """
        return self.single_backtest_dataframe.loc[factor_name, pool_name]

    # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ Integrated Single Factor Analysis Functions ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓

    # ==== Factor Coverage Analysis ====
    def plot_factor_coverage(self, factor_name: str, pool_name: str):
        """
        Plot factor coverage curve

        Parameters
        ----
        factor_name: str
            Factor name
        pool_name: str
            Stock pool name

        Returns
        ------
        ax: matplotlib.axes.Axes
            Coordinate axis object for plotting factor coverage curve
        """
        return self.get_backtest(factor_name=factor_name, pool_name=pool_name).plot_factor_coverage()

    # ==== Factor Descriptive Statistics ====
    def analyse_factor_descriptive_statistics(self, factor_name: str, pool_name: str, by: str = 'quantile'):
        """
        Analyze descriptive statistics of the factor

        Parameters
        ----
        factor_name: str
            Factor name
        pool_name: str
            Pool name
        by: str, optional
            Variable used for grouping (default is 'quantile')
            quantile: Statistics classified by factor grouping
            year: Statistics classified by year

        Returns
        ------
        analysis: pd.DataFrame
            Analysis results of factor descriptive statistics
        """
        return self.get_backtest(factor_name=factor_name, pool_name=pool_name).analyse_factor_descriptive_statistics(
            by=by)

    def plot_factor_distribution(self, factor_name: str, pool_name: str):
        """
        Plot the distribution of a factor based on the given factor name and pool name.

        Parameters
        ----------
        factor_name : str
            The name of the factor.
        pool_name : str
            The name of the data pool.

        Returns
        -------
        None
        """
        return self.get_backtest(factor_name=factor_name, pool_name=pool_name).plot_factor_distribution()

    # ========================================= Factor IC Analysis =========================================
    def analyse_ic(self, factor_name: str, pool_name: str, ic_type: str = 'ic'):
        """
        Output factor IC analysis result.

        Parameters
        ----------
        factor_name : str
            Factor name
        pool_name : str
            Stock pool name.
        ic_type : {'ic', 'quantile_ic', 'grouped_ic'}, default 'ic'
            Type of IC to analyze.
            * ic: Regular IC
            * quantile_ic: Quantile IC
            * grouped_ic: Industry-grouped IC

        Returns
        -------
        dict
            A dictionary containing the IC analysis result.
        """
        bt = self.get_backtest(factor_name=factor_name, pool_name=pool_name)
        if ic_type == 'ic':
            return bt.analyse_ic()
        elif ic_type == 'quantile_ic':
            return bt.analyse_quantile_ic()
        elif ic_type == 'grouped_ic':
            return bt.analyse_grouped_ic()
        else:
            raise ValueError("Invalid IC type. Possible values are 'ic', 'quantile_ic' and 'grouped_ic'.")

    def plot_ic(self, factor_name: str, pool_name: str, ic_type: str = 'ic', bar_figure: bool = False):
        """
        Plot the IC analysis result.

        Parameters
        ----------
        factor_name : str
            Factor name
        pool_name : str
            Stock pool name.
        ic_type : {'ic', 'quantile_ic', 'grouped_ic'}, default 'ic'
            Type of IC to analyze.
            * ic: Regular IC
            * quantile_ic: Quantile IC
            * grouped_ic: Industry-grouped IC
        bar_figure : bool, default False
            Whether to plot the IC analysis result as a bar chart. Default is False.

        Returns
        -------
        None
        """
        bt = self.get_backtest(factor_name=factor_name, pool_name=pool_name)
        if ic_type == 'ic':
            return bt.plot_ic(bar_figure=bar_figure)
        elif ic_type == 'quantile_ic':
            return bt.plot_quantile_ic(bar_figure=bar_figure)
        elif ic_type == 'grouped_ic':
            return bt.plot_grouped_ic()
        else:
            raise ValueError("Invalid IC type. Possible values are 'ic', 'quantile_ic' and 'grouped_ic'.")

    def analyse_ic_decay(self, factor_name: str, pool_name: str, max_lag: int = 10):
        """
        Analyze factor IC decay

        Parameters
        ----------
        factor_name : str
            Factor name.
        pool_name : str
            Stock pool name.
        max_lag : int
            Maximum number of periods for IC decay analysis

        Returns
        -------
        Analysis result of IC decay
        """
        return self.get_backtest(factor_name=factor_name, pool_name=pool_name).analyse_ic_decay(max_lag=max_lag)

    def plot_ic_dacay(self, factor_name: str, pool_name: str, max_lag: int = 10):
        """
        Plot factor IC decay

        Parameters
        ----------
        factor_name : str
            Factor name.
        pool_name : str
            Stock pool name.
        max_lag : int
            Maximum number of periods for IC decay analysis

        Returns
        -------
        Plot of IC decay
        """
        return self.get_backtest(factor_name=factor_name, pool_name=pool_name).plot_ic_dacay(max_lag=max_lag)

    # ========================================= Factor Autocorrelation and Turnover Analysis =========================================
    def analyse_factor_autocorrelation(self, factor_name: str, pool_name: str, max_lag: int = 5):
        """
        Analyze factor autocorrelation

        Parameters
        ----------
        factor_name : str
            Factor name.
        pool_name : str
            Stock pool name.
        max_lag : int
            Maximum lookback periods

        Returns
        -------
        Factor autocorrelation analysis
        """
        bt = self.get_backtest(factor_name=factor_name, pool_name=pool_name)
        return bt.analyse_factor_autocorrelation(max_lag=max_lag)

    def analyse_factor_turnover(self, factor_name: str, pool_name: str, used_factor_freq: bool = True):
        """
        Analyze factor turnover

        Parameters
        ----------
        factor_name : str
            Factor name.
        pool_name : str
            Stock pool name.
        used_factor_freq : bool, default
            If True, calculate the average turnover per rebalance; if False, calculate the average turnover per trading day

        Returns
        -------
        Factor turnover analysis
        """
        bt = self.get_backtest(factor_name=factor_name, pool_name=pool_name)
        return bt.analyse_factor_turnover(used_factor_freq=used_factor_freq)

    # ========================================= Analysis of Industry Distribution in Factor Long and Short Portfolios =========================================
    # (Must provide group_data parameter when creating backtest instance to use this)
    def _check_group_data(self):
        """
        Check if industry data is provided; cannot call industry analysis methods if not provided
        """
        if self.group_data is None:
            raise ValueError("Please provide group_data when creating backtest instance.")

    def analyse_factor_group_distribution(self, factor_name: str, pool_name: str, long: bool = True):
        """
        Analyze the proportion of different industries in the factor's long and short portfolios

        Parameters
        ----------
        factor_name : str
            Factor name.
        pool_name : str
            Stock pool name.
        long : bool, optional, default True
            If True, analyze long portfolio; if False, analyze short portfolio.

        Returns
        -------
        Analysis of industry distribution
        """
        self._check_group_data()
        bt = self.get_backtest(factor_name=factor_name, pool_name=pool_name)
        return bt.analyse_factor_group_distribution(long=long)

    def analyse_factor_group_distribution_topN_per_year(
            self,
            factor_name: str,
            pool_name: str,
            long: bool = True,
            display_num: int = 5
    ):
        """
        Analyze the top n industries with the highest proportion in factor's long and short portfolios

        Parameters
        ----------
        factor_name : str
            Factor name.
        pool_name : str
            Stock pool name.
        long : bool, optional, default True
            If True, analyze long portfolio; if False, analyze short portfolio.
        display_num : int, optional, default 5
            Display the top n industries with highest proportion

        Returns
        -------
        Analysis of top industries by proportion
        """
        self._check_group_data()
        bt = self.get_backtest(factor_name=factor_name, pool_name=pool_name)
        return bt.analyse_factor_group_distribution_topN_per_year(long=long, display_num=display_num)

    def plot_factor_group_distribution(self, factor_name: str, pool_name: str, long: bool = True):
        """
        Plot the industry distribution of factor's long and short portfolios

        Parameters
        ----------
        factor_name : str
            Factor name.
        pool_name : str
            Stock pool name.
        long : bool, optional, default True
            If True, analyze long portfolio; if False, analyze short portfolio.

        Returns
        -------
        Plot of industry distribution
        """
        bt = self.get_backtest(factor_name=factor_name, pool_name=pool_name)
        return bt.plot_factor_group_distribution(long=long)

    # ========================================= Factor Group Return Analysis =========================================

    def analyse_return_array(
            self,
            factor_name: str,
            pool_name: str,
            commission: float = 0
    ):
        """
        Output detailed analysis of factor group returns

        Parameters
        ----------
        factor_name : str
            Factor name.
        pool_name : str
            Stock pool name.
        commission : float, optional, default 0
            Trading commission (one-way), 1/10000 for one basis point

        Returns
        -------
        Detailed analysis of factor returns
        """
        bt = self.get_backtest(factor_name=factor_name, pool_name=pool_name)
        return bt.analyse_return_array(commission=commission)

    def analyse_return_briefly(
            self,
            factor_name: str,
            pool_name: str,
            commission: float = 0
    ):
        """
        Output brief analysis of factor group returns

        Parameters
        ----------
        factor_name : str
            Factor name.
        pool_name : str
            Stock pool name.
        commission : float, optional, default 0
            Trading commission (one-way), 1/10000 for one basis point

        Returns
        -------
        Brief analysis of factor returns
        """
        bt = self.get_backtest(factor_name=factor_name, pool_name=pool_name)
        return bt.analyse_return_briefly(commission=commission)

    def plot_annual_return_heatmap(
            self,
            factor_name: str,
            pool_name: str,
            commission: float = 0,
            excess_return: bool = False
    ):
        """
        Plot heatmap of factor returns for different groups across different years

        Parameters
        ----------
        factor_name : str
            Factor name.
        pool_name : str
            Stock pool name.
        commission : float, optional, default 0
            Trading commission (one-way), 1/10000 for one basis point
        excess_return : bool, optional, default False
            If True, use excess returns; if False, use absolute returns

        Returns
        -------
        Heatmap of annual returns
        """
        bt = self.get_backtest(factor_name=factor_name, pool_name=pool_name)
        return bt.plot_annual_return_heatmap(commission=commission, excess_return=excess_return)

    def plot_quantile_annualized_return(
            self,
            factor_name: str,
            pool_name: str,
            commission: float = 0,
            excess_return: bool = True
    ):
        """
        Plot bar chart of annualized returns for different groups

        Parameters
        ----------
        factor_name : str
            Factor name.
        pool_name : str
            Stock pool name.
        commission : float, optional, default 0
            Trading commission (one-way), 1/10000 for one basis point
        excess_return : bool, optional, default True
            If True, use excess returns; if False, use absolute returns

        Returns
        -------
        Bar chart of annualized returns
        """
        bt = self.get_backtest(factor_name=factor_name, pool_name=pool_name)
        return bt.plot_quantile_annualized_return(commission=commission, excess_return=excess_return)

    def plot_accumulated_net_value(
            self,
            factor_name: str,
            pool_name: str,
            plot_type: str = 'quantile',
            commission: float = 0,
            excess_return: bool = False
    ):
        """
        Plot net value curves.

        Parameters
        ----------
        factor_name : str
            Factor name.
        pool_name : str
            Stock pool name.
        commission : float, optional, default 0
            Trading commission (one-way), 1/10000 for one basis point
        excess_return : bool, optional, default False
            If True, use excess returns; if False, use absolute returns
        plot_type: str, {'quantile', 'long_short'}, default 'quantile'
            Set plot type.
            * quantile: Plot net value curves for each quantile group
            * long_short: Plot net value curves for long, short, long-short, and benchmark

        Returns
        -------
        plot
            Accumulated Net Value over time for different factor quantiles or long/short positions.
        """
        bt = self.get_backtest(factor_name=factor_name, pool_name=pool_name)
        if plot_type == 'quantile':
            return bt.plot_quantile_accumulated_net_value(commission=commission, excess_return=excess_return)
        elif plot_type == 'long_short':
            return bt.plot_long_short_accumulated_net_value(commission=commission, excess_return=excess_return)
        else:
            raise ValueError("Invalid type. Type can only be 'quantile' or 'long_short'.")

    # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ Integrated Single Factor Analysis Functions ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

    # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ Multi-Factor Comparison Analysis Functions ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
    def get_backtest_list(
            self,
            compare_level: str,
            target: str,
            filter_list: Optional[list] = None
    ):
        """
        Get list of backtests for comparison

        Parameters
        ----------
        compare_level : str, {'pool', 'factor'}
            Dimension for comparison.
            * pool: Compare "specific factor" across "multiple stock pools"
            * factor: Compare "multiple factors" on a "specific stock pool"
        target : str
            Specific factor name (compare_level='pool') or specific stock pool name (compare_level='factor')
        filter_list : None or list, optional, default=None
            If not provided, compare all stock pools (compare_level='pool') or compare all factors (compare_level='factor')
            If provided, only compare the items in the list.

        Returns
        -------
        filter_list : list
            List of stock pool names (compare_level='pool') or factor names (compare_level='factor')
        backtest_list : list
            List of backtest objects corresponding to filter_list
        """
        if compare_level == 'pool':
            if target not in self.get_factor_list():
                raise ValueError("Factor does not exist")
            if filter_list is None:
                filter_list = self.get_pool_list()
            return filter_list, list(self.single_backtest_dataframe.loc[target, filter_list])
        elif compare_level == 'factor':
            if target not in self.get_pool_list():
                raise ValueError("Stock pool does not exist")
            if filter_list is None:
                filter_list = self.get_factor_list()
            return filter_list, list(self.single_backtest_dataframe.loc[filter_list, target])

    def compare_plot_factor_coverage(
            self,
            compare_level: str,
            target: str,
            filter_list: Optional[list] = None
    ):
        """
        Compare factor coverage of a specific factor across different stock pools, or different factors in a specific stock pool

        Parameters
        ----------
        compare_level : str, {'pool', 'factor'}
            Dimension for comparison.
            * pool: Compare "specific factor" across "multiple stock pools"
            * factor: Compare "multiple factors" on a "specific stock pool"
        target : str
            Specific factor name (compare_level='pool') or specific stock pool name (compare_level='factor')
        filter_list : None or list, optional, default=None
            If not provided, compare all stock pools (compare_level='pool') or compare all factors (compare_level='factor')
            If provided, only compare the items in the list.

        Returns
        -------
        Comparison plot of factor coverage
        """
        fig, ax1 = plt.subplots(figsize=(16, 8))
        backtest_name_list, backtest_list = self.get_backtest_list(compare_level=compare_level, target=target,
                                                                   filter_list=filter_list)
        for bt_name, bt in zip(backtest_name_list, backtest_list):
            sub = bt.get_factor_coverage().rename(columns={'Factor Coverage': bt_name})
            ax1.plot(sub.index, sub[bt_name], label=bt_name)
        ax1.legend()
        ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax1.set_ylim(0, 1.1)
        plt.show()

    def compare_analyse_ic(
            self,
            compare_level: str,
            target: str,
            filter_list: Optional[list] = None,
            ic_type: str = 'ic'
    ):
        """
        Compare IC analysis results

        Parameters
        ----------
        compare_level : str, {'pool', 'factor'}
            Dimension for comparison.
            * pool: Compare "specific factor" across "multiple stock pools"
            * factor: Compare "multiple factors" on a "specific stock pool"
        target : str
            Specific factor name (compare_level='pool') or specific stock pool name (compare_level='factor')
        filter_list : None or list, optional, default=None
            If not provided, compare all stock pools (compare_level='pool') or compare all factors (compare_level='factor')
            If provided, only compare the items in the list.
        ic_type : str, {'ic', 'q_ic}, default 'ic'
            * ic: Perform IC analysis
            * q_ic: Perform quantile_IC analysis

        Returns
        -------
        Comparison of IC analysis results
        """
        backtest_name_list, backtest_list = self.get_backtest_list(compare_level=compare_level, target=target,
                                                                   filter_list=filter_list)
        s_list = []
        for bt in backtest_list:
            if ic_type == 'ic':
                s_list.append(bt.analyse_ic())
            elif ic_type == 'q_ic':
                s_list.append(bt.analyse_quantile_ic())
        result = pd.concat(s_list)
        result.index = backtest_name_list
        return result

    def compare_analyse_factor_turnover(
            self,
            compare_level: str,
            target: str,
            filter_list: Optional[list] = None,
    ):
        """
        Compare turnover rates of a specific factor across different stock pools, or different factors in a specific stock pool

        Parameters
        ----------
        compare_level : str, {'pool', 'factor'}
            Dimension for comparison.
            * pool: Compare "specific factor" across "multiple stock pools"
            * factor: Compare "multiple factors" on a "specific stock pool"
        target : str
            Specific factor name (compare_level='pool') or specific stock pool name (compare_level='factor')
        filter_list : None or list, optional, default=None
            If not provided, compare all stock pools (compare_level='pool') or compare all factors (compare_level='factor')
            If provided, only compare the items in the list.

        Returns
        -------
        Comparison of factor turnover rates
        """
        backtest_name_list, backtest_list = self.get_backtest_list(compare_level=compare_level, target=target,
                                                                   filter_list=filter_list)
        s_list = []
        for bt in backtest_list:
            s_list.append(bt.analyse_factor_turnover()[['Mean']])
        result = pd.concat(s_list, axis=1)
        result.columns = backtest_name_list
        return result

    def compare_analyse_return_briefly(
            self,
            compare_level: str,
            target: str,
            filter_list: Optional[list] = None,
            commission: float = 0
    ):
        """
        Compare return analysis of a specific factor across different stock pools, or different factors in a specific stock pool

        Parameters
        ----------
        compare_level : str, {'pool', 'factor'}
            Dimension for comparison.
            * pool: Compare "specific factor" across "multiple stock pools"
            * factor: Compare "multiple factors" on a "specific stock pool"
        target : str
            Specific factor name (compare_level='pool') or specific stock pool name (compare_level='factor')
        filter_list : None or list, optional, default=None
            If not provided, compare all stock pools (compare_level='pool') or compare all factors (compare_level='factor')
            If provided, only compare the items in the list.
        commission : float, optional, default 0
            Trading commission (one-way), 1/10000 for one basis point

        Returns
        -------
        Comparison of return analysis
        """
        backtest_name_list, backtest_list = self.get_backtest_list(compare_level=compare_level, target=target,
                                                                   filter_list=filter_list)
        s_list = []
        for bt in backtest_list:
            s_list.append(bt.analyse_return_briefly(commission=commission))
        result = pd.concat(s_list)
        result.index = backtest_name_list
        return result

    def compare_plot_accumulated_net_value(
            self,
            compare_level: str,
            target: str,
            filter_list: Optional[list] = None,
            commission: float = 0,
            nv_type: str = 'l',
            excess_return: bool = True
    ):
        """
        Plot net value curves of a specific factor across different stock pools, or different factors in a specific stock pool

        Parameters
        ----------
        compare_level : str, {'pool', 'factor'}
            Dimension for comparison.
            * pool: Compare "specific factor" across "multiple stock pools"
            * factor: Compare "multiple factors" on a "specific stock pool"
        target : str
            Specific factor name (compare_level='pool') or specific stock pool name (compare_level='factor')
        filter_list : None or list, optional, default=None
            If not provided, compare all stock pools (compare_level='pool') or compare all factors (compare_level='factor')
            If provided, only compare the items in the list.
        commission : float, optional, default 0
            Trading commission (one-way), 1/10000 for one basis point
        nv_type : str, {'l', 'ls'}, default 'l'
            * l: Plot long portfolio net value
            * ls: Plot long-short portfolio net value
        excess_return : bool, default False
            If True, use excess returns; if False, use absolute returns

        Returns
        -------
        Comparison plot of net value curves
        """
        if nv_type == 'l' and excess_return is True:
            suffix = 'Long Excess'
        elif nv_type == 'l' and excess_return is False:
            suffix = 'Long Return'
        elif nv_type == 'ls':
            suffix = 'Long-Short Return'
        else:
            raise ValueError("Invalid input values")
        fig, ax1 = plt.subplots(figsize=(16, 8))
        backtest_name_list, backtest_list = self.get_backtest_list(compare_level=compare_level, target=target,
                                                                   filter_list=filter_list)
        for bt, bt_name in zip(backtest_list, backtest_name_list):
            nav_array = bt.get_single_net_value_array(nv_type=nv_type, commission=commission,
                                                      excess_return=excess_return, used_factor_freq=False)
            label = f'{bt_name}_{suffix}'
            ax1.plot(nav_array.index, nav_array.values, label=label)
        fig.legend(loc=2, bbox_transform=ax1.transAxes, bbox_to_anchor=(0, 0, 1, 1))
        plt.show()