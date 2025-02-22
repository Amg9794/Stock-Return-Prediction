#!/usr/bin/env python
# -*- coding: UTF-8 -*-


class FactorCapacity:
    def __init__(self):
        self.long_excess_annualized_retrun = None
        self.long_excess_drawdown = None
        self.long_information_ratio = None
        self.win_rate_longshort = None
        self.win_rate_long = None

        # Factor Coverage Analysis
        self.factor_coverage_array = None

        # Factor IC (Information Coefficient) Analysis
        self.ic_array = None
        self.quantile_ic_array = None
        self.grouped_ic_array = None

        self.ic_summary = None
        self.quantile_ic_summary = None

        # Factor Autocorrelation and Turnover Analysis
        self.autocorrelation_array = None
        self.turnover_array = None

        # Factor Return Analysis
        self.return_summary = None