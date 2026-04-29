import warnings
warnings.filterwarnings('ignore')
import time, gc
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, r2_score,
                              classification_report, confusion_matrix,
                              ConfusionMatrixDisplay)
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# configuration
SEED    = 42
SEEDS   = [42, 123, 456, 789, 2024]   # 5 seeds
N_FOLDS = 5
N_XGB   = 40    # Optuna trials
N_LGB   = 30
N_CAT   = 45
TUNE_FOLDS = 3
np.random.seed(SEED)
T0 = time.time()

def elapsed(): return f"[{(time.time()-T0)/60:.1f}m]"

print('XGBoost :', xgb.__version__)
print('LightGBM:', lgb.__version__)
print('CatBoost:', cb.__version__)

# load data and basic setup
train_raw = pd.read_csv('credit_train.csv')
test_raw  = pd.read_csv('credit_test.csv')
print(f'{elapsed()} Train {train_raw.shape} | Test {test_raw.shape}')
TARGET_CLS = 'RiskTier'
TARGET_REG = 'InterestRate'

CAT_COLS_BASE = [
    'EducationLevel', 'MaritalStatus', 'HomeOwnership', 'State',
    'EmploymentStatus', 'EmployerType', 'JobCategory', 'LoanPurpose',
    'CollateralType'
]

# feature engineering
def engineer(df, ref=None):
    d   = df.copy()
    ref = ref if ref is not None else df

    # missing-value indicators and missing-count summary
    miss_cols = [
        'PropertyValue','MortgageOutstandingBalance','StudentLoanOutstandingBalance',
        'InvestmentPortfolioValue','VehicleValue','AutoLoanOutstandingBalance',
        'CollateralValue','SecondaryMonthlyIncome','EmploymentLengthYears',
        'RevolvingUtilizationRate','TotalCreditLimit','CreditHistoryLengthMonths',
        'NumberOfPublicRecords','NumberOfBankruptcies','NumberOfHardInquiries12Mo'
    ]
    for c in miss_cols:
        if c in d.columns:
            d['Miss_' + c] = d[c].isna().astype('int8')
    d['TotalMissingCount'] = d[[c for c in miss_cols if c in d.columns]].isna().sum(axis=1)

    # delinquency and derogatory-history signals
    d['TotalLatePayments'] = (
        d['NumberOfLatePayments30Days'].fillna(0) +
        d['NumberOfLatePayments60Days'].fillna(0) +
        d['NumberOfLatePayments90Days'].fillna(0)
    )
    d['WeightedDerog'] = (
        d['NumberOfLatePayments30Days'].fillna(0) * 1 +
        d['NumberOfLatePayments60Days'].fillna(0) * 2 +
        d['NumberOfLatePayments90Days'].fillna(0) * 4 +
        d['NumberOfChargeOffs'].fillna(0) * 6 +
        d['NumberOfCollections'].fillna(0) * 5 +
        d['NumberOfBankruptcies'].fillna(0) * 10 +
        d['NumberOfPublicRecords'].fillna(0) * 3
    )
    co      = d['NumberOfChargeOffs'].fillna(0)
    bankr   = d['NumberOfBankruptcies'].fillna(0)
    colls   = d['NumberOfCollections'].fillna(0)
    util_med = ref['RevolvingUtilizationRate'].median() if 'RevolvingUtilizationRate' in ref.columns else 0.35
    dti_med  = ref['DebtToIncomeRatio'].median()        if 'DebtToIncomeRatio'        in ref.columns else 0.30
    util    = d['RevolvingUtilizationRate'].fillna(util_med)
    dti     = d['DebtToIncomeRatio'].fillna(dti_med)

    d['HasAnyDerog']    = (d['WeightedDerog'] > 0).astype('int8')
    d['HasSevereDerog'] = ((co > 0) | (colls > 0) | (bankr > 0)).astype('int8')
    d['HasBankruptcy']  = (bankr > 0).astype('int8')
    d['HasChargeOff']   = (co > 0).astype('int8')
    d['SqrtDerog']      = np.sqrt(d['WeightedDerog'])
    d['DerogRate']      = d['WeightedDerog'] / (d['CreditHistoryLengthMonths'].fillna(12)/12 + 1)
    d['DerogWorsening'] = d['NumberOfLatePayments90Days'].fillna(0) - d['NumberOfLatePayments30Days'].fillna(0)
    d['RecentDerogFrac']= d['NumberOfLatePayments30Days'].fillna(0) / (d['TotalLatePayments'] + 1)
    d['InquiryConc']    = d['NumberOfHardInquiries12Mo'].fillna(0) / (d['NumberOfHardInquiries24Mo'] + 1)

    # charge-off interaction features
    d['LogChargeOffs']         = np.log1p(co)
    d['ChargeOffs_x_Util']     = co * util
    d['ChargeOffs_x_Bankrupt'] = co * bankr
    d['ChargeOffs_x_Late']     = co * d['TotalLatePayments']
    d['ChargeOffs_x_DTI']      = co * dti
    d['SevereDerogScore']      = co * 3 + bankr * 4 + colls * 2
    # higher-order interaction terms
    d['Co_x_Bu_x_Util']        = co * bankr * util
    d['Co_x_Late_x_Util']      = co * d['TotalLatePayments'] * util

    # revolving utilization features
    d['UtilBucket']     = pd.cut(util, bins=[-0.01, 0.10, 0.30, 0.50, 0.75, 1.01],
                                  labels=[0, 1, 2, 3, 4]).astype(float)
    d['UtilSquared']    = util ** 2
    d['LogUtil']        = np.log1p(util)
    d['UtilXDerog']     = util * d['WeightedDerog']
    d['AvailableCredit']      = d['TotalCreditLimit'].fillna(0) * (1 - util.clip(0, 1))
    d['AvailCreditToLoan']    = d['AvailableCredit'] / (d['RequestedLoanAmount'] + 1)

    # credit history and account structure features
    d['AccountHealthRatio']   = d['NumberOfSatisfactoryAccounts'] / (d['NumberOfOpenAccounts'] + 1)
    d['AccountAgeDiversity']  = d['OldestAccountAgeMonths'] - d['AverageAccountAgeMonths']
    d['CreditHistPerAcct']    = d['CreditHistoryLengthMonths'].fillna(0) / (d['NumberOfOpenAccounts'] + 1)
    d['CreditLimitPerCard']   = d['TotalCreditLimit'].fillna(0) / (d['NumberOfCreditCards'] + 1)
    d['CreditLimitPerAcct']   = d['TotalCreditLimit'].fillna(0) / (d['NumberOfOpenAccounts'] + 1)
    d['LogCreditHistory']     = np.log1p(d['CreditHistoryLengthMonths'].fillna(0))
    d['CreditAgeToLife']      = d['CreditHistoryLengthMonths'].fillna(0) / (d['Age'] * 12)
    d['CreditMixScore']       = (
        (d['NumberOfCreditCards'] > 0).astype(int) +
        (d['MortgageOutstandingBalance'].fillna(0) > 0).astype(int) +
        (d['AutoLoanOutstandingBalance'].fillna(0) > 0).astype(int) +
        (d['StudentLoanOutstandingBalance'].fillna(0) > 0).astype(int)
    )
    d['SatisfactoryRatio']    = d['NumberOfSatisfactoryAccounts'] / (d['NumberOfOpenAccounts'] + 1)
    d['InquiryTrend']         = d['NumberOfHardInquiries12Mo'].fillna(0) / (d['NumberOfHardInquiries24Mo'] + 1)
    d['LatePaymentRatio']     = d['TotalLatePayments'] / (d['NumberOfOpenAccounts'] + 1)
    d['ChargeOffRatio']       = co / (d['NumberOfOpenAccounts'] + 1)
    d['DerogPerAccount']      = d['WeightedDerog'] / (d['NumberOfOpenAccounts'] + 1)
    d['DerogPerYear']         = d['WeightedDerog'] / (d['CreditHistoryLengthMonths'].fillna(12)/12 + 1)

    # credit-score-style composite
    pay_s  = np.clip(1 - d['WeightedDerog'] / 20, 0, 1)
    util_s = np.clip(1 - util, 0, 1)
    len_s  = np.clip(d['CreditHistoryLengthMonths'].fillna(0) / 360, 0, 1)
    mix_s  = d['CreditMixScore'] / 4
    new_s  = np.clip(1 - d['NumberOfHardInquiries12Mo'].fillna(0) / 6, 0, 1)
    d['FICOScore']      = pay_s*0.35 + util_s*0.30 + len_s*0.15 + mix_s*0.10 + new_s*0.10
    d['FICOScore_sq']   = d['FICOScore'] ** 2
    d['InquiryDistress']= d['NumberOfHardInquiries12Mo'].fillna(0) * (1 - d['FICOScore'])

    # income, debt, and affordability features
    d['LogAnnualIncome']   = np.log1p(d['AnnualIncome'])
    d['LogMonthlyIncome']  = np.log1p(d['TotalMonthlyIncome'])
    d['SecondaryIncRatio'] = d['SecondaryMonthlyIncome'].fillna(0) / (d['TotalMonthlyIncome'] + 1)
    d['HasSecondaryInc']   = (d['SecondaryMonthlyIncome'].fillna(0) > 0).astype('int8')
    d['IncomeConsistency'] = d['AnnualIncome'] / (d['MonthlyGrossIncome'] * 12 + 1e-5)
    d['DisposableIncome']  = d['TotalMonthlyIncome'] - d['MonthlyPaymentEstimate'].fillna(0)
    d['DisposableRatio']   = d['DisposableIncome'] / (d['TotalMonthlyIncome'] + 1)
    d['TotalOutstandDebt'] = (
        d['MortgageOutstandingBalance'].fillna(0) +
        d['AutoLoanOutstandingBalance'].fillna(0) +
        d['StudentLoanOutstandingBalance'].fillna(0)
    )
    d['TotalDebtToIncome'] = (d['TotalOutstandDebt'] + d['RequestedLoanAmount']) / (d['AnnualIncome'] + 1)
    d['DTISquared']        = dti ** 2
    d['LogDTI']            = np.log1p(d['DebtToIncomeRatio'].fillna(0))
    monthly_oblig = (
        d['MonthlyPaymentEstimate'].fillna(0) +
        d['MortgageOutstandingBalance'].fillna(0) / 360 * 2 +
        d['AutoLoanOutstandingBalance'].fillna(0) / 60 +
        d['StudentLoanOutstandingBalance'].fillna(0) / 120
    )
    d['TotalMonthlyOblig']  = monthly_oblig
    d['ObligCovRatio']      = monthly_oblig / (d['TotalMonthlyIncome'] + 1)
    d['IncomePerDependent'] = d['AnnualIncome'] / (d['NumberOfDependents'] + 1)
    d['MonthlyIncToLoan']   = d['TotalMonthlyIncome'] / (d['RequestedLoanAmount'] + 1)
    d['IsFullTime']         = (d['EmploymentStatus'] == 'Full-Time').astype('int8')
    d['IsUnemployed']       = (d['EmploymentStatus'] == 'Unemployed').astype('int8')

    # assets, liquidity, and collateral features
    d['NetWorth']          = d['TotalAssets'] - d['TotalOutstandDebt']
    d['LogNetWorth']       = np.log1p(np.maximum(d['NetWorth'], 0))
    d['LogTotalAssets']    = np.log1p(d['TotalAssets'])
    d['AssetToLoanRatio']  = d['TotalAssets'] / (d['RequestedLoanAmount'] + 1)
    d['AssetToDebtRatio']  = d['TotalAssets'] / (d['TotalOutstandDebt'] + 1)
    d['LiquidAssets']      = d['SavingsBalance'].fillna(0) + d['CheckingBalance'].fillna(0)
    d['LiquidityRatio']    = d['LiquidAssets'] / (d['TotalAssets'] + 1)
    d['EmergencyFundMo']   = d['LiquidAssets'] / (d['TotalMonthlyIncome'] + 1)
    d['SavingsRate']       = d['SavingsBalance'].fillna(0) / (d['AnnualIncome'] + 1)
    d['LogSavings']        = np.log1p(d['SavingsBalance'].fillna(0))
    d['HasInvestments']    = (d['InvestmentPortfolioValue'].fillna(0) > 0).astype('int8')
    d['InvestToIncome']    = d['InvestmentPortfolioValue'].fillna(0) / (d['AnnualIncome'] + 1)
    d['PropertyEquity']    = d['PropertyValue'].fillna(0) - d['MortgageOutstandingBalance'].fillna(0)
    d['PropertyEquityPct'] = d['PropertyEquity'] / (d['PropertyValue'].fillna(0) + 1)
    d['UnderwaterMortgage']= (d['PropertyEquity'] < 0).astype('int8')
    d['NetLTVRatio']       = d['RequestedLoanAmount'] / (d['TotalAssets'] + 1)
    d['LoanToLiquidRatio'] = d['RequestedLoanAmount'] / (d['LiquidAssets'] + 1)
    d['SavingsToLoanRatio']= d['SavingsBalance'].fillna(0) / (d['RequestedLoanAmount'] + 1)
    d['LiquidToLoanRatio'] = d['LiquidAssets'] / (d['RequestedLoanAmount'] + 1)
    d['HasProperty']       = (d['PropertyValue'].fillna(0) > 0).astype('int8')

    # loan request features
    purpose_risk = {'Debt Consolidation':3,'Home Improvement':1,'Auto':1,
                    'Personal':2,'Medical':2,'Education':1,'Business':3,'Vacation':3,'Other':2}
    d['LoanPurposeRisk']   = d['LoanPurpose'].map(purpose_risk).fillna(2)
    d['IsDebtConsol']      = (d['LoanPurpose'] == 'Debt Consolidation').astype('int8')
    d['IsLongTerm']        = (d['RequestedTermMonths'] > 48).astype('int8')
    d['LogLoanAmount']     = np.log1p(d['RequestedLoanAmount'])
    collat_rank            = {'Property':3,'Vehicle':2,'Savings Account':1,'Other':1}
    d['CollateralQuality'] = d['CollateralType'].map(collat_rank).fillna(0)
    d['HasCollateral']     = (d['CollateralValue'].fillna(0) > 0).astype('int8')
    d['CollateralCoverage']= d['CollateralValue'].fillna(0) / (d['RequestedLoanAmount'] + 1)
    d['ConsolDebtTrap']    = d['IsDebtConsol'] * dti
    d['BankruptConsol']    = d['HasBankruptcy'] * d['IsDebtConsol']

    # demographic and employment features
    edu_rank = {'No Diploma':0,'HS Diploma':1,'Some College':2,'Bachelor':3,'Master':4,'PhD':5}
    d['EducationRank']     = d['EducationLevel'].map(edu_rank).fillna(1)
    d['IsHighEducation']   = d['EducationLevel'].isin(['Bachelor','Master','PhD']).astype('int8')
    d['IsPrimeAge']        = ((d['Age']>=35) & (d['Age']<=55)).astype('int8')
    d['IsYoungBorrower']   = (d['Age'] < 30).astype('int8')
    d['AgeSquared']        = d['Age'] ** 2
    emp_stab = {'Full-Time':4,'Government':4,'Self-Employed':2,'Part-Time':2,'Retired':3,'Unemployed':0}
    d['EmpStability']      = d['EmploymentStatus'].map(emp_stab).fillna(2)
    d['LogEmpLength']      = np.log1p(d['EmploymentLengthYears'].fillna(0))
    d['LogYrsAtEmployer']  = np.log1p(d['YearsAtCurrentEmployer'].fillna(0))
    d['JobTenureRatio']    = d['YearsAtCurrentEmployer'].fillna(0) / (d['EmploymentLengthYears'].fillna(1) + 0.1)
    d['EmpStabilityRatio'] = d['YearsAtCurrentEmployer'].fillna(0) / (d['EmploymentLengthYears'].fillna(0) + 1)
    emp_risk = {'Government':1,'Corporate':2,'Non-Profit':2,'SME':3,'Self':3}
    d['EmployerRisk']      = d['EmployerType'].map(emp_risk).fillna(2)
    home_stab = {'Own':4,'Mortgage':3,'Rent':2,'Living with Family':1}
    d['HomeOwnerScore']    = d['HomeOwnership'].map(home_stab).fillna(2)
    d['IsHomeowner']       = d['HomeOwnership'].isin(['Own','Mortgage']).astype('int8')
    d['DependentBurden']   = d['NumberOfDependents'] / (d['TotalMonthlyIncome'] / 1000 + 1)
    d['HasMinorDep']       = (d['NumberOfDependentsUnder18'] > 0).astype('int8')
    d['IncomeStabilScore'] = d['LogAnnualIncome'] * d['EmpStability']

    # composite lending-style scores
    d['CapacityScore'] = (
        np.clip(1 - dti, 0, 1) * 0.40 +
        np.clip(d['LiquidityRatio'], 0, 1) * 0.30 +
        (d['EmpStability'] / 4) * 0.30
    )
    d['CharacterScore'] = (
        d['PreviousLoanWithBank'] * 0.20 +
        d['HasCoApplicant'] * 0.15 +
        (d['HomeOwnerScore'] / 4) * 0.25 +
        (d['EducationRank'] / 5) * 0.20 +
        (d['EmpStability'] / 4) * 0.20
    )
    d['CollateralScore'] = np.clip(
        d['CollateralCoverage'] * 0.5 +
        (d['CollateralQuality'] / 3) * 0.3 +
        (d['AssetToLoanRatio'] / 10) * 0.2, 0, 1
    )
    d['CreditworthScore'] = d['FICOScore']*0.35 + d['CapacityScore']*0.25 + d['CharacterScore']*0.15 + d['CollateralScore']*0.10

    # aggregate risk and stability scores
    d['CreditRiskScore'] = (
        util * 20 + d['WeightedDerog'] * 5 +
        dti * 10 - d['SatisfactoryRatio'] * 10 - d['LogCreditHistory'] * 2
    )
    d['StabilityScore'] = (
        d['LogCreditHistory'] * 2 + d['YearsAtCurrentEmployer'].fillna(0) * 3 +
        d['ResidencyYears'].fillna(0) + d['IsFullTime'] * 5 + d['IsHomeowner'] * 3
    )
    d['AffordabilityScore'] = (
        d['LogAnnualIncome'] * 3 + d['LogNetWorth'] * 2 -
        d['LoanToIncomeRatio'] * 5 - d['PaymentToIncomeRatio'] * 10
    )

    # nonlinear transforms
    d['DerogSeverity_sq'] = d['WeightedDerog'] ** 2
    d['RevUtil_sq']        = util ** 2
    d['TotalLate_sq']      = d['TotalLatePayments'] ** 2
    d['ChargeOffs_sq']     = co ** 2
    d['CreditRisk_sq']     = d['CreditRiskScore'] ** 2
    d['TotalLate_log']     = np.log1p(d['TotalLatePayments'])
    d['Severity_log']      = np.log1p(d['WeightedDerog'])
    d['CreditRisk_log']    = np.log1p(d['CreditRiskScore'].clip(lower=0))
    d['Income_sq']         = d['LogAnnualIncome'] ** 2
    d['DTI_sq']            = dti ** 2

    # cross-feature interactions
    d['FICOxDTI']             = d['FICOScore'] * dti
    d['FICOxIncome']          = d['FICOScore'] * d['LogAnnualIncome']
    d['FICOxNetWorth']        = d['FICOScore'] * d['LogNetWorth']
    d['FICOxDerog']           = d['FICOScore'] * d['WeightedDerog']
    d['DerogxUtil']           = d['WeightedDerog'] * util
    d['CreditworthxLoan']     = d['CreditworthScore'] * d['LogLoanAmount']
    d['IncomePerDebt']        = d['LogAnnualIncome'] / (d['TotalDebtToIncome'] + 0.01)
    d['DelinquencyXUtil']     = d['TotalLatePayments'] * util
    d['SeverityXUtil']        = d['WeightedDerog'] * util
    d['LateXDTI']             = d['TotalLatePayments'] * dti
    d['ChargeOffXUtil']       = co * util
    d['IncomeXSatisfactory']  = d['LogAnnualIncome'] * d['SatisfactoryRatio']
    d['IncomeXUtil']          = d['LogAnnualIncome'] * util
    d['HistoryXDelinq']       = d['LogCreditHistory'] * d['WeightedDerog']
    d['NetWorthXDelinq']      = d['LogNetWorth'] * d['WeightedDerog']
    d['LiquidXFICO']          = d['LiquidityRatio'] * d['FICOScore']
    d['SatisfXHistory']       = d['SatisfactoryRatio'] * d['LogCreditHistory']
    d['CoXNetWorth']          = co * d['LogNetWorth']

    # relative position within job category
    job_stats = ref.groupby('JobCategory')['AnnualIncome'].agg(['median','std']).rename(
        columns={'median':'_JMedI','std':'_JStdI'}).reset_index()
    d = d.merge(job_stats, on='JobCategory', how='left')
    d['IncomeVsPeers'] = (d['AnnualIncome'] - d['_JMedI']) / (d['_JStdI'] + 1)
    d.drop(['_JMedI','_JStdI'], axis=1, inplace=True)

    # relative position within state
    state_stats = ref.groupby('State').agg(
        _SMedI=('AnnualIncome','median'),
        _SMedU=('RevolvingUtilizationRate','median'),
        _SMedD=('NumberOfChargeOffs','median'),
    ).reset_index()
    d = d.merge(state_stats, on='State', how='left')
    d['IncomeVsState'] = d['AnnualIncome'] / (d['_SMedI'] + 1)
    d['UtilVsState']   = d['RevolvingUtilizationRate'].fillna(util_med) - d['_SMedU'].fillna(0)
    d.drop(['_SMedI','_SMedU','_SMedD'], axis=1, inplace=True)

    # frequency encoding for key categoricals
    for col in ['State','LoanPurpose','JobCategory','EmployerType']:
        freq = ref[col].value_counts(normalize=True)
        d[f'{col}_freq'] = d[col].map(freq).fillna(0)

    # combined categorical labels
    d['Edu_Home']     = d['EducationLevel'].astype(str) + '_' + d['HomeOwnership'].astype(str)
    d['Emp_Home']     = d['EmploymentStatus'].astype(str) + '_' + d['HomeOwnership'].astype(str)
    d['Purpose_Term'] = d['LoanPurpose'].astype(str) + '_' + d['RequestedTermMonths'].astype(str)
    d['Edu_Emp']      = d['EducationLevel'].astype(str) + '_' + d['EmploymentStatus'].astype(str)
    d['State_Edu']    = d['State'].astype(str) + '_' + d['EducationLevel'].astype(str)
    d['Emp_State']    = d['EmploymentStatus'].astype(str) + '_' + d['State'].astype(str)
    d['Purpose_Emp']  = d['LoanPurpose'].astype(str) + '_' + d['EmploymentStatus'].astype(str)

    return d

print(f'{elapsed()} Engineering features...')
train_fe = engineer(train_raw, train_raw)
test_fe  = engineer(test_raw,  train_raw)
print(f'{elapsed()} Columns → train:{train_fe.shape[1]} test:{test_fe.shape[1]}')

# out-of-fold target encoding
COMBO_COLS  = ['Edu_Home','Emp_Home','Purpose_Term','Edu_Emp','State_Edu',
               'Emp_State','Purpose_Emp']
ALL_TE_COLS = CAT_COLS_BASE + COMBO_COLS

def target_encode_oof(tr, te, cols, target, n_splits=5, smooth=10):
    gm  = tr[target].mean()
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=SEED)
    for col in cols:
        nc = f'{col}_te_{target[:4]}'
        tr[nc] = gm
        for ti, vi in skf.split(tr, tr['RiskTier']):
            agg = tr.iloc[ti].groupby(col)[target].agg(['mean','count'])
            sm  = (agg['count']*agg['mean'] + smooth*gm) / (agg['count'] + smooth)
            tr.loc[tr.index[vi], nc] = tr.iloc[vi][col].map(sm).fillna(gm).values
        agg_f = tr.groupby(col)[target].agg(['mean','count'])
        sm_f  = (agg_f['count']*agg_f['mean'] + smooth*gm) / (agg_f['count'] + smooth)
        te[nc] = te[col].map(sm_f).fillna(gm).values
    return tr, te

print(f'{elapsed()} Target encoding RiskTier...')
train_fe, test_fe = target_encode_oof(train_fe, test_fe, ALL_TE_COLS, 'RiskTier')
print(f'{elapsed()} Target encoding InterestRate...')
train_fe, test_fe = target_encode_oof(train_fe, test_fe, ALL_TE_COLS, 'InterestRate')
print(f'{elapsed()} After TE: {train_fe.shape[1]} features')

# encode categorical features
ALL_CAT_COLS = CAT_COLS_BASE + COMBO_COLS
for col in ALL_CAT_COLS:
    le = LabelEncoder()
    combined = pd.concat([train_fe[col], test_fe[col]]).astype(str).fillna('__NA__')
    le.fit(combined)
    train_fe[col] = le.transform(train_fe[col].astype(str).fillna('__NA__'))
    test_fe[col]  = le.transform(test_fe[col].astype(str).fillna('__NA__'))

for c in train_fe.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    combined = pd.concat([train_fe[c], test_fe[c]]).astype(str).fillna('__NA__')
    le.fit(combined)
    train_fe[c] = le.transform(train_fe[c].astype(str).fillna('__NA__'))
    test_fe[c]  = le.transform(test_fe[c].astype(str).fillna('__NA__'))

y_cls = train_fe[TARGET_CLS].values
y_reg = train_fe[TARGET_REG].values
train_fe.drop(columns=[TARGET_CLS, TARGET_REG], inplace=True)
test_fe.drop(columns=[c for c in [TARGET_CLS, TARGET_REG] if c in test_fe.columns], inplace=True)

train_fe = train_fe.fillna(train_fe.median(numeric_only=True))
test_fe  = test_fe.fillna(train_fe.median(numeric_only=True))
train_fe.replace([np.inf, -np.inf], 0, inplace=True)
test_fe.replace([np.inf, -np.inf],  0, inplace=True)

feat_names = train_fe.columns.tolist()
print(f'{elapsed()} Features after encoding: {len(feat_names)}')

# rank and group-relative features
combined_rk = pd.concat([train_fe, test_fe], axis=0, ignore_index=True)
n_tr_rk = len(train_fe)

for col in [c for c in [
    'AnnualIncome','TotalAssets','RevolvingUtilizationRate','DebtToIncomeRatio',
    'CreditHistoryLengthMonths','TotalCreditLimit','RequestedLoanAmount',
    'WeightedDerog','FICOScore','CreditRiskScore','NetWorth','LiquidAssets',
    'TotalLatePayments'
] if c in combined_rk.columns]:
    combined_rk[f'{col}_pctrank'] = combined_rk[col].rank(pct=True, method='average')

for grp in ['EducationLevel','HomeOwnership','EmploymentStatus','State']:
    if grp in combined_rk.columns:
        combined_rk[f'IncRank_{grp}']    = combined_rk.groupby(grp)['AnnualIncome'].rank(pct=True)
        combined_rk[f'UtilRank_{grp}']   = combined_rk.groupby(grp)['RevolvingUtilizationRate'].rank(pct=True)
        combined_rk[f'DScoreRank_{grp}'] = combined_rk.groupby(grp)['WeightedDerog'].rank(pct=True)

train_fe  = combined_rk.iloc[:n_tr_rk].reset_index(drop=True)
test_fe   = combined_rk.iloc[n_tr_rk:].reset_index(drop=True)
feat_names = train_fe.columns.tolist()
print(f'{elapsed()} After rank features: {len(feat_names)} features')

# feature pruning
Xfull    = train_fe.values.astype('float32')
Xte_full = test_fe.values.astype('float32')

print(f'{elapsed()} Quick importance pass for pruning...')
_qm = xgb.XGBClassifier(
    n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, tree_method='hist', device='cuda',
    objective='multi:softprob', num_class=5, random_state=SEED, n_jobs=-1
)
_qm.fit(Xfull, y_cls, verbose=False)
importances = pd.Series(_qm.feature_importances_, index=feat_names)

keep_imp = importances[importances >= importances.max() * 0.0018].index.tolist()
print(f'{elapsed()} Importance pruning: {len(feat_names)} → {len(keep_imp)}')

corr_mat  = train_fe[keep_imp].corr().abs()
upper     = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
drop_corr = [c for c in upper.columns if any(upper[c] > 0.97)]
keep_final = [f for f in keep_imp if f not in drop_corr]
print(f'{elapsed()} Correlation pruning: {len(keep_imp)} → {len(keep_final)}')

train_fe   = train_fe[keep_final]
test_fe    = test_fe[keep_final]
feat_names = keep_final

Xtr = train_fe.values.astype('float32')
Xte = test_fe.values.astype('float32')
n_tr = len(Xtr); n_te = len(Xte)
print(f'{elapsed()} Final: {len(feat_names)} features | class dist: {np.bincount(y_cls)}')

# hyperparameter tuning with Optuna

def xgb_cls_obj(trial):
    p = dict(
        n_estimators      = trial.suggest_int('n_estimators', 500, 1800),
        max_depth         = trial.suggest_int('max_depth', 4, 9),
        learning_rate     = trial.suggest_float('learning_rate', 0.008, 0.10, log=True),
        subsample         = trial.suggest_float('subsample', 0.60, 1.0),
        colsample_bytree  = trial.suggest_float('colsample_bytree', 0.45, 1.0),
        colsample_bylevel = trial.suggest_float('colsample_bylevel', 0.45, 1.0),
        min_child_weight  = trial.suggest_int('min_child_weight', 1, 20),
        gamma             = trial.suggest_float('gamma', 0, 1.0),
        reg_alpha         = trial.suggest_float('reg_alpha', 1e-4, 8.0, log=True),
        reg_lambda        = trial.suggest_float('reg_lambda', 1e-4, 8.0, log=True),
    )
    skf = StratifiedKFold(TUNE_FOLDS, shuffle=True, random_state=SEED)
    scores = []
    for ti, vi in skf.split(Xtr, y_cls):
        m = xgb.XGBClassifier(**p, tree_method='hist', device='cuda',
                               objective='multi:softprob', num_class=5,
                               eval_metric='mlogloss', early_stopping_rounds=60,
                               random_state=SEED, n_jobs=-1)
        m.fit(Xtr[ti], y_cls[ti], eval_set=[(Xtr[vi], y_cls[vi])], verbose=False)
        scores.append(accuracy_score(y_cls[vi], m.predict(Xtr[vi])))
    return np.mean(scores)

def xgb_reg_obj(trial):
    p = dict(
        n_estimators      = trial.suggest_int('n_estimators', 500, 1800),
        max_depth         = trial.suggest_int('max_depth', 4, 9),
        learning_rate     = trial.suggest_float('learning_rate', 0.008, 0.10, log=True),
        subsample         = trial.suggest_float('subsample', 0.60, 1.0),
        colsample_bytree  = trial.suggest_float('colsample_bytree', 0.45, 1.0),
        colsample_bylevel = trial.suggest_float('colsample_bylevel', 0.45, 1.0),
        min_child_weight  = trial.suggest_int('min_child_weight', 1, 20),
        gamma             = trial.suggest_float('gamma', 0, 1.0),
        reg_alpha         = trial.suggest_float('reg_alpha', 1e-4, 8.0, log=True),
        reg_lambda        = trial.suggest_float('reg_lambda', 1e-4, 8.0, log=True),
    )
    kf = KFold(TUNE_FOLDS, shuffle=True, random_state=SEED)
    scores = []
    for ti, vi in kf.split(Xtr):
        m = xgb.XGBRegressor(**p, tree_method='hist', device='cuda',
                              objective='reg:squarederror', early_stopping_rounds=60,
                              random_state=SEED, n_jobs=-1)
        m.fit(Xtr[ti], y_reg[ti], eval_set=[(Xtr[vi], y_reg[vi])], verbose=False)
        scores.append(r2_score(y_reg[vi], m.predict(Xtr[vi])))
    return np.mean(scores)

def lgb_cls_obj(trial):
    # LGB — CPU only (no CUDA back-end in LightGBM)
    p = dict(
        n_estimators     = trial.suggest_int('n_estimators', 500, 1800),
        learning_rate    = trial.suggest_float('learning_rate', 0.008, 0.08, log=True),
        num_leaves       = trial.suggest_int('num_leaves', 63, 255),
        max_depth        = trial.suggest_int('max_depth', 5, 12),
        min_child_samples= trial.suggest_int('min_child_samples', 10, 60),
        subsample        = trial.suggest_float('subsample', 0.60, 1.0),
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.45, 1.0),
        reg_alpha        = trial.suggest_float('reg_alpha', 1e-4, 8.0, log=True),
        reg_lambda       = trial.suggest_float('reg_lambda', 1e-4, 8.0, log=True),
        min_split_gain   = trial.suggest_float('min_split_gain', 0.0, 1.0),
    )
    skf = StratifiedKFold(TUNE_FOLDS, shuffle=True, random_state=SEED)
    scores = []
    for ti, vi in skf.split(Xtr, y_cls):
        m = lgb.LGBMClassifier(**p, objective='multiclass', num_class=5,
                                metric='multi_logloss', verbose=-1, n_jobs=-1,
                                random_state=SEED)
        m.fit(Xtr[ti], y_cls[ti], eval_set=[(Xtr[vi], y_cls[vi])],
              callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(-1)])
        scores.append(accuracy_score(y_cls[vi], m.predict(Xtr[vi])))
    return np.mean(scores)

def lgb_reg_obj(trial):
    p = dict(
        n_estimators     = trial.suggest_int('n_estimators', 500, 1800),
        learning_rate    = trial.suggest_float('learning_rate', 0.008, 0.08, log=True),
        num_leaves       = trial.suggest_int('num_leaves', 63, 255),
        max_depth        = trial.suggest_int('max_depth', 5, 12),
        min_child_samples= trial.suggest_int('min_child_samples', 10, 60),
        subsample        = trial.suggest_float('subsample', 0.60, 1.0),
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.45, 1.0),
        reg_alpha        = trial.suggest_float('reg_alpha', 1e-4, 8.0, log=True),
        reg_lambda       = trial.suggest_float('reg_lambda', 1e-4, 8.0, log=True),
        min_split_gain   = trial.suggest_float('min_split_gain', 0.0, 1.0),
    )
    kf = KFold(TUNE_FOLDS, shuffle=True, random_state=SEED)
    scores = []
    for ti, vi in kf.split(Xtr):
        m = lgb.LGBMRegressor(**p, objective='regression', metric='rmse',
                               verbose=-1, n_jobs=-1, random_state=SEED)
        m.fit(Xtr[ti], y_reg[ti], eval_set=[(Xtr[vi], y_reg[vi])],
              callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(-1)])
        scores.append(r2_score(y_reg[vi], m.predict(Xtr[vi])))
    return np.mean(scores)

def cat_cls_obj(trial):
    p = dict(
        iterations          = trial.suggest_int('iterations', 800, 3000),
        learning_rate       = trial.suggest_float('learning_rate', 0.008, 0.08, log=True),
        depth               = trial.suggest_int('depth', 6, 10),
        l2_leaf_reg         = trial.suggest_float('l2_leaf_reg', 0.5, 15, log=True),
        bagging_temperature = trial.suggest_float('bagging_temperature', 0, 1),
        random_strength     = trial.suggest_float('random_strength', 0.0, 2.0),
        min_data_in_leaf    = trial.suggest_int('min_data_in_leaf', 3, 30),
    )
    skf = StratifiedKFold(TUNE_FOLDS, shuffle=True, random_state=SEED)
    scores = []
    for ti, vi in skf.split(Xtr, y_cls):
        m = cb.CatBoostClassifier(**p, loss_function='MultiClass', eval_metric='Accuracy',
                                   task_type='GPU', devices='0', random_seed=SEED,
                                   verbose=0, early_stopping_rounds=60)
        m.fit(Xtr[ti], y_cls[ti], eval_set=(Xtr[vi], y_cls[vi]), verbose=False)
        scores.append(accuracy_score(y_cls[vi], m.predict(Xtr[vi]).ravel()))
    return np.mean(scores)

def cat_reg_obj(trial):
    p = dict(
        iterations          = trial.suggest_int('iterations', 800, 3000),
        learning_rate       = trial.suggest_float('learning_rate', 0.008, 0.08, log=True),
        depth               = trial.suggest_int('depth', 6, 10),
        l2_leaf_reg         = trial.suggest_float('l2_leaf_reg', 0.5, 15, log=True),
        bagging_temperature = trial.suggest_float('bagging_temperature', 0, 1),
        random_strength     = trial.suggest_float('random_strength', 0.0, 2.0),
        min_data_in_leaf    = trial.suggest_int('min_data_in_leaf', 3, 30),
    )
    kf = KFold(TUNE_FOLDS, shuffle=True, random_state=SEED)
    scores = []
    for ti, vi in kf.split(Xtr):
        m = cb.CatBoostRegressor(**p, loss_function='RMSE', task_type='GPU', devices='0',
                                  random_seed=SEED, verbose=0, early_stopping_rounds=60)
        m.fit(Xtr[ti], y_reg[ti], eval_set=(Xtr[vi], y_reg[vi]), verbose=False)
        scores.append(r2_score(y_reg[vi], m.predict(Xtr[vi])))
    return np.mean(scores)

print(f'{elapsed()} === Optuna XGB cls ({N_XGB}) ===')
sc_xgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
sc_xgb.optimize(xgb_cls_obj, n_trials=N_XGB, show_progress_bar=True)
best_xgb_cls = sc_xgb.best_params
print(f'{elapsed()} XGB cls best: {sc_xgb.best_value:.4f}')

print(f'{elapsed()} === Optuna XGB reg ({N_XGB}) ===')
sr_xgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
sr_xgb.optimize(xgb_reg_obj, n_trials=N_XGB, show_progress_bar=True)
best_xgb_reg = sr_xgb.best_params
print(f'{elapsed()} XGB reg best: {sr_xgb.best_value:.4f}')

print(f'{elapsed()} === Optuna LGB cls ({N_LGB}) ===')
sc_lgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
sc_lgb.optimize(lgb_cls_obj, n_trials=N_LGB, show_progress_bar=True)
best_lgb_cls = sc_lgb.best_params
print(f'{elapsed()} LGB cls best: {sc_lgb.best_value:.4f}')

print(f'{elapsed()} === Optuna LGB reg ({N_LGB}) ===')
sr_lgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
sr_lgb.optimize(lgb_reg_obj, n_trials=N_LGB, show_progress_bar=True)
best_lgb_reg = sr_lgb.best_params
print(f'{elapsed()} LGB reg best: {sr_lgb.best_value:.4f}')

print(f'{elapsed()} === Optuna CAT cls ({N_CAT}) ===')
sc_cat = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
sc_cat.optimize(cat_cls_obj, n_trials=N_CAT, show_progress_bar=True)
best_cat_cls = sc_cat.best_params
print(f'{elapsed()} CAT cls best: {sc_cat.best_value:.4f}')

print(f'{elapsed()} === Optuna CAT reg ({N_CAT}) ===')
sr_cat = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
sr_cat.optimize(cat_reg_obj, n_trials=N_CAT, show_progress_bar=True)
best_cat_reg = sr_cat.best_params
print(f'{elapsed()} CAT reg best: {sr_cat.best_value:.4f}')

# final params from optuna (seed gets set per run)
xgb_cls_cfg = dict(tree_method='hist', device='cuda', n_jobs=-1,
                   objective='multi:softprob', num_class=5, eval_metric='mlogloss',
                   early_stopping_rounds=70, **best_xgb_cls)

xgb_reg_cfg = dict(tree_method='hist', device='cuda', n_jobs=-1,
                   objective='reg:squarederror', early_stopping_rounds=70,
                   **best_xgb_reg)

lgb_cls_cfg = dict(objective='multiclass', num_class=5, metric='multi_logloss',
                   verbose=-1, n_jobs=-1, **best_lgb_cls)

lgb_reg_cfg = dict(objective='regression', metric='rmse',
                   verbose=-1, n_jobs=-1, **best_lgb_reg)

cat_cls_cfg = dict(loss_function='MultiClass', eval_metric='Accuracy',
                   task_type='GPU', devices='0', verbose=0,
                   early_stopping_rounds=70, **best_cat_cls)

cat_reg_cfg = dict(loss_function='RMSE', task_type='GPU', devices='0',
                   verbose=0, early_stopping_rounds=70, **best_cat_reg)


# helper utilities
def ordinal_to_class_proba(bp):
    n = bp[0].shape[0]; p = np.zeros((n, 5))
    p[:, 0] = 1 - bp[0]
    for k in range(1, 4): p[:, k] = bp[k-1] - bp[k]
    p[:, 4] = bp[3]
    return np.clip(p, 1e-8, 1.0)


# level 1: 3 libraries x (multiclass, ordinal, regression), 5 seeds x 5 folds, all out-of-fold
MODEL_MC  = ['xgb_mc',  'lgb_mc',  'cat_mc']
MODEL_ORD = ['xgb_ord', 'lgb_ord', 'cat_ord']
MODEL_REG = ['xgb_reg', 'lgb_reg', 'cat_reg']
ALL_CLS   = MODEL_MC + MODEL_ORD

oof_cls = {n: np.zeros((n_tr, 5)) for n in ALL_CLS}
tst_cls = {n: np.zeros((n_te, 5)) for n in ALL_CLS}
oof_reg = {n: np.zeros(n_tr) for n in MODEL_REG}
tst_reg = {n: np.zeros(n_te) for n in MODEL_REG}

n_seeds = len(SEEDS)

for si, cur_seed in enumerate(SEEDS):
    print(f'\n{elapsed()} ===== SEED {cur_seed} ({si+1}/{n_seeds}) =====')
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=cur_seed)

    for fold, (ti, vi) in enumerate(skf.split(Xtr, y_cls)):
        print(f'  {elapsed()} FOLD {fold+1}/{N_FOLDS}')
        Xf, Xv   = Xtr[ti], Xtr[vi]
        yc_f, yc_v = y_cls[ti], y_cls[vi]
        yr_f, yr_v = y_reg[ti],  y_reg[vi]

        # xgboost multiclass model
        mc = xgb.XGBClassifier(**xgb_cls_cfg, random_state=cur_seed)
        mc.fit(Xf, yc_f, eval_set=[(Xv, yc_v)], verbose=False)
        oof_cls['xgb_mc'][vi] += mc.predict_proba(Xv)
        tst_cls['xgb_mc']    += mc.predict_proba(Xte)

        # xgboost ordinal model
        cfg_bin = {k:v for k,v in xgb_cls_cfg.items()
                   if k not in ('num_class','objective','eval_metric')}
        bps, bpt = [], []
        for thr in range(1, 5):
            yb = (yc_f >= thr).astype(int); yv = (yc_v >= thr).astype(int)
            mb = xgb.XGBClassifier(**cfg_bin, objective='binary:logistic',
                                    eval_metric='logloss', random_state=cur_seed)
            mb.fit(Xf, yb, eval_set=[(Xv, yv)], verbose=False)
            bps.append(mb.predict_proba(Xv)[:, 1])
            bpt.append(mb.predict_proba(Xte)[:, 1])
        oof_cls['xgb_ord'][vi] += ordinal_to_class_proba(bps)
        tst_cls['xgb_ord']     += ordinal_to_class_proba(bpt)

        # xgboost regression model
        mr = xgb.XGBRegressor(**xgb_reg_cfg, random_state=cur_seed)
        mr.fit(Xf, yr_f, eval_set=[(Xv, yr_v)], verbose=False)
        oof_reg['xgb_reg'][vi] += mr.predict(Xv)
        tst_reg['xgb_reg']    += mr.predict(Xte)

        # lightgbm multiclass model
        mc = lgb.LGBMClassifier(**lgb_cls_cfg, random_state=cur_seed)
        mc.fit(Xf, yc_f, eval_set=[(Xv, yc_v)],
               callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
        oof_cls['lgb_mc'][vi] += mc.predict_proba(Xv)
        tst_cls['lgb_mc']    += mc.predict_proba(Xte)

        # lightgbm ordinal model
        cfg_lbin = {k:v for k,v in lgb_cls_cfg.items()
                    if k not in ('num_class','objective','metric')}
        bps, bpt = [], []
        for thr in range(1, 5):
            yb = (yc_f >= thr).astype(int); yv = (yc_v >= thr).astype(int)
            mb = lgb.LGBMClassifier(**cfg_lbin, objective='binary',
                                     metric='binary_logloss', random_state=cur_seed)
            mb.fit(Xf, yb, eval_set=[(Xv, yv)],
                   callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
            bps.append(mb.predict_proba(Xv)[:, 1])
            bpt.append(mb.predict_proba(Xte)[:, 1])
        oof_cls['lgb_ord'][vi] += ordinal_to_class_proba(bps)
        tst_cls['lgb_ord']     += ordinal_to_class_proba(bpt)

        # lightgbm regression model
        mr = lgb.LGBMRegressor(**lgb_reg_cfg, random_state=cur_seed)
        mr.fit(Xf, yr_f, eval_set=[(Xv, yr_v)],
               callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
        oof_reg['lgb_reg'][vi] += mr.predict(Xv)
        tst_reg['lgb_reg']    += mr.predict(Xte)

        # catboost multiclass model
        mc = cb.CatBoostClassifier(**cat_cls_cfg, random_seed=cur_seed)
        mc.fit(Xf, yc_f, eval_set=(Xv, yc_v), verbose=False)
        oof_cls['cat_mc'][vi] += mc.predict_proba(Xv)
        tst_cls['cat_mc']    += mc.predict_proba(Xte)

        # catboost ordinal model
        cfg_cbin = {k:v for k,v in cat_cls_cfg.items()
                    if k not in ('loss_function','eval_metric')}
        bps, bpt = [], []
        for thr in range(1, 5):
            yb = (yc_f >= thr).astype(int); yv = (yc_v >= thr).astype(int)
            mb = cb.CatBoostClassifier(**cfg_cbin, loss_function='Logloss',
                                        eval_metric='AUC', random_seed=cur_seed)
            mb.fit(Xf, yb, eval_set=(Xv, yv), verbose=False)
            bps.append(mb.predict_proba(Xv)[:, 1])
            bpt.append(mb.predict_proba(Xte)[:, 1])
        oof_cls['cat_ord'][vi] += ordinal_to_class_proba(bps)
        tst_cls['cat_ord']     += ordinal_to_class_proba(bpt)

        # catboost regression model
        mr = cb.CatBoostRegressor(**cat_reg_cfg, random_seed=cur_seed)
        mr.fit(Xf, yr_f, eval_set=(Xv, yr_v), verbose=False)
        oof_reg['cat_reg'][vi] += mr.predict(Xv)
        tst_reg['cat_reg']    += mr.predict(Xte)

        # quick fold summary for monitoring
        fold_acc = accuracy_score(
            yc_v, (oof_cls['cat_ord'][vi]).argmax(1))
        fold_r2  = r2_score(
            yr_v, (oof_reg['xgb_reg'][vi] + oof_reg['lgb_reg'][vi] + oof_reg['cat_reg'][vi]) / 3)
        print(f'  {elapsed()} cat_ord acc={fold_acc:.4f} | avg_r2={fold_r2:.4f}')

    gc.collect()

# average predictions across seeds
n_sf = n_seeds * N_FOLDS
for n in ALL_CLS:
    oof_cls[n] /= n_seeds
    tst_cls[n] /= n_sf
for n in MODEL_REG:
    oof_reg[n] /= n_seeds
    tst_reg[n] /= n_sf

print(f'\n{elapsed()} === Individual OOF ===')
for n in ALL_CLS:
    print(f'  {n:10s}: acc={accuracy_score(y_cls, oof_cls[n].argmax(1)):.4f}')
for n in MODEL_REG:
    print(f'  {n:10s}: r2 ={r2_score(y_reg, np.clip(oof_reg[n], 4.99, 35.99)):.4f}')

# optimize base ensemble weights
print(f'\n{elapsed()} Optimising base weights...')

def neg_acc(w):
    ws = np.exp(w) / np.exp(w).sum()
    return -accuracy_score(y_cls, sum(oof_cls[k]*ws[i] for i,k in enumerate(ALL_CLS)).argmax(1))

def neg_r2(w):
    ws = np.exp(w) / np.exp(w).sum()
    blend = sum(oof_reg[k]*ws[i] for i,k in enumerate(MODEL_REG))
    return -r2_score(y_reg, np.clip(blend, 4.99, 35.99))

# random restarts, nelder-mead gets stuck otherwise
best_acc_val, best_w_cls = -np.inf, np.zeros(len(ALL_CLS))
for _ in range(10):
    w0 = np.random.randn(len(ALL_CLS)) * 0.5
    res = minimize(neg_acc, w0, method='Nelder-Mead',
                   options={'maxiter':3000, 'xatol':1e-6, 'fatol':1e-6})
    if -res.fun > best_acc_val:
        best_acc_val = -res.fun; best_w_cls = res.x.copy()

best_r2_val, best_w_reg = -np.inf, np.zeros(len(MODEL_REG))
for _ in range(10):
    w0 = np.random.randn(len(MODEL_REG)) * 0.5
    res = minimize(neg_r2, w0, method='Nelder-Mead',
                   options={'maxiter':3000, 'xatol':1e-6, 'fatol':1e-6})
    if -res.fun > best_r2_val:
        best_r2_val = -res.fun; best_w_reg = res.x.copy()

w_cls = np.exp(best_w_cls) / np.exp(best_w_cls).sum()
w_reg = np.exp(best_w_reg) / np.exp(best_w_reg).sum()
print('Cls: ' + ' | '.join(f'{k}={v:.3f}' for k,v in zip(ALL_CLS, w_cls)))
print('Reg: ' + ' | '.join(f'{k}={v:.3f}' for k,v in zip(MODEL_REG, w_reg)))

base_oof_c = sum(oof_cls[k]*w_cls[i] for i,k in enumerate(ALL_CLS))
base_te_c  = sum(tst_cls[k]*w_cls[i] for i,k in enumerate(ALL_CLS))
base_oof_r = sum(oof_reg[k]*w_reg[i] for i,k in enumerate(MODEL_REG))
base_te_r  = sum(tst_reg[k]*w_reg[i] for i,k in enumerate(MODEL_REG))

print(f'{elapsed()} Base cls acc: {accuracy_score(y_cls, base_oof_c.argmax(1)):.4f}')
print(f'{elapsed()} Base reg R2:  {r2_score(y_reg, np.clip(base_oof_r,4.99,35.99)):.4f}')

# level-2 stacking with LightGBM and CatBoost
print(f'\n{elapsed()} Building meta features...')
meta_tr = np.hstack([oof_cls[k] for k in ALL_CLS] +
                    [oof_reg[k].reshape(-1,1) for k in MODEL_REG])
meta_te = np.hstack([tst_cls[k] for k in ALL_CLS] +
                    [tst_reg[k].reshape(-1,1) for k in MODEL_REG])
print(f'{elapsed()} Meta dim: {meta_tr.shape[1]}')

# store separate outputs for both meta-model families
lgb_meta_cls_oof = np.zeros((n_tr, 5));  lgb_meta_cls_te = np.zeros((n_te, 5))
lgb_meta_ord_oof = np.zeros((n_tr, 5));  lgb_meta_ord_te = np.zeros((n_te, 5))
lgb_meta_reg_oof = np.zeros(n_tr);       lgb_meta_reg_te = np.zeros(n_te)
cat_meta_cls_oof = np.zeros((n_tr, 5));  cat_meta_cls_te = np.zeros((n_te, 5))
cat_meta_ord_oof = np.zeros((n_tr, 5));  cat_meta_ord_te = np.zeros((n_te, 5))
cat_meta_reg_oof = np.zeros(n_tr);       cat_meta_reg_te = np.zeros(n_te)

# fixed meta-model settings used for stacking
lgb_m_base = dict(n_estimators=3000, learning_rate=0.015, num_leaves=63,
                  max_depth=6, min_child_samples=30, subsample=0.7,
                  colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=2.0,
                  verbose=-1, n_jobs=-1)
cat_m_base = dict(iterations=2000, learning_rate=0.015, depth=7,
                  l2_leaf_reg=3.0, bagging_temperature=0.5, random_strength=1.0,
                  task_type='GPU', devices='0', verbose=0, early_stopping_rounds=100)

for ms in SEEDS:
    print(f'{elapsed()} Meta seed={ms}...')
    skf_m = StratifiedKFold(N_FOLDS, shuffle=True, random_state=ms)
    for ti, vi in skf_m.split(meta_tr, y_cls):

        # lightgbm multiclass meta-model
        m = lgb.LGBMClassifier(**lgb_m_base, objective='multiclass',
                                num_class=5, metric='multi_logloss', random_state=ms)
        m.fit(meta_tr[ti], y_cls[ti], eval_set=[(meta_tr[vi], y_cls[vi])],
              callbacks=[lgb.early_stopping(150, verbose=False)])
        lgb_meta_cls_oof[vi] += m.predict_proba(meta_tr[vi])
        lgb_meta_cls_te      += m.predict_proba(meta_te)

        # lightgbm ordinal meta-model
        bps, bpt = [], []
        for thr in range(1, 5):
            yb = (y_cls[ti] >= thr).astype(int)
            m2 = lgb.LGBMClassifier(**lgb_m_base, objective='binary',
                                     metric='binary_logloss', random_state=ms)
            m2.fit(meta_tr[ti], yb, eval_set=[(meta_tr[vi], (y_cls[vi]>=thr).astype(int))],
                   callbacks=[lgb.early_stopping(150, verbose=False)])
            bps.append(m2.predict_proba(meta_tr[vi])[:, 1])
            bpt.append(m2.predict_proba(meta_te)[:, 1])
        lgb_meta_ord_oof[vi] += ordinal_to_class_proba(bps)
        lgb_meta_ord_te      += ordinal_to_class_proba(bpt)

        # lightgbm regression meta
        m3 = lgb.LGBMRegressor(**lgb_m_base, objective='regression',
                                metric='rmse', random_state=ms)
        m3.fit(meta_tr[ti], y_reg[ti], eval_set=[(meta_tr[vi], y_reg[vi])],
               callbacks=[lgb.early_stopping(150, verbose=False)])
        lgb_meta_reg_oof[vi] += m3.predict(meta_tr[vi])
        lgb_meta_reg_te      += m3.predict(meta_te)

        # catboost multiclass meta
        mc4 = cb.CatBoostClassifier(**cat_m_base, loss_function='MultiClass',
                                     eval_metric='Accuracy', random_seed=ms)
        mc4.fit(meta_tr[ti], y_cls[ti], eval_set=(meta_tr[vi], y_cls[vi]), verbose=False)
        cat_meta_cls_oof[vi] += mc4.predict_proba(meta_tr[vi])
        cat_meta_cls_te      += mc4.predict_proba(meta_te)

        # catboost ordinal meta
        bps, bpt = [], []
        cat_bin_base = {k:v for k,v in cat_m_base.items()
                        if k not in ('loss_function','eval_metric')}
        for thr in range(1, 5):
            yb = (y_cls[ti] >= thr).astype(int)
            mb5 = cb.CatBoostClassifier(**cat_bin_base, loss_function='Logloss',
                                         eval_metric='AUC', random_seed=ms)
            mb5.fit(meta_tr[ti], yb, eval_set=(meta_tr[vi], (y_cls[vi]>=thr).astype(int)), verbose=False)
            bps.append(mb5.predict_proba(meta_tr[vi])[:, 1])
            bpt.append(mb5.predict_proba(meta_te)[:, 1])
        cat_meta_ord_oof[vi] += ordinal_to_class_proba(bps)
        cat_meta_ord_te      += ordinal_to_class_proba(bpt)

        # catboost regression meta
        mr6 = cb.CatBoostRegressor(**{k:v for k,v in cat_m_base.items()
                                      if k not in ('loss_function',)},
                                    loss_function='RMSE', random_seed=ms)
        mr6.fit(meta_tr[ti], y_reg[ti], eval_set=(meta_tr[vi], y_reg[vi]), verbose=False)
        cat_meta_reg_oof[vi] += mr6.predict(meta_tr[vi])
        cat_meta_reg_te      += mr6.predict(meta_te)

n_ms = len(SEEDS)
for arr in [lgb_meta_cls_oof, lgb_meta_ord_oof, cat_meta_cls_oof, cat_meta_ord_oof]:
    arr /= n_ms
for arr in [lgb_meta_cls_te, lgb_meta_ord_te, cat_meta_cls_te, cat_meta_ord_te]:
    arr /= n_ms * N_FOLDS
for arr in [lgb_meta_reg_oof, cat_meta_reg_oof]:
    arr /= n_ms
for arr in [lgb_meta_reg_te, cat_meta_reg_te]:
    arr /= n_ms * N_FOLDS

# eval meta signals
for name, arr in [('lgb_mc', lgb_meta_cls_oof), ('lgb_ord', lgb_meta_ord_oof),
                  ('cat_mc', cat_meta_cls_oof),  ('cat_ord', cat_meta_ord_oof)]:
    print(f'{elapsed()} Meta {name} acc: {accuracy_score(y_cls, arr.argmax(1)):.4f}')
for name, arr in [('lgb_reg', lgb_meta_reg_oof), ('cat_reg', cat_meta_reg_oof)]:
    print(f'{elapsed()} Meta {name} R2:  {r2_score(y_reg, np.clip(arr,4.99,35.99)):.4f}')

# best final blend
# classification: blend 5 signals with random-start Nelder-Mead
cls_signals_oof = {
    'base':     base_oof_c,
    'lgb_mc':   lgb_meta_cls_oof,
    'lgb_ord':  lgb_meta_ord_oof,
    'cat_mc':   cat_meta_cls_oof,
    'cat_ord':  cat_meta_ord_oof,
}
cls_signals_te = {
    'base':     base_te_c,
    'lgb_mc':   lgb_meta_cls_te,
    'lgb_ord':  lgb_meta_ord_te,
    'cat_mc':   cat_meta_cls_te,
    'cat_ord':  cat_meta_ord_te,
}
cls_keys = list(cls_signals_oof.keys())

def neg_blend_acc(w):
    ws = np.exp(w) / np.exp(w).sum()
    blend = sum(cls_signals_oof[k]*ws[i] for i,k in enumerate(cls_keys))
    return -accuracy_score(y_cls, blend.argmax(1))

print(f'\n{elapsed()} Searching final cls blend (random restarts)...')
rng = np.random.RandomState(SEED)
best_v, best_wc = -np.inf, None
for _ in range(20):
    w0 = rng.randn(len(cls_keys)) * 0.5
    res = minimize(neg_blend_acc, w0, method='Nelder-Mead',
                   options={'maxiter':5000, 'xatol':1e-6, 'fatol':1e-6})
    if -res.fun > best_v:
        best_v = -res.fun; best_wc = res.x.copy()

w_final_cls = np.exp(best_wc) / np.exp(best_wc).sum()
print('Final cls blend: ' + ' | '.join(f'{k}={v:.3f}' for k,v in zip(cls_keys,w_final_cls)))
final_oof_c = sum(cls_signals_oof[k]*w_final_cls[i] for i,k in enumerate(cls_keys))
final_te_c  = sum(cls_signals_te[k]*w_final_cls[i]  for i,k in enumerate(cls_keys))
final_cls_acc = accuracy_score(y_cls, final_oof_c.argmax(1))
print(f'{elapsed()} Final cls acc: {final_cls_acc:.4f}')

# blend regression signals
reg_signals_oof = {'base': base_oof_r, 'lgb': lgb_meta_reg_oof, 'cat': cat_meta_reg_oof}
reg_signals_te  = {'base': base_te_r,  'lgb': lgb_meta_reg_te,  'cat': cat_meta_reg_te}
reg_keys = list(reg_signals_oof.keys())

def neg_blend_r2(w):
    ws = np.exp(w) / np.exp(w).sum()
    blend = sum(reg_signals_oof[k]*ws[i] for i,k in enumerate(reg_keys))
    return -r2_score(y_reg, np.clip(blend, 4.99, 35.99))

best_rv, best_wr = -np.inf, None
for _ in range(20):
    w0 = rng.randn(len(reg_keys)) * 0.5
    res = minimize(neg_blend_r2, w0, method='Nelder-Mead',
                   options={'maxiter':5000, 'xatol':1e-6, 'fatol':1e-6})
    if -res.fun > best_rv:
        best_rv = -res.fun; best_wr = res.x.copy()

w_final_reg = np.exp(best_wr) / np.exp(best_wr).sum()
print('Final reg blend: ' + ' | '.join(f'{k}={v:.3f}' for k,v in zip(reg_keys,w_final_reg)))
final_oof_r = sum(reg_signals_oof[k]*w_final_reg[i] for i,k in enumerate(reg_keys))
final_te_r  = sum(reg_signals_te[k]*w_final_reg[i]  for i,k in enumerate(reg_keys))
final_oof_r = np.clip(final_oof_r, 4.99, 35.99)
final_te_r  = np.clip(final_te_r,  4.99, 35.99)

final_r2 = r2_score(y_reg, final_oof_r)
print(f'{elapsed()} Final reg R2:  {final_r2:.4f}')

# final out-of-fold scores
final_comb = 0.5 * final_cls_acc + 0.5 * final_r2
print('\n' + '='*58)
print('FINAL OOF SCORES')
print(f'Accuracy  : {final_cls_acc:.4f}')
print(f'R2        : {final_r2:.4f}')
print(f'Combined  : {final_comb:.4f}')
print(f'Runtime   : {(time.time()-T0)/60:.1f} min')
print('='*58)
print(classification_report(y_cls, final_oof_c.argmax(1),
      target_names=['VeryLow','Low','Moderate','High','VeryHigh']))

# diagnostics and plots
xgb_d = xgb.XGBClassifier(**{k:v for k,v in xgb_cls_cfg.items()
                               if k != 'early_stopping_rounds'}, random_state=SEED)
xgb_d.fit(Xtr, y_cls, verbose=False)
fi = pd.Series(xgb_d.feature_importances_, index=feat_names).sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(22, 10))
fi.head(30).sort_values().plot.barh(ax=axes[0], color='steelblue')
axes[0].set_title('Top 30 Features', fontsize=12)
cm = confusion_matrix(y_cls, final_oof_c.argmax(1))
ConfusionMatrixDisplay(cm, display_labels=[0,1,2,3,4]).plot(ax=axes[1], cmap='Blues')
axes[1].set_title(f'Confusion Matrix  Acc={final_cls_acc:.4f}', fontsize=12)
plt.tight_layout()
plt.savefig('analysis.png', dpi=100, bbox_inches='tight')
print('\nTop 20 features:')
print(fi.head(20).to_string())

# build submission file
final_cls_pred = final_te_c.argmax(1).astype(int)
final_reg_pred = np.round(final_te_r, 2)

sub = pd.DataFrame({
    'Id':           np.arange(len(final_cls_pred)),
    'RiskTier':     final_cls_pred,
    'InterestRate': final_reg_pred
})

assert len(sub) == 15000
assert sub['RiskTier'].between(0, 4).all()
assert sub['InterestRate'].between(4.99, 35.99).all()

sub.to_csv('submission.csv', index=False)
print('\nsubmission.csv saved!')
print(sub.head(10).to_string())
print('RiskTier dist:', sub['RiskTier'].value_counts().sort_index().to_dict())
print('Rate  mean=%.2f  std=%.2f' % (sub['InterestRate'].mean(), sub['InterestRate'].std()))
floor_n = (sub['InterestRate'] == 4.99).sum()
print(f'Floor-rate (4.99): {floor_n} ({floor_n/150:.1f}%)')
print(f'\n>>> Estimated Kaggle Combined Score: {final_comb:.4f} <<<')
print(f'Total runtime: {(time.time()-T0)/60:.1f} min')
