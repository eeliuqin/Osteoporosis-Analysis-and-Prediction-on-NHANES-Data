import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_TYPES = {
    "SEQN": "str",
    "RIDAGEYR": "float",
    "RIAGENDR": "category",
    "RIDRETH3": "category", 
    "BMXBMI": "float",
    "SMQ020": "category",
    "ALQ151": "category",
    "SLD010H": "float",
    "MCQ160A": "category",
    "MCQ160L": "category",
    "OSQ060": "category",
    "OSQ150": "category",
    "BPQ020": "category",    
}

COLUMN_NAMES_REPLACED = {
    "SEQN": "ID",
    "RIDAGEYR": "Age",
    "RIAGENDR": "Gender",
    "RIDRETH3": "Race", 
    "BMXBMI": "BMI",
    "SMQ020": "Smoking",
    "ALQ151": "Heavy Drinking",
    "SLD010H": "Sleep Hours",
    "MCQ160A": "Arthritis",
    "MCQ160L": "Liver Condition",
    "OSQ060": "Osteoporosis",
    "OSQ150": "Parent Osteoporosis",
    "BPQ020": "High blood Pressure", 
}

GENERAL_CODE_REPLACED = {
    1: "Yes",
    2: "No",
    7: "Refused",
    9: "Don't know"
}

DEMOGRAPHIC_REPLACED = {
    "RIAGENDR": {1: "Male", 2: "Female"},
    "RIDRETH3": {
        1: "Mexican American",
        2: "Other Hispanic",
        3: "Non-Hispanic White",
        4: "Non-Hispanic Black",
        6: "Non-Hispanic Asian",
        7: "Other Race - Including Multi-Racial",
    },
}

THRESHOLD_DICT = {
    'Age':  {
        0: "40-44",
        45: "45-49", 
        50: "50-54",
        55: "55-59",
        60: "60-64", 
        65: "65-69",
        70: "70-74",
        75: "75-79", 
        80: "80+",
    },
    'BMI': {
        0: "Underweight",
        18.5: "Healthy Weight", 
        25: "Overweight",
        30: "Obesity"
    },
    'Sleep Hours': {
        0: "4 Hours and Less",
        5: "5-6 Hours",
        7: "7-8 Hours",
        9: "9 Hours and More",
    }  
}



def import_data(file,
                folder1='data/2013-2014-NHANES',
                folder2='data/2017-2020-NHANES', 
                split=False):
    data1 = pd.read_sas(f"{folder1}/{file}")
    data2 = pd.read_sas(f"{folder2}/{file}")
    if not split:
        data = pd.concat([data1, data2])
        return data
    else:
        return data1, data2
    
def data_replace(data, general_replace=False, replace_dict=""):
    # replace values according to the replace_dict, 
    # which contains specific values for specific columns
    if replace_dict and not general_replace:
        for var, value_dict in replace_dict.items():
            for value, new_value in value_dict.items():
                if var in data.columns:
                    data[var] = data[var].replace(value, new_value) 
    # replace values of all columns, according to GENERAL_CODE_REPLACED
    else:
        for var in data.columns:
            for value, new_value in GENERAL_CODE_REPLACED.items():
                data[var] = data[var].replace(value, new_value)
    
    return data
        

def negative_data_replace(x):
    x = np.where(x.isin(['Yes']), x, 'No')
    
    return x
    
def change_data_types(data):
    for var, data_type in DATA_TYPES.items():
        if var in data.columns:
            data[var] = data[var].astype(data_type)
        
    return data

def rename_columns(data):
    data = data.rename(columns=COLUMN_NAMES_REPLACED)
    
    return data
      
def common_clean(data):
    """Common process for data cleaning 
    
    Args:
        data: the pandas dataframe
    Returns:
        a clean dataframe without missing values and duplicates
    """
    # fill in missing values
    data = data.fillna(0)
    # remove white spaces from field names
    data.columns = data.columns.str.strip()
    # remove duplicates
    data = data.drop_duplicates()  
    # changes data types
    data = change_data_types(data)
    # rename columns
    data = rename_columns(data)
    
    return data
  

def clean_demographic(data):
    """Method for cleaning the demographics data, including 
       replacing certain values and changing data types
    
    Args:
        data: the pandas dataframe of demographics
    Returns:
        a clean dataframe
    """
    df = data.copy()
    
    # replace values  
    df = data_replace(df, replace_dict=DEMOGRAPHIC_REPLACED)
    # fill in missing values an remove duplicates
    df = common_clean(df)
    
    return df

def clean_bmi(data):
    """Method for cleaning the BMI data, including 
       replacing certain values and changing data types
    
    Args:
        data: the pandas dataframe of demographics
    Returns:
        a clean dataframe
    """
    df = data.copy()
    # drop missing values
    df = df.dropna()
    df = common_clean(df)
    
    return df


def clean_sleep(data):
    df = data.copy()
    # only keep rows with valid sleep hours, 77 means Refused
    df = df.query("SLD010H < 24")
    df = common_clean(df)
    
    return df

def number_to_category(data, var_col, new_var_col):
    """Group by a specific variable based on pre-defined dictionary
    
    Args:
        data: the dataframe with var_col
        var_col: the numeric variable to be grouped
        new_var_col: the new variable with data type string
    Returns:
        a dataframe in which the numeric variable has been converted to strings
    """
    df = data.copy()
    if var_col in THRESHOLD_DICT:
        # find the corresponding dict
        threshold_dict = THRESHOLD_DICT[var_col]
        for threshold_value, new_value in threshold_dict.items():
            df.loc[df[var_col] >= threshold_value, new_var_col] = new_value 
     
    return df


def clean_variable(data, col_list):
    """Method for cleaning the target data, including 
       removing data with unsure values other than Yes/No
    
    Args:
        data: the pandas dataframe of the target
    Returns:
        a clean dataframe
    """
    
    df = data.copy()
    # only keep rows answered "Yes" or "No" in the target_col 
    for target_col in col_list:
        df = df.query(f"{target_col}==1 | {target_col}==2")
    # replace positive answers with "Yes" and negative answers with "No"
    df = data_replace(df, general_replace=True)
    df = common_clean(df)
    
    return df
  
def get_percentage(data, var_col, target_col):
    df = data.copy()     
    df = df.groupby([var_col, target_col]).size().unstack(fill_value=0).reset_index()
    # get percentage of each group
    df['Percentage'] = round(df["Yes"]/(df["Yes"]+df["No"])*100, 2)
    
    return df

def barplot_percentage(data, var_col, target_col, order='', title=''):
    """Grouping the dataset by var_col and target_col 
       and return the target's percentage in each age group
    
    Args:
        var_col: the variable with Yes/No values
        target_col: the target variable name
    Returns:
        a dataframe displays percentage of the target variable
    """
    
    # get percentage
    df = get_percentage(data, var_col, target_col)
    
    # barplot with percentage as annotation
    if order:
        ax = sns.barplot(data=df, x=var_col, y="Percentage", order=order)
    else:
        ax = sns.barplot(data=df, x=var_col, y="Percentage")
    for p in ax.patches:
        x = p.get_width()*0.5 + p.get_x()
        y = p.get_height()+0.1
        ax.annotate(p.get_height(), (x, y), ha='center', color="black", fontsize=10, weight="bold")
    ax.set_title(title, pad=20)
    
    return ax

def move_legend(ax, new_loc, **kws):
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()
    ax.legend(handles, labels, loc=new_loc, title=None, **kws)
    


def multi_hist(data, hue, var_list):
    """Plot histgrams or boxplots for multiple variables
    
    Args:
        data: the input pandas dataframe
        hue: the column in the data frame that should be used for colour encoding
        var_list: the list of variables
    """
    
    nrow = int(np.ceil(len(var_list)/2))
    ncol = 2
    i = 1
    for var in var_list:
        plt.subplot(nrow, ncol, int(i))
        plt.gca().set_title(f"{var}",
                    fontsize=14, weight='bold')
        # histplot for each categorical variable
        if data[var].dtype.name == 'category':
            ax = sns.histplot(
                data=data,
                y=var,
                discrete=True,
                stat='probability',
                hue=hue,
                multiple='dodge', 
                shrink=.8
            )
#             move_legend(ax, "center left", bbox_to_anchor=(1, 0.5))
            move_legend(ax, new_loc="lower right", fontsize=8)
        else:
            sns.boxplot(data=data, x=var, y=hue)
            
        # increase subplot number
        i += 1
        
    plt.tight_layout()
    
def countplot_by_category(data, category):
    df_counts = data[category].value_counts()
    ax = sns.countplot(data=data, y=category, order=df_counts.index)
    total = df_counts.sum()
    # add percentage annotation
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_width()/total)
        x = p.get_width()*0.5
        y = p.get_y() + p.get_height()/1.5
        ax.annotate(percentage, (x, y), ha='center', color="white", fontsize=8, weight="bold")


def multi_countplot(data, var_list):
    nrow = int(np.ceil(len(var_list)/2))
    ncol = 2
    i = 1
    for var in var_list:
        plt.subplot(nrow, ncol, int(i))
        plt.gca().set_title(f"{var}",
                    fontsize=14, weight='bold')
        if data[var].dtype.name == 'category':
            countplot_by_category(data, category=var) 
        else:
            sns.histplot(data=data, x=var)
        i += 1
    plt.tight_layout()
    
    
def plot_by_gender(data, gender, var_col, target_col, 
                   annotate_x=0, annotate_y=0, annotate_text='',
                   order='', x_tick_rotation=0):
    data_gender = data.query(f"Gender == '{gender}'")
             
    ax = barplot_percentage(data_gender, var_col, target_col, order, title=f"Prevalence of {target_col}, {gender}")
    ax.axhline(y=annotate_y, color='red', alpha=0.5, linestyle='dashed')
    ax.text(annotate_x, annotate_y*1.06, f"{gender} {target_col}: {annotate_text}%", fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=x_tick_rotation, horizontalalignment='right')
    
    return ax

def plot_both_gender(data, var_col, target_col, 
                     text_male, text_female, 
                     annotate_x_male, annotate_x_female,
                     annotate_y_male, annotate_y_female,
                     order='', x_tick_rotation=0
                    ):
    plt.subplot(121)
    ax = plot_by_gender(data, 'Male', var_col, target_col, 
                        annotate_x_male, annotate_y_male, 
                        text_male, order, x_tick_rotation)
    plt.subplot(122)
    ax = plot_by_gender(data, 'Female', var_col, target_col, 
                        annotate_x_female, annotate_y_female, 
                        text_female, order, x_tick_rotation)
    plt.tight_layout()


# group by var_col and target_col, for a specific gender
def gender_groupby(data, gender, var_col, target_col):
    df = (
            data.query(f"Gender=='{gender}'")
            .groupby([var_col, target_col])
            .size()
            .unstack()
            .transpose()
         )

    return df

def get_pr_ci(data, group1_name, group2_name):
    """Get 95% confidence interval for the prevalence ratio of group1/group2
        
    Args:
        data: the dataframe with numbers of outcome/no outcome
        separated by groups
        group1_name: the group name of the numerator of the prevalence ratio
        group2_name: the group name of the denominator of the prevalence ratio
    Returns:
        95% confidence interval and whether it's statistically significant 
        for the prevalence ratio of group1/group2
    """
    
    # number of outcome in group 1
    group1_outcome = data.loc['Yes', group1_name]
    # number of without outcome in group 1
    group1_no_outcome = data.loc['No', group1_name]
    # number of outcome in group 2
    group2_outcome = data.loc['Yes', group2_name]
    # number of without outcome in group 2
    group2_no_outcome = data.loc['No', group2_name]
    # PR of group 1, group 2
    prev1 = group1_outcome/(group1_outcome + group1_no_outcome)
    prev2 = group2_outcome/(group2_outcome + group2_no_outcome)
    # log of PR
    ln_pr = np.log(prev1/prev2)
    # standard error of ln(PR)
    se_ln_pr = np.sqrt(1/group1_outcome - 1/(group1_outcome+group1_no_outcome)
                      + 1/group2_outcome - 1/(group2_outcome+group2_no_outcome))
    lower_ci = round(np.exp(ln_pr - 1.96*se_ln_pr), 3)
    upper_ci = round(np.exp(ln_pr + 1.96*se_ln_pr), 3)
    # 95% confidence interval for the prevalence ratio
    ci_pr = (lower_ci, upper_ci)
    pr = round(prev1/prev2, 3)
    pr_ci_str = str(pr) + ' ' + str(ci_pr)
    significant = 'Yes' if upper_ci < 1 or lower_ci > 1 else 'No'
    
    # return prevalence ratio with 95% CI
    return pr_ci_str, significant

def get_prevalence_ratios(data, reference_group):
    """Get Prevalence Ratios (95% CI)
        
    Args:
        data: the dataframe contains reference group and other groups
        reference_group: the reference group name (the denominator of the prevalence ratio)
    Returns:
        a dataframe shows 95% CI for the prevalence ratio of 
        all the other groups vs. the reference group
    """
    
    group_list = [x for x in data.columns.values if x!= reference_group]
    predictors = [reference_group] + group_list
    ci_list = [get_pr_ci(data, x, reference_group) for x in group_list]
    significants = [c[1] for c in ci_list]
    confidence_intervals = ['ref'] + [c[0] for c in ci_list]
    df_pr =  pd.DataFrame({
        'Predictor': predictors, 
        'PR (95% CI)': confidence_intervals,
        'Statistically Significant': [''] + significants
    })
    
    return df_pr
    