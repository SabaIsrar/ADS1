import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import errors as err

def read_data(file_paths, selected_country, start_year, end_year):
    dataframes_list = []

    for path in file_paths:
        file_name = path.split('/')[-1].split('.')[0]
        df = pd.read_csv(path, skiprows=4)
        df = df.rename(columns={'Country Name': 'Country'})
        df = df.set_index('Country')
        df_selected_country = df.loc[selected_country, str(start_year):str(end_year)].transpose().reset_index()
        df_selected_country = df_selected_country.rename(columns={'index': 'Year', selected_country: file_name})
        dataframes_list.append(df_selected_country)

    # Concatenate all DataFrames based on the 'Year' column
    result_df = pd.concat(dataframes_list, axis=1)

    # Replace null values with the mean of each column
    result_df = result_df.apply(lambda col: col.fillna(col.mean()))

    return result_df

def plot_data_fit(df, title, xlabel, ylabel, save_path):
    plt.figure()
    plt.plot(df["Year"], df["Data"], label="data")
    plt.plot(df["Year"], df["pop_exp"], label="fit")
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_path, dpi=300)
    plt.show()

def fit_and_predict(df, function, initial_guess, save_path):
    popt, pcovar = opt.curve_fit(function, df["Year"], df["Data"], p0=initial_guess, maxfev=10000)
    print("Fit parameters:", popt)
    
    df["pop_exp"] = function(df["Year"], *popt)
    
    plot_data_fit(df, f"Data Fit attempt for {df.columns[1]}", "Year", df.columns[1], save_path)

    years = np.linspace(1960, 2030)
    pop_exp_growth = function(years, *popt)
    sigma = err.error_prop(years, function, popt, pcovar)
    low = pop_exp_growth - sigma
    up = pop_exp_growth + sigma

    plt.figure()
    plt.title(f"{df.columns[1]} of {selected_country} in 2030")
    plt.plot(df["Year"], df["Data"], label="data")
    plt.plot(years, pop_exp_growth, label="fit")
    plt.fill_between(years, low, up, alpha=0.3, color="y", label="95% Confidence Interval")
    plt.legend(loc="upper left")
    plt.xlabel("Year")
    plt.ylabel(df.columns[1])
    plt.savefig(save_path, dpi=300)
    plt.show()

    pop_2030 = function(np.array([2030]), *popt)
    sigma_2030 = err.error_prop(np.array([2030]), function, popt, pcovar)
    print(f"{df.columns[1]} in")
    print("2030:", function(2030, *popt) / 1.0e6, "Mill.")

    print(f"{df.columns[1]} in")
    for year in range(2024, 2034):
        print(f"{df.columns[1]} in", year)
        print("2030:", function(year, *popt) / 1.0e6, "Mill.")


def exp_growth(t, scale, growth):
        """ Computes exponential function with scale and growth as free parameters """
   
        f = scale * np.exp(growth * t)
        return f



# Example usage
selected_country = "Pakistan"
start_year = 1970
end_year = 2021

file_paths = ['Foreign investment.csv', 'IMF credit.csv', 'Total reserves.csv']

for file_path in file_paths:
    df = read_data([file_path], selected_country, start_year, end_year)
    df["Year"] = pd.to_numeric(df["Year"], errors='coerce')
    df["Data"] = pd.to_numeric(df[df.columns[1]], errors='coerce')
    initial_guess = [1.0, 0.02]
    fit_and_predict(df, exp_growth, initial_guess, f"{file_path.split('.')[0]}_{selected_country}.png")
