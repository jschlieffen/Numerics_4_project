import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    return os, pd


@app.cell
def _(os, pd):
    df_lst: list[pd.DataFrame] = []
    for filename in os.listdir('../data'):
        if filename.startswith('COVIDcast'):
            dict_name: str = filename.split('.')[0].split('_', 1)[-1]
            df: pd.DataFrame = pd.read_csv(os.path.join('../data', filename),
            usecols=['date', 'value'], parse_dates=['date'])
            df.rename(columns={'date': 'Day', 'value': dict_name}, inplace=True)
            df_lst.append(df)
        else:
            dict_name: str = filename.split('.')[0]
            df = pd.read_csv(os.path.join('../data', filename), parse_dates=['Day'])
            df.rename(columns={df.columns[-1]: dict_name}, inplace=True)
            df_lst.append(df[['Day', dict_name]])
    return (df_lst,)


@app.cell
def _(df_lst: "list[pd.DataFrame]"):
    df_lst
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
