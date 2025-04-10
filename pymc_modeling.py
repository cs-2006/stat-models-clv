from datetime import datetime
import pymc_marketing.clv as clv
import xarray
import numpy as np
import pandas as pd
from pathlib import Path
import polars as pl


def filter_purchases_purchases_per_month_pl(
    df_pl: pl.DataFrame, train_end: datetime.date, group_by_channel_id: bool = False
):
    # If sales_channel_id as covariate uncomment here
    # grouped_df = df_pl.group_by(['customer_id', 'date', 'sales_channel_id']).agg([
    df_pl = df_pl.with_columns(pl.col("date").cast(pl.Date))

    # Used for multi variate time series
    if group_by_channel_id:
        grouped_df = (
            df_pl.lazy()
            .group_by(["customer_id", "date", "sales_channel_id"])
            .agg(
                [
                    pl.col("article_id").explode().alias("article_ids"),
                    pl.col("price").sum().round(2).alias("total_price"),
                    pl.col("price").explode().alias("prices"),
                ]
            )
            .with_columns(pl.col("article_ids").list.len().alias("num_items"))
        )
    else:
        grouped_df = (
            df_pl.lazy()
            .group_by(["customer_id", "date"])
            .agg(
                [
                    pl.col("article_id").explode().alias("article_ids"),
                    pl.col("price").sum().round(2).alias("total_price"),
                    pl.col("sales_channel_id").explode().alias("sales_channel_ids"),
                    pl.col("price").explode().alias("prices"),
                ]
            )
            .with_columns(pl.col("article_ids").list.len().alias("num_items"))
        )

    # Only remove customers with extreme purchases in train period
    customers_summary = (
        df_pl.lazy()
        .filter(pl.col("date") < train_end)
        .group_by("customer_id")
        .agg(
            [
                pl.col("date").n_unique().alias("total_purchases"),
                pl.col("price").sum().round(2).alias("total_spent"),
                pl.col("article_id").flatten().alias("flattened_ids"),
            ]
        )
        .with_columns(pl.col("flattened_ids").list.len().alias("total_items"))
    )

    quantile = 0.99
    total_purchases_99, total_spending_99, total_items_99 = (
        customers_summary.select(
            [
                pl.col("total_purchases").quantile(quantile),
                pl.col("total_spent").quantile(quantile),
                pl.col("total_items").quantile(quantile),
            ]
        )
        .collect()
        .to_numpy()
        .flatten()
    )

    # Get customers to remove
    extreme_customers = customers_summary.filter(
        (pl.col("total_items") >= total_items_99)
        # | (pl.col("total_purchases") >= total_purchases_99)
        # | (pl.col("total_spent") >= total_spending_99)
    )

    # Remove customer who have a transaction with a lot of items
    only_train_dates = grouped_df.filter(pl.col("date") <= train_end)
    length_99 = (
        only_train_dates.group_by("customer_id")
        .agg(pl.col("num_items").max().alias("num_items"))
        .select(pl.col("num_items").quantile(quantile))
        .collect()
        .to_numpy()
        .flatten()[0]
    )
    long_tx_customers = (
        only_train_dates.filter(pl.col("num_items") > length_99)
        .select("customer_id")
        .unique()
    )

    extreme_customers = extreme_customers.select("customer_id").unique()
    # extreme_customers = (
    #     pl.concat([extreme_customers, long_tx_customers]).unique()
    # )
    extreme_customers = extreme_customers.collect()

    print(
        f"""
        Cutoff Values for {quantile*100}th Percentiles:
        -----------------------------------
        Total Purchases:       {total_purchases_99:.2f}
        Total Spent:           ${total_spending_99:.2f}
        Transaction Length:    {length_99:.0f} items
        *Total items bought:    {total_items_99:.0f} items

        -----------------------------------
        Removed Customers:     {len(extreme_customers):,}
        """
    )

    # filtered_user = grouped_df.join(extreme_customers, on="customer_id", how="anti")

    return grouped_df.collect(), extreme_customers


def load_data_rem_outlier_pl(
    data_path: Path, train_end: datetime.date, group_by_channel_id: bool = False
):
    file_path = data_path / "transactions_polars.parquet"
    df_pl = pl.read_parquet(file_path)

    df_pl = df_pl.with_columns(
        pl.col("t_dat").alias("date").cast(pl.Date), pl.col("article_id").cast(pl.Int32)
    )
    # rescale price according to https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations/discussion/310496
    df_pl = df_pl.with_columns(
        pl.col("price").mul(590).cast(pl.Float32).round(2).alias("price")
    )

    _, extreme_customers = filter_purchases_purchases_per_month_pl(
        df_pl, train_end=train_end, group_by_channel_id=group_by_channel_id
    )
    df_pl = df_pl.join(extreme_customers, on="customer_id", how="anti")
    return df_pl, extreme_customers


def generate_clv_data_pl(
    df: pl.DataFrame,
    label_threshold: datetime.date,
    pred_end: datetime.date,
):
    # Filter transactions between label_threshold and end_date for each period
    filtered_df = df.filter(
        (pl.col("date") >= label_threshold) & (pl.col("date") <= pred_end)
    )
    # Sum total_price for the filtered transactions by customer_id
    summed_period_df = filtered_df.group_by("customer_id").agg(
        pl.sum("price").round(2).alias(f"regression_label")
    )

    return summed_period_df


def get_data(data_path: Path, config: dict, subset=None):
    train_begin = datetime.strptime(config.get("train_begin"), "%Y-%m-%d")
    train_label_start = datetime.strptime(config.get("train_label_begin"), "%Y-%m-%d")
    train_end = datetime.strptime(config.get("train_end"), "%Y-%m-%d")
    test_begin = datetime.strptime(config.get("test_begin"), "%Y-%m-%d")
    test_label_start = datetime.strptime(config.get("test_label_begin"), "%Y-%m-%d")
    test_end = datetime.strptime(config.get("test_end"), "%Y-%m-%d")

    tx_df, _ = load_data_rem_outlier_pl(
        data_path,
        train_end=train_end,
        group_by_channel_id=False,
    )

    # Generate labels
    train_labels = generate_clv_data_pl(
        tx_df, label_threshold=train_label_start, pred_end=train_end
    ).to_pandas()
    test_labels = generate_clv_data_pl(
        tx_df, label_threshold=test_label_start, pred_end=test_end
    ).to_pandas()
    # Generate training data
    train_data = tx_df.filter(
        (pl.col("date") >= train_begin) & (pl.col("date") < train_label_start)
    ).to_pandas()
    # Generate test data
    test_data = tx_df.filter(
        (pl.col("date") >= test_begin) & (pl.col("date") < test_label_start)
    ).to_pandas()

    if subset is not None:
        # Subset data first subset rows of data
        train_data = train_data.iloc[:subset]
        test_data = test_data.iloc[:subset]

    train_data = clv.utils.rfm_summary(
        train_data,
        customer_id_col="customer_id",
        datetime_col="date",
        monetary_value_col="price",
        datetime_format="%Y-%m-%d",
    )
    test_data = clv.utils.rfm_summary(
        test_data,
        customer_id_col="customer_id",
        datetime_col="date",
        monetary_value_col="price",
        datetime_format="%Y-%m-%d",
    )

    return train_data, test_data, train_labels, test_labels


# Def prep data
def get_data_subsample(data_path: Path):
    train_labels = pd.read_csv(data_path / "train_label.csv")
    test_labels = pd.read_csv(data_path / "test_label.csv")

    train_data = pd.read_csv(data_path / "train_tx.csv")
    test_data = pd.read_csv(data_path / "test_tx.csv")

    train_data = clv.utils.rfm_summary(
        train_data,
        customer_id_col="customer_id",
        datetime_col="date",
        monetary_value_col="price",
        datetime_format="%Y-%m-%d",
    )
    test_data = clv.utils.rfm_summary(
        test_data,
        customer_id_col="customer_id",
        datetime_col="date",
        monetary_value_col="price",
        datetime_format="%Y-%m-%d",
    )

    return train_data, test_data, train_labels, test_labels


# Fetch models
def get_MBG_NDB_model(
    data, model_config: dict = None, sampler_config: dict = None
) -> clv.ModifiedBetaGeoModel:
    model = clv.ModifiedBetaGeoModel(
        data=data, model_config=model_config, sampler_config=sampler_config
    )
    model.fit(method="map")
    return model


def get_BG_NBD_model(
    data, model_config: dict = None, sampler_config: dict = None
) -> clv.BetaGeoModel:
    model = clv.BetaGeoModel(
        data=data, model_config=model_config, sampler_config=sampler_config
    )
    model.fit(method="map")
    return model


def get_Pareto_NBD_model(
    data, model_config: dict = None, sampler_config: dict = None
) -> clv.ParetoNBDModel:
    model = clv.ParetoNBDModel(
        data=data, model_config=model_config, sampler_config=sampler_config
    )
    model.fit(method="map")
    return model


def get_Gamma_Gamma_model(
    data, model_config: dict = None, sampler_config: dict = None
) -> clv.GammaGammaModel:
    model = clv.GammaGammaModel(
        data=data, model_config=model_config, sampler_config=sampler_config
    )
    model.fit(method="map")
    return model


# Function adapted from pymc package
def to_xarray(customer_id, *arrays, dim: str = "customer_id"):
    """Convert vector arrays to xarray with a common dim (default "customer_id")."""
    dims = (dim,)
    coords = {dim: np.asarray(customer_id)}

    res = tuple(
        xarray.DataArray(data=array, coords=coords, dims=dims) for array in arrays
    )

    return res[0] if len(arrays) == 1 else res


def customer_lifetime_value(
    transaction_model,
    monetary_model,
    data: pd.DataFrame,
    future_t: int = 180,
    discount_rate: float = 0.00,
    time_unit: str = "D",
) -> xarray.DataArray:
    """
    Compute customer lifetime value.

    Compute the average lifetime value for a group of one or more customers
    and apply a discount rate for net present value estimations.
    Note `future_t` is measured in months regardless of `time_unit` specified.

    Adapted from lifetimes package
    https://github.com/CamDavidsonPilon/lifetimes/blob/41e394923ad72b17b5da93e88cfabab43f51abe2/lifetimes/utils.py#L449

    Parameters
    ----------
    transaction_model : ~CLVModel
        Predictive model for future transactions. `BetaGeoModel` and `ParetoNBDModel` are currently supported.
    data : ~pandas.DataFrame
        DataFrame containing the following columns:

        * `customer_id`: Unique customer identifier
        * `frequency`: Number of repeat purchases observed for each customer
        * `recency`: Time between the first and the last purchase
        * `T`: Time between the first purchase and the end of the observation period
        * `future_spend`: Predicted monetary values for each customer
    future_t : int, optional
        The lifetime expected for the user in DAYS. Default: 180
    discount_rate : float, optional
        The monthly adjusted discount rate. Default: 0.00
    time_unit : string, optional
        Unit of time of the purchase history. Defaults to "D" for daily.
        Other options are "W" (weekly), "M" (monthly), and "H" (hourly).
        Example: If your dataset contains information about weekly purchases,
        you should use "W".

    Returns
    -------
    xarray
        DataArray containing estimated customer lifetime values

    """

    def _squeeze_dims(x: xarray.DataArray):
        """
        Squeeze dimensions for MAP-fitted model predictions.

        This utility is required for MAP-fitted model predictions to broadcast properly.

        Parameters
        ----------
        x : xarray.DataArray
            DataArray to squeeze dimensions for.

        Returns
        -------
        xarray.DataArray
            DataArray with squeezed dimensions.
        """
        dims_to_squeeze: tuple[str, ...] = ()
        if "chain" in x.dims and len(x.chain) == 1:
            dims_to_squeeze += ("chain",)
        if "draw" in x.dims and len(x.draw) == 1:
            dims_to_squeeze += ("draw",)
        x = x.squeeze(dims_to_squeeze)
        return x

    predicted_monetary_value = monetary_model.expected_customer_spend(data=data)
    data.loc[:, "future_spend"] = predicted_monetary_value.mean(
        ("chain", "draw")
    ).copy()

    if discount_rate == 0.0:
        # no discount rate: just compute a single time step from 0 to `time`
        steps = np.arange(future_t, future_t + 1)
    else:
        steps = np.arange(1, future_t + 1)

    # factor = {"W": 4.345, "M": 1.0, "D": 30, "H": 30 * 24}[time_unit]

    monetary_value = to_xarray(data["customer_id"], data["future_spend"])

    clv_array = xarray.DataArray(0.0)

    # initialize FOR loop with 0 purchases at future_t = 0
    prev_expected_purchases = 0

    for i in steps:  # * factor:
        # since the prediction of number of transactions is cumulative, we have to subtract off the previous periods
        new_expected_purchases = _squeeze_dims(
            transaction_model.expected_purchases(
                data=data,
                future_t=i,
            )
        )
        # if isinstance(transaction_model, clv.ModifiedBetaGeoModel) or isinstance(transaction_model, clv.ParetoNBDModel):
        # else:
        #     new_expected_purchases = _squeeze_dims(
        #         transaction_model.expected_num_purchases(
        #             data=data,
        #             future_t=i,
        #         )
        #     )

        expected_transactions = new_expected_purchases - prev_expected_purchases
        prev_expected_purchases = new_expected_purchases

        # sum up the CLV estimates of all the periods and apply discounted cash flow
        clv_array = clv_array + (monetary_value * expected_transactions) / (
            1 + discount_rate
        ) ** (i)

    # Add squeezed chain/draw dims
    if "draw" not in clv_array.dims:
        clv_array = clv_array.expand_dims({"draw": 1})
    if "chain" not in clv_array.dims:
        clv_array = clv_array.expand_dims({"chain": 1})

    clv_array = clv_array.transpose("chain", "draw", "customer_id")

    customer_ids = clv_array.coords["customer_id"].values

    # Extract the predictions
    predictions = clv_array.values[0, 0, :]

    return pd.DataFrame({"customer_id": customer_ids, "prediction": predictions})


def print_metrics(merged_preds: pd.DataFrame):
    # Calculate RMSE, MAE, and WMAPE for the three model types
    rmse_mbg = np.sqrt(
        np.mean((merged_preds["mbg_pred"] - merged_preds["regression_label"]) ** 2)
    )
    rmse_bg = np.sqrt(
        np.mean((merged_preds["bg_pred"] - merged_preds["regression_label"]) ** 2)
    )
    rmse_pn = np.sqrt(
        np.mean((merged_preds["pn_pred"] - merged_preds["regression_label"]) ** 2)
    )
    mae_mbg = np.mean(
        np.abs(merged_preds["mbg_pred"] - merged_preds["regression_label"])
    )
    mae_bg = np.mean(np.abs(merged_preds["bg_pred"] - merged_preds["regression_label"]))
    mae_pn = np.mean(np.abs(merged_preds["pn_pred"] - merged_preds["regression_label"]))
    wmape_mbg = np.sum(
        np.abs(merged_preds["mbg_pred"] - merged_preds["regression_label"])
    ) / np.sum(merged_preds["regression_label"])
    wmape_bg = np.sum(
        np.abs(merged_preds["bg_pred"] - merged_preds["regression_label"])
    ) / np.sum(merged_preds["regression_label"])
    wmape_pn = np.sum(
        np.abs(merged_preds["pn_pred"] - merged_preds["regression_label"])
    ) / np.sum(merged_preds["regression_label"])
    # Print results
    print(f"RMSE MBG: {rmse_mbg}")
    print(f"RMSE BG: {rmse_bg}")
    print(f"RMSE PN: {rmse_pn}")
    print(f"MAE MBG: {mae_mbg}")
    print(f"MAE BG: {mae_bg}")
    print(f"MAE PN: {mae_pn}")
    print(f"WMAPE MBG: {wmape_mbg}")
    print(f"WMAPE BG: {wmape_bg}")
    print(f"WMAPE PN: {wmape_pn}")


def get_clv_pred(transaction_model, monetary_model, data, column_name):
    preds = customer_lifetime_value(
        transaction_model=transaction_model,
        monetary_model=monetary_model,
        data=data,
        future_t=180,
        discount_rate=0.00,
        time_unit="D",
    )
    preds = preds.rename({"prediction": column_name}, axis=1)
    return preds


def get_pred_df(mbg_model, bg_model, pn_model, gg_model, data, labels):
    mbg_preds = get_clv_pred(
        transaction_model=mbg_model,
        monetary_model=gg_model,
        data=data,
        column_name="mbg_pred",
    )
    bg_preds = get_clv_pred(
        transaction_model=bg_model,
        monetary_model=gg_model,
        data=data,
        column_name="bg_pred",
    )
    pn_preds = get_clv_pred(
        transaction_model=pn_model,
        monetary_model=gg_model,
        data=data,
        column_name="pn_pred",
    )

    # Merge predictions
    merged_preds = pd.merge(mbg_preds, bg_preds, on="customer_id")
    merged_preds = pd.merge(merged_preds, pn_preds, on="customer_id")
    merged_preds = pd.merge(merged_preds, labels, on="customer_id", how="left")
    # Fill null values with 0
    merged_preds = merged_preds.fillna(0)
    return merged_preds


def main():
    config = {
        "train_begin": "2018-09-20",
        "train_label_begin": "2019-09-20",
        "train_end": "2020-03-17",
        "test_begin": "2019-03-19",
        "test_label_begin": "2020-03-18",
        "test_end": "2020-09-13",
    }
    # config = {
    #     "train_begin": "2018-09-20",
    #     "train_label_begin": "2020-03-18",
    #     "train_end": "2020-09-13",
    #     "test_begin": "2019-03-19",
    #     "test_label_begin": "2020-03-18",
    #     "test_end": "2020-09-13",
    # }


    # Load data
    sub_data_path = Path(
        "/Users/dschoess/Documents/PhD/Research/CLV/benchmarks/stat-models-clv/subsample/100000"
    )
    train_data, test_data, train_labels, test_labels = get_data_subsample(sub_data_path)

    # data_path = Path("/Users/dschoess/Documents/PhD/Research/CLV/code/clv/data")
    # train_data, test_data, train_labels, test_labels = get_data(
    #     data_path, config, subset=None
    # )

    return_train = train_data.query("frequency > 0")

    # Fit models
    print("Fitting models...")
    print("Fitting Gamma Gamma model...")
    # Need to use data of returning customers
    gg_model = get_Gamma_Gamma_model(return_train)

    print("Fitting Modified Beta Geo model...")
    mbg_model = get_MBG_NDB_model(train_data)

    print("Fitting Beta Geo model...")
    bg_model = get_BG_NBD_model(train_data)

    print("Fitting Pareto NBD model...")
    pn_model = get_Pareto_NBD_model(train_data)

    train_preds = get_pred_df(
        mbg_model=mbg_model,
        bg_model=bg_model,
        pn_model=pn_model,
        gg_model=gg_model,
        data=train_data,
        labels=train_labels,
    )
    test_preds = get_pred_df(
        mbg_model=mbg_model,
        bg_model=bg_model,
        pn_model=pn_model,
        gg_model=gg_model,
        data=test_data,
        labels=test_labels,
    )
    # Get evaluations
    print("Evaluating models...")
    print("Train predictions:")
    print_metrics(train_preds)
    print("Test predictions:")
    print_metrics(test_preds)

    prediction_path = Path("predictions")
    prediction_path.mkdir(parents=True, exist_ok=True)
    # Save results
    train_preds.to_csv(prediction_path/"predictions_train.csv.zip", index=False, compression="zip")
    test_preds.to_csv(prediction_path/"predictions_test.csv.zip", index=False, compression="zip")


if __name__ == "__main__":
    main()
