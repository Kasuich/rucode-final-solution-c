import pickle
import re

import ahocorasick
import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas()


class ReModel:
    columns_to_add = [
        "region",
        "municipality",
        "settlement",
        "location",
        "street",
        "house",
        "source",
    ]
    columns_re = ["municipality", "region", "settlement", "location", "street"]
    columns_to_rename = [
        "region",
        "region_type",
        "municipality",
        "municipality_type",
        "settlement",
        "settlement_type",
        "location",
        "location_type",
        "street",
        "street_type",
        "house",
        "source",
    ]

    def __init__(
        self,
        models: dict[str, ahocorasick.Automaton] = None,
        diff_poselok_poselenie: list = None,
    ) -> None:
        """
        Initializes an instance of the class.

        Args:
            models (dict[str, ahocorasick.Automaton], optional): A dictionary of models. Defaults to None.
            diff_poselok_poselenie (list, optional): A list of different poselok poselenie. Defaults to None.

        Returns:
            None
        """
        self.models = models
        self.diff_poselok_poselenie = diff_poselok_poselenie
        self.cut = np.vectorize(self.cut, otypes=[object])

    def split_address(self, df: pd.DataFrame):
        """
        Splits the address column in the given DataFrame by semicolon (;) and adds the split values as new columns.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the address column to be split.

        Returns:
        None
        """
        train_df = df
        columns_to_add = [x + "_sp" for x in self.columns_to_add]

        add_split = train_df["address"].str.split(";")
        add_split = np.array(add_split.tolist(), dtype=object)
        train_df[columns_to_add] = add_split

    def re_patterns(self, train_df: pd.DataFrame):
        """
        Generate regular expression patterns for each column in the given DataFrame.

        Parameters:
            train_df (pd.DataFrame): The DataFrame to generate patterns for.

        Returns:
            None
        """
        for column in self.columns_re:
            idx = ~train_df[column].isna()

            patterns = {}
            for value in train_df.loc[idx, column].unique():
                compiled = re.compile(r"^(.*?)" + re.escape(value))
                patterns[value] = compiled

            prefix = train_df.loc[idx, [column, column + "_sp"]].progress_apply(
                lambda x: re.search(patterns[x[column]], x[column + "_sp"]).group(1),
                axis=1,
            )
            train_df.loc[idx, column + "_re"] = prefix.str.lower().astype("category")

    def get_re2type(self, train_df: pd.DataFrame) -> dict[str, dict[str, str]]:
        """
        Generate a dictionary mapping regular expressions to types for each column in the given DataFrame.

        Parameters:
            train_df (pd.DataFrame): The DataFrame containing the training data.

        Returns:
            dict[str, dict[str, str]]: A dictionary mapping column names to dictionaries, where each dictionary
                                       maps a regular expression to a corresponding type.
        """
        re2type = {column: {} for column in self.columns_re}
        for column in self.columns_re:
            for row in (
                train_df[[f"{column}_re", f"{column}_type"]]
                .drop_duplicates()
                .dropna()
                .iterrows()
            ):
                re2type[column][row[1][f"{column}_re"]] = row[1][f"{column}_type"]
        re2type["location"].pop(" ")
        re2type["settlement"][" п.ст. "] = "поселок и(при) станция(и)"
        return re2type

    def get_models(
        self, re2type: dict[str, dict[str, str]]
    ) -> dict[str, ahocorasick.Automaton]:
        """
        Generate the models for each column in the dataframe.

        Args:
            re2type (dict[str, dict[str, str]]): A dictionary mapping column names to a dictionary of regular expression patterns and their corresponding types.

        Returns:
            dict[str, ahocorasick.Automaton]: A dictionary mapping column names to the corresponding Aho-Corasick automaton models.
        """
        models = {}
        for column in self.columns_re:
            automaton = ahocorasick.Automaton()
            for key, value in re2type[column].items():
                automaton.add_word(key, value)
            automaton.make_automaton()

            models[column] = automaton
        return models

    def get_diff_poselok_poselenie(self, train_df: pd.DataFrame) -> list:
        """
        Returns a list of settlement names that are categorized as "поселение" but not as "поселок" based on the input train_df DataFrame.

        Parameters:
            train_df (pd.DataFrame): The input DataFrame containing the settlement and settlement_type columns.

        Returns:
            list: A list of settlement names that belong to the "поселение" category but not to the "поселок" category.
        """
        settlement_poselenie = train_df["settlement"][
            train_df["settlement_type"] == "поселение"
        ].value_counts()
        settlement_poselenie = settlement_poselenie[settlement_poselenie > 0]
        settlement_poselenie = set(settlement_poselenie.index.tolist())

        settlement_poselok = train_df["settlement"][
            train_df["settlement_type"] == "поселок"
        ].value_counts()
        settlement_poselok = settlement_poselok[settlement_poselok > 0]
        settlement_poselok = set(settlement_poselok.index.tolist())

        diff_poselok_poselenie = {
            x
            for x in settlement_poselenie
            if x not in (settlement_poselok & settlement_poselenie)
        }
        return diff_poselok_poselenie

    def fit(self, train_df: pd.DataFrame) -> tuple[dict, list]:
        """
        Fits the model using the training data.

        Parameters:
            train_df (pd.DataFrame): The training data as a pandas DataFrame.

        Returns:
            tuple[dict, list]: A tuple containing the trained models and a list of different poselok poselenie.
        """
        self.diff_poselok_poselenie = self.get_diff_poselok_poselenie(train_df)

        self.split_address(train_df)
        self.re_patterns(train_df)
        re2type = self.get_re2type(train_df)
        self.models = self.get_models(re2type)

        return self.models, self.diff_poselok_poselenie

    def aho_predict(self, x: str, column: str) -> tuple:
        """
        Predicts the value of a given feature based on the input string and the specified column.

        Args:
            x (str): The input string to be used for prediction.
            column (str): The name of the column to predict the value for.

        Returns:
            tuple: A tuple containing the predicted value and its corresponding type. If the prediction is not available, returns (None, "").
        """
        if x.strip() != "":
            if (
                column == "location"
                and x.startswith(" территория ")
                and (x[12:14].isupper() or x[12:15] == "Гск")
            ):
                x = x[:12]
            outp = list(self.models[column].iter_long(x.lower()))

            if outp:
                if column == "settlement":
                    if (
                        outp[0][1] == "поселение"
                        and not x.startswith(" пос-е ")
                        and not x.startswith(" поселение ")
                    ):
                        return (outp[0][0], "поселок")
                elif (
                    column == "street"
                    and outp[0][1] == "территория снт"
                    and (
                        not x.startswith(" территория снт ")
                        or x == " территория снт Рассвет"
                    )
                ):
                    return (outp[0][0] - 4, "территория")

                if (
                    x.lower().startswith(" м. ")
                    and outp[0][1] == "местечко"
                    and column != "street"
                ):
                    return (outp[0][0], "массив")
                return outp[0]

        return (None, "")

    def cut(self, s: str, idx: int) -> str:
        """
        Cuts a string `s` from the index `idx + 1` and returns the resulting string.

        Parameters:
            s (str): The string to be cut.
            idx (int): The index from where the string should be cut.

        Returns:
            str: The resulting string after cutting.
        """
        if len(s.strip()) == 0 or idx is None:
            return ""
        try:
            return s[idx + 1 :]
        except Exception as ex:
            print((s, idx, ex))

    def parse_address(self, df):
        """
        Parse the address data in the given DataFrame.

        Args:
            df (DataFrame): The DataFrame containing the address data.

        Returns:
            None
        """
        train_df = df
        for column in self.columns_re:
            preds = train_df[f"{column}_sp"].progress_apply(
                lambda x: self.aho_predict(x, column)
            )
            preds = pd.DataFrame(preds.to_list(), columns=["prefix_size", "type_pred"])
            preds["prefix_size"] = preds["prefix_size"].astype("Int32")

            train_df[[f"{column}_pref_size", f"{column}_type_pred"]] = preds

            train_df[f"{column}_pred"] = self.cut(
                train_df[f"{column}_sp"], train_df[f"{column}_pref_size"]
            )

        train_df["house_pred"] = train_df["house_sp"].str.strip().astype(object)
        train_df["source_pred"] = train_df["source_sp"].str.strip().astype("category")

    def post_proccess(self, df: pd.DataFrame):
        """
        Post process the given DataFrame by making specific modifications.

        Parameters:
            df (pd.DataFrame): The DataFrame to be post processed.

        Returns:
            None
        """
        df.loc[
            (df["settlement_pred"].isin(self.diff_poselok_poselenie))
            & (df["settlement_type_pred"] == "поселок"),
            "settlement_type_pred",
        ] = "поселение"

        df.loc[
            df["location_type_pred"] == "некоммерческое партнерство",
            "location_type_pred",
        ] = "населенный пункт"

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates and returns the predicted values for the given input DataFrame.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing the data to be predicted.

        Returns:
            pd.DataFrame: The DataFrame with the predicted values for the specified columns.
        """
        self.split_address(df)
        self.parse_address(df)
        self.post_proccess(df)

        df.rename(
            columns={key + "_pred": key for key in self.columns_to_rename}, inplace=True
        )

        return df[self.columns_to_rename]


if __name__ == "__main__":
    train_df = pd.read_csv(
        "data/addresses-train.csv",
        sep=",",
        encoding="windows-1251",
        dtype={
            k: "category"
            for k in ReModel.columns_to_add
            + [
                "region_type",
                "municipality_type",
                "settlement_type",
                "location_type",
                "street_type",
            ]
        },
    )
    re_model = ReModel()
    re_model.fit(train_df)

    pickle.dump(re_model, open("re_model.pkl", mode="wb"))
