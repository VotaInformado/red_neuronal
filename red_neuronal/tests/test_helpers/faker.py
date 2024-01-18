import random
import pandas as pd
from faker import Faker

# Project
from red_neuronal.utils.enums import VoteChoices


fake = Faker()

provinces = ["Buenos Aires", "CABA", "Córdoba", "Mendoza", "Santa Fe"]


def create_vote():
    return random.choice(VoteChoices.values)


def create_file_id():
    year = fake.year()
    file_id = fake.random_int(min=100, max=1250)
    file_chamber = random.choice(["D", "S"])
    return f"{year}-{file_id}-{file_chamber}"


def create_fake_df(df_columns: dict, n=100, as_dict: bool = True, **kwargs):
    column_names = list(df_columns.keys())
    column_types = list(df_columns.values())
    fake_data = {}
    for i in range(n):
        new_record = {}
        for column_name, column_type in zip(column_names, column_types):
            new_value = create_fake_value(column_type, n, **kwargs)
            new_record[column_name] = new_value
        fake_data[i] = new_record
    if as_dict:
        return fake_data
    return pd.DataFrame.from_dict(fake_data, orient="index")


def create_fake_value(column_type: str, n: int, **kwargs):
    if column_type == "parties":
        parties_list = [
            "1",  # we mock string ids since we receive string ids from recoleccion
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
        ]
        chosen_parties = random.choices(parties_list, k=3)
        return ";".join(chosen_parties)
    if column_type == "str":
        return fake.name()
    if column_type == "exp":
        return create_file_id()
    if column_type == "year":
        return fake.random_int(min=2015, max=2022)
    if column_type == "text":
        return fake.text()
    if column_type == "short_text":
        return fake.sentence()
    if column_type == "province":
        return random.choice(provinces)
    elif column_type == "email":
        return fake.email()
    elif column_type == "phone":
        return fake.phone_number()
    elif column_type == "vote":
        return create_vote()
    elif column_type == "date":
        if kwargs.get("dates_as_str", True):
            return fake.date_of_birth(minimum_age=18, maximum_age=80)
        return fake.date_of_birth(minimum_age=18, maximum_age=80).strftime("%Y-%m-%d")
    elif column_type == "int" or column_type == int:
        return fake.random_int(min=1, max=n)
    elif column_type == "unique-int":
        return fake.unique.random_int(min=1, max=n)
    else:
        raise ValueError(f"Column type {column_type} not supported")


def reset_fake_data():
    fake.unique.clear()