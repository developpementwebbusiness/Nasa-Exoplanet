# -*- coding: utf-8 -*-
import pandas as pd
import random


def rearrange_columns_by_non_none_ratio(df):
    ratios = df.notna().sum() / len(df)
    sorted_columns = ratios.sort_values(ascending=False).index
    return df[sorted_columns]

def clean_dataframe_by_g_logic(df):
    def recursive_clean(df, col_idx):
        if col_idx < 0 or df.empty:
            return [df]

        col = df.columns[col_idx]
        total_values = df[col].notna().sum()

        rows_with_none = df[df[col].isna()]
        non_none_in_those_rows = rows_with_none.notna().sum(axis=1).sum()

        G = total_values - non_none_in_those_rows

        results = []
        if G > 0:
            # Remove rows with None in this column
            updated = df[df[col].notna()]
            results.extend(recursive_clean(updated, col_idx - 1))
        elif G < 0:
            # Remove the column
            updated = df.drop(columns=[col])
            results.extend(recursive_clean(updated, col_idx - 1))
        else:
            # G == 0: branch into two paths
            path1 = df[df[col].notna()]
            results.extend(recursive_clean(path1, col_idx - 1))

            path2 = df.drop(columns=[col])
            results.extend(recursive_clean(path2, col_idx - 1))

        return results

    return recursive_clean(df, len(df.columns) - 1)

def compute_retention_ratios(original_df, cleaned_versions):
    original_count = original_df.notna().sum().sum()
    ratios = []

    for i, cleaned_df in enumerate(cleaned_versions):
        cleaned_count = cleaned_df.notna().sum().sum()
        ratio = cleaned_count / original_count if original_count > 0 else 0
        ratios.append((i + 1, cleaned_count, original_count, ratio))
    return ratios

def generate_test_dataframe(rows=3000, cols=30, missing_rate=0.2):
    data = []
    for _ in range(rows):
        row = [random.randint(1, 100) if random.random() > missing_rate else None for _ in range(cols)]
        data.append(row)
    return pd.DataFrame(data)


def clean_array(L):
    #array=pd.DataFrame(L)
    rearrange_array=rearrange_columns_by_non_none_ratio(L)
    cleaned_arrays=clean_dataframe_by_g_logic(rearrange_array)
    return cleaned_arrays

def show_R(L):
    print(compute_retention_ratios(pd.DataFrame(L),clean_array(L)))
# array=pd.read_csv("C:\\Users\\maxim\\Downloads\\cumulative_2025.10.04_15.31.17.csv",skiprows=45)
# rearrange_array=rearrange_columns_by_non_none_ratio(array)
# cleaned_arrays=clean_dataframe_by_g_logic(rearrange_array)
 #print(array,"\n",rearrange_array,"\n",cleaned_arrays,"\n",compute_retention_ratios(array, cleaned_arrays))



# # Small example
# data = [
#     [1, None, 3],
#     [4, None, None],
#     [None, 8, 9]
# ]
# df = pd.DataFrame(data)
# rearranged_df = rearrange_columns_by_non_none_ratio(df)
# cleaned_versions = clean_dataframe_by_g_logic(rearranged_df)

# print("Rearranged DataFrame:")
# print(rearranged_df)

# print("\nCleaned versions:")
# for i, version in enumerate(cleaned_versions, 1):
#     print(f"\nVersion {i}:")
#     print(version)

# # Large test
# test_df = generate_test_dataframe()
# rearranged_test_df = rearrange_columns_by_non_none_ratio(test_df)
# cleaned_test_versions = clean_dataframe_by_g_logic(rearranged_test_df)

# print(f"\nGenerated {len(cleaned_test_versions)} cleaned versions from test DataFrame.")
# print(cleaned_test_versions)



#     return ratios
# # Generate test DataFrame
# test_df = generate_test_dataframe()

# # Rearrange and clean
# rearranged_test_df = rearrange_columns_by_non_none_ratio(test_df)
# cleaned_test_versions = clean_dataframe_by_g_logic(rearranged_test_df)

# # Compute retention ratios
# ratios = compute_retention_ratios(test_df, cleaned_test_versions)

# # Display results
# print("\nRetention Ratios:")
# for version, cleaned, original, ratio in ratios:
#     print(f"Version {version}: {cleaned} / {original} = {ratio:.2%}")