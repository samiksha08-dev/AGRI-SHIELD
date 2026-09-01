import pandas as pd

file_path = "agri_shield_buyer_transport_prototype.xlsx"

buyers = pd.read_excel(
    file_path,
    sheet_name="Buyer_Prototype"
)

print(buyers.head())

print("\nColumns in the dataset:")
print(buyers.columns.tolist())

'''crop = input("Enter crop: ")
variety = input("Enter variety: ")
quantity = float(input("Enter quantity (MT): "))
state = input("Enter state: ")
district = input("Enter district: ")

matching_buyers = buyers[
    (buyers["Crop"].str.lower() == crop.lower()) &
    (buyers["Variety"].str.lower() == variety.lower()) &
    (buyers["Required_Quantity_MT"] >= quantity)
]

print("\nMatching Buyers:")

if matching_buyers.empty:
    print("No suitable buyers found.")
else:
    result = matching_buyers[
        [
            "Buyer_Name",
            "Buyer_Type",
            "State",
            "District",
            "Crop",
            "Variety",
            "Required_Quantity_MT",
            "Indicative_Price_INR_per_MT"
        ]
    ]

    print(result.to_string(index=False))'''