import pandas as pd
from langchain.schema import Document

input_excel_file = "~/Downloads/AdventureWorks Sales.xlsx"
df = pd.read_excel(input_excel_file, sheet_name="Sales_data", nrows=10)
dict_data_list = df.to_dict(orient="records")

docs_list = []

content_fields = [
    "SalesOrderLineKey",
    "ResellerKey",
    "OrderDateKey",
    "DueDateKey",
    "ShipDateKey",
    "Order Quantity",
]
metadata_fields = [
    "CustomerKey",
    "ProductKey",
    "OrderDateKey",
]
for data in dict_data_list:

    page_content = "\n".join([f"{field}: {data[field]}" for field in content_fields])
    metadata_map = {field: data[field] for field in metadata_fields}

    doc = Document(page_content=page_content, metadata=metadata_map)
    docs_list.append(doc)

print(docs_list[-1])
