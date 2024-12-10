import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

news = pd.read_csv(
    "news.tsv", 
    sep="\t",
    names=["itemId","category","subcategory","title","abstract","url","title_entities","abstract_entities"])
print(news.head(2))