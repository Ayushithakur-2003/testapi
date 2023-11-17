from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

model_id = "facebook/bart-large-mnli"
classifier = pipeline("zero-shot-classification", model=model_id)

@app.get("/{feedback}")
async def root(feedback):
    product_aspects =  classifier(feedback,
      candidate_labels=["quality","price", "features", "shipping", "reliability", "durability", "performance", "user experience"],
                                 )
    aspect_res = ("aspect :", product_aspects["labels"][0], product_aspects["labels"][1])
    return {"result": aspect_res}