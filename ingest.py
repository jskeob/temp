from colpali_engine.models import ColQwen2, ColQwen2Processor
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from datasets import load_dataset
from qdrant_client import QdrantClient, models
import torch
from pathlib import Path
from tqdm import tqdm
import uuid
import numpy as np
import random
from PIL import Image
import requests
import pypdfium2 as pdfium
from io import BytesIO


base_dir = Path(__file__).resolve().parent
urls = base_dir / "urls"
TOP_K=5


qdrant_client = QdrantClient(
    url="https://74b146ea-70e3-4569-a9c2-4eae7f79f665.eu-central-1-0.aws.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6MDhmZmZlZTEtZWNkMS00ZDBkLWJkMjUtNGQwMGMxNWIyMTYxIn0.koCDKOCU9P3IZPlLm9w4QsgEf0ICAmEu-jE4nQo0kSY",
)
#print(qdrant_client.get_collections())

colqwen_model = ColQwen2.from_pretrained(
        "vidore/colqwen2-v0.1",
        torch_dtype=torch.bfloat16,
        device_map="auto", # Use "cuda:0" for GPU, "cpu" for CPU, or "mps" for Apple Silicon
    ).eval()

colqwen_processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1")

qwen_vl_model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",#switch to flash_attention_2 when possible 
        device_map="auto",
)
qwen_vl_processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")


def init_qdrant_vis(collection_name):
    if(qdrant_client.collection_exists(collection_name)==False):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "original":models.VectorParams( #switch off HNSW
                        size=128,
                        distance=models.Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM
                        ),
                        hnsw_config=models.HnswConfigDiff(
                            m=0 #switching off HNSW
                        )
                    ),
                "mean_pooling_columns": models.VectorParams(
                        size=128,
                        distance=models.Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM
                        )
                    ),
                "mean_pooling_rows": models.VectorParams(
                        size=128,
                        distance=models.Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM
                        )
                    )
            }
        )

def get_patches(image_size, model_processor, model, model_name):
    return model_processor.get_n_patches(
        image_size, spatial_merge_size=model.spatial_merge_size)

def embed_and_mean_pool_batch(image_batch, model_processor, model, model_name):
    #embed
    with torch.no_grad():
        processed_images = model_processor.process_images(image_batch).to(model.device)
        image_embeddings = model(**processed_images)

    image_embeddings_batch = image_embeddings.cpu().float().numpy().tolist()

    #mean pooling
    pooled_by_rows_batch = []
    pooled_by_columns_batch = []


    for image_embedding, tokenized_image, image in zip(image_embeddings,
                                                       processed_images.input_ids,
                                                       image_batch):
        x_patches, y_patches = get_patches(image.size, model_processor, model, model_name)
        #print(f"{model_name} model divided this PDF page in {x_patches} rows and {y_patches} columns")

        image_tokens_mask = (tokenized_image == model_processor.image_token_id)

        image_tokens = image_embedding[image_tokens_mask].view(x_patches, y_patches, model.dim)
        pooled_by_rows = torch.mean(image_tokens, dim=0)
        pooled_by_columns = torch.mean(image_tokens, dim=1)

        image_token_idxs = torch.nonzero(image_tokens_mask.int(), as_tuple=False)
        first_image_token_idx = image_token_idxs[0].cpu().item()
        last_image_token_idx = image_token_idxs[-1].cpu().item()

        prefix_tokens = image_embedding[:first_image_token_idx]
        postfix_tokens = image_embedding[last_image_token_idx + 1:]

        #print(f"There are {len(prefix_tokens)} prefix tokens and {len(postfix_tokens)} in a {model_name} PDF page embedding")

        #adding back prefix and postfix special tokens
        pooled_by_rows = torch.cat((prefix_tokens, pooled_by_rows, postfix_tokens), dim=0).cpu().float().numpy().tolist()
        pooled_by_columns = torch.cat((prefix_tokens, pooled_by_columns, postfix_tokens), dim=0).cpu().float().numpy().tolist()

        pooled_by_rows_batch.append(pooled_by_rows)
        pooled_by_columns_batch.append(pooled_by_columns)

    return image_embeddings_batch, pooled_by_rows_batch, pooled_by_columns_batch

def upload_batch(original_batch, pooled_by_rows_batch, pooled_by_columns_batch, payload_batch, collection_name):
    try:
        qdrant_client.upload_collection(
            collection_name=collection_name,
            vectors={
                "mean_pooling_columns": pooled_by_columns_batch,
                "original": original_batch,
                "mean_pooling_rows": pooled_by_rows_batch
            },
            payload=payload_batch,
            ids=[str(uuid.uuid4()) for i in range(len(original_batch))]
        )
    except Exception as e:
        print(f"Error during upsert: {e}")



def upload_images(dataset,batch_size,collection_name,dataset_source):
    with tqdm(total=len(dataset), desc=f"Uploading progress of \"{dataset_source}\" dataset to \"{collection_name}\" collection") as pbar:
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            image_batch = [item["image"] for item in batch]
            current_batch_size = len(image_batch)
            try:
                original_batch, pooled_by_rows_batch, pooled_by_columns_batch = embed_and_mean_pool_batch(image_batch,
                                                                                            colqwen_processor,
                                                                                            colqwen_model,
                                                                                            "colQwen")
            except Exception as e:
                print(f"Error during embed: {e}")
                continue
            try:
                upload_batch(
                    np.asarray(original_batch, dtype=np.float32),
                    np.asarray(pooled_by_rows_batch, dtype=np.float32),
                    np.asarray(pooled_by_columns_batch, dtype=np.float32),
                    [
                        {
                            "source": dataset_source,
                            "page_number": batch[j]["page_number"]

                        }
                        for j in range(current_batch_size)
                    ],
                    collection_name
                )
            except Exception as e:
                print(f"Error during upsert: {e}")
                continue
            # Update the progress bar
            pbar.update(current_batch_size)
    print("Uploading complete!")

def process_urls():
    for file in urls.iterdir():
        print("Processing:", file.name)
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                url=line.strip()
                print(f"Processing: {url}")
                try:
                    # 1. Download PDF to a memory buffer
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    pdf_bytes = BytesIO(response.content)

                    # 2. Open PDF from memory
                    pdf = pdfium.PdfDocument(pdf_bytes)
                    
                    # 3. Convert all pages to images
                    pages_data = []
                    for i in range(len(pdf)):
                        page = pdf[i]
                        # scale=2 provides high-res images for graphs/text
                        image=(page.render(scale=2).to_pil().convert("RGB"))
                        pages_data.append({
                            "image": image,
                            "page_number": i+1
                        })

                    # 4. Use your batch embedding and upload logic here
                    # ingest_images_to_qdrant(images, source_url=url)
                    print(f"Successfully indexed {len(pages_data)} pages from {url}")
                    
                    upload_images(pages_data,1,"colqwen_tutorial",url)
                    print(f"Uploaded images to qdrant")                

                except Exception as e:
                    print(f"Failed to process {url}: {e}")



def batch_embed_query(query_batch, model_processor, model):
    with torch.no_grad():
        processed_queries = model_processor.process_queries(query_batch).to(model.device)
        query_embeddings_batch = model(**processed_queries)
    return query_embeddings_batch.cpu().float().numpy()

def reranking_search_batch(query_batch,
                           collection_name,
                           search_limit=TOP_K,
                           prefetch_limit=200):
    search_queries = [
      models.QueryRequest(
          query=query,
          prefetch=[
              models.Prefetch(
                  query=query,
                  limit=prefetch_limit,
                  using="mean_pooling_columns"
              ),
              models.Prefetch(
                  query=query,
                  limit=prefetch_limit,
                  using="mean_pooling_rows"
              ),
          ],
          limit=search_limit,
          with_payload=True,
          with_vector=False,
          using="original"
      ) for query in query_batch
    ]
    return qdrant_client.query_batch_points(
        collection_name=collection_name,
        requests=search_queries
    )

def get_image_for_url(url, page_number):
    try:
        response = requests.get(url, timeout=10)
        pdf_bytes = BytesIO(response.content)
        
        pdf = pdfium.PdfDocument(pdf_bytes)
        page = pdf[page_number - 1] # pypdfium uporablja 0-indeksiranje
        
        # Pretvoriš v PIL sliko, ki jo Qwen3-VL razume
        print(f"  - Dodana stran {page_number} iz {url}")
        return page.render(scale=2).to_pil().convert("RGB")
        
    except Exception as e:
        print(f"  - Napaka pri nalaganju strani {url}, page number:{page_number}: {e}")
        empty_img = Image.new('RGB', (1024, 1024), color='white')
        return empty_img


def answer_question(retreived,question):
    #za preveriti
    results = retreived[0].points
    content = []
    for hit in results:
        img = get_image_for_url(hit.payload["source"], hit.payload["page_number"])
        content.append({"type": "image", "image": img})
    
    prompt_text = (
        f"Si strokovni asistent za slovensko zakonodajo in komunikacije za AKOS (Agencija za komunikacijska omrežja in storitve Republike Slovenije). "
        f"Na podlagi priloženih slikovnih in textovnih dokumentov odgovori na vprašanje: {question}. "
        f"Odgovori izključno v slovenščini. Če iz podatkov ni možno odgovoriti na vprašanje, to povej."
    )
    content.append({"type": "text", "text": prompt_text})

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    #primer iz dokumentacije
    # # Preparation for inference
    # inputs = qwen_vl_processor.apply_chat_template(
    #     messages,
    #     tokenize=True,
    #     add_generation_prompt=True,
    #     return_dict=True,
    #     return_tensors="pt"
    # )
    # inputs = inputs.to(qwen_vl_model.device)

    # # Inference: Generation of the output
    # generated_ids = qwen_vl_model.generate(**inputs, max_new_tokens=1024)
    # generated_ids_trimmed = [
    #     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    # ]
    # output_text = qwen_vl_processor.batch_decode(
    #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    # )
    #gemini koda
    text = qwen_vl_processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    # 2. Ročni izvleček slik iz messages
    # To zagotovi, da procesor dejansko vidi PIL objekte
    images = [item["image"] for m in messages for item in m["content"] if item["type"] == "image"]

    # 3. Priprava končnih vhodov (tukaj se zgodi čarovnija)
    inputs = qwen_vl_processor(
        text=[text],
        images=images,
        padding=True,
        return_tensors="pt",
    ).to(qwen_vl_model.device)

    # 4. Generiranje
    generated_ids = qwen_vl_model.generate(**inputs, max_new_tokens=1024)

    # 5. Rezanje in dekodiranje
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = qwen_vl_processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )

    return output_text[0]


def main():        
    while True:
        try:
            question = input("Uporabnik: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in {"odnehaj", "koncaj", "q"}:
            print("Adijo!")
            break
        colqwen_query = batch_embed_query([question], colqwen_processor, colqwen_model)
        #print(f"ColQwen embedded query \"{query}\" with {len(colqwen_query[0])} multivectors of dim {len(colqwen_query[0][0])}")
        answer_colqwen = reranking_search_batch(colqwen_query, "colqwen_tutorial")
        
        print("Generiram odgovor na podlagi najdenih slikovnih in tekstovnih dokumentov... ... ...")
        
        final_answer = answer_question(answer_colqwen, question)
        print("\nODGOVOR (Qwen3-VL):")
        print(final_answer)
        print("-" * 50)

if __name__ == "__main__":
    init_qdrant_vis("vis_embed")
    process_urls()
    print(qdrant_client.get_collections())
    main()

#TO-DO
#narediti embedding proces za textovno dokumente torej
#baza urljev da pregledujemo ce je dokument ze notri 
#automatski update url-jev iz akosa
