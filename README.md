## Argo

<img src="https://github.com/user-attachments/assets/8eb6a4c1-bbcc-486d-92a5-6c24e6e2f0be" alt="argo" width="200"/>

> Argo, the ship that carried Jason and the Argonauts on their quest for the Golden Fleece

This is a playground to re-implement model architectures from industry/academic papers in Pytorch. The primary goal is educational and the target audience is for those who would like to start journey in machine learning & machine learning infra. The code implementation is optimized for readability and expandability, while not for the best of performance.

### Repo structure

- `data`: functions for dataset management, such as downloading public dataset, cache management, etc
- `embedding`: scripts used for generating embedding
- `feature`: functions for featuer engeering, right now primarily read data from benchmark and use Pandas to do certain feature engineer
- `get-started`: some userful notebooks to help you get faimilar with common techniques & concept in machine learning and recommendation system
- `model`: model code implementation
- `trainer`: simple wrapper around train/val/eval loop
- `server`: simple inference stack for recommendation system, including retrieval engine, feature server, model manager and inference engine
- `scripts`: some scripts used for setup the system, such as DB ingestion

### Prepare Step

#### Embedding Based Retrieval Setup

1. run `python movie_len_embedding.py` to generate the embeddings (only support the collabrative embedding)
2. run `python movie_len_index.py` to generate the FAISS index
3. run `python scripts/vector_db.py` to ingest embedding into DuckDB

### How to run locally

1. install the dependency `pip install -r requirements.txt`, `pip install -e .`
2. run `python main.py` to train the model with current env config.
3. run `python -m server/ebr_server.py` to start the grpc server for embedding based retrieval, it would listen on port 50051 by default; if you use DuckDB then this step could be skipped
4. run `python server/inference_engine.py` to start the inference server, it would listen on 8000 port
5. run `bash scripts/server_request.sh` to send a dummy request (there is one for DIN and one for TransAct as of now, will parameterized the request in the future)

### Papers

- :white_check_mark: [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1706.06978)
- :white_check_mark: [TransAct: Transformer-based Realtime User Action Model for Recommendation at Pinterest](https://arxiv.org/abs/2306.00248)
- [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://arxiv.org/pdf/2402.17152)
- :white_check_mark: [Better Generalization with Semantic IDs: A Case Study in Ranking for Recommendations](https://dl.acm.org/doi/abs/10.1145/3640457.3688190)
- :white_check_mark: [DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems](https://arxiv.org/pdf/2008.13535)
- [LONGER: Scaling Up Long Sequence Modeling in Industrial Recommenders](https://www.arxiv.org/pdf/2505.04421)
- [MTGR: Industrial-Scale Generative Recommendation Framework in Meituan](https://www.arxiv.org/abs/2505.18654)
- [TransAct V2: Lifelong User Action Sequence Modeling on Pinterest Recommendation](https://arxiv.org/pdf/2506.02267)
- [Learnable Item Tokenization for Generative Recommendation](https://arxiv.org/pdf/2405.07314)

### Road Map

Modeling

- [x] :white_check_mark: Deep Interest Network E2E training & inference example, MovieLen Small
- [x] :white_check_mark: TransAct training & inference example, MovieLen Large
- [x] :white_check_mark: MovieLen item embedding generation, collaborative filtering, two-towers, LLM ([QWen3-embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) is out)
- [ ] 🚧 HSTU training & inference example, MoiveLen Small
- [x] :white_check_mark: RQ-VAE
- [ ] Generative Retrieval via various strategies: NTP, MTP with semtanic ids, token represention with ANN

Data & Feature Engineering

- [x] :white_check_mark: Kuaishou Dataset: https://kuairand.com/
- [ ] Ray integration (DPP reader + trainer arch)
- [ ] Daft, Polars exploartion
      
Infra

- [x] :white_check_mark: Embedding Based Retrieval (EBR): DuckDB, FAISS
- [ ] Nearline item embedding update
- [ ] Feature store integration: FEAST
- [ ] Feature logging & training data generation pipeline
- [ ] Pytorch Lightening integration
- [ ] Reinforcement learning training infrastructure for recommendation task

GPU

- [ ] GPU training & inference enablement
- [ ] Integrate profiling, benchmarking, tuning, and monitoring for accelerator optimization
- [ ] Optimize representative models with auto-tuning, kernel fusion, quantization, dynamic batching, etc

### Reference
- [DuckDB](https://duckdb.org/docs/stable/clients/python/overview.html)
- [QWen3](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
