# Bridge_the_GAP
This work presents The Bridge the Gap Model (BGM) which aim to improve the performance of Retrieval-Augmented Generation (RAG) systems by bridging the preference gap between retrievers and language models (LLMs). In many current systems, retrievers and LLMs operate independently, causing a potential incompatibility between retrieved information and LLM requirements.

In this project, we introduce BGM, a novel model designed to select, reorder, and adapt the retrieved information, making it optimal for the LLM generation process. BGM is implemented as a sequence-to-sequence (seq2seq) model, which takes as input the retriever results and the user's query, returning an ordered and filtered sequence of relevant steps, adapted to the specific preferences of the LLM. Currently, the model is trained through supervised training, exploiting optimized sequences generated through a greedy search algorithm.

Experiments on public Question Answering and custom generation datasets show that the proposed approach significantly improves performance over traditional RAG methods, confirming the importance of fitting retrieved information to language patterns.
