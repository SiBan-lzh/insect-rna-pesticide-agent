---
name: your_knowledge_base_name
description: > 
  A useful knowledge base when you need to search for content about "your_knowledge_base_name".
# Describe what this knowledge base contains — what kind of documents, which topics or species they cover, and when the LLM should use this KB.
chunk_strategy: general
#  Available strategies: general | parent_child | qa
#  general: Each natural segment is a chunk. Retrieval and context use the same chunk.
#  parent_child: Child chunks used for retrieval; parent chunk attached as context. Default: parent_marker="\n\n", child_marker="\n".
#  qa: Question part used for retrieval; answer part returned as context.  Default: question_col="Question", answer_col="Answer".
chunk_params: {}
#  Override defaults when needed.
#  parent_child: { parent_marker: "...", child_marker: "..." }
#  qa: { question_col: "...", answer_col: "..." }
recommended_search_mode: hybrid
# hybrid | semantic | keyword
recommended_semantic_weight: 0.4
# 0.0 = keyword only, 1.0 = semantic only
---
