### Introduction

LLMs are highly fascinating, and there’s so much to learn about what goes on under its hood. Inspired by the great minds around me and in this generation, I’ve built this repository to showcase my learnings and experimentations with LLMs. I’m sure there are many others who have build their own which are more robust than mine. Even so, I hope to catch up.

Where possible, I will share all my resources that have helped me understood more about LLMs.

### What you’ll find

In this repository, you’ll find a LLM repository that is still a work-in-progress (WIP). 

I started out by following Andrej Karparthy’s tutorial on building a GPT tokenizer. This was where I built a gpt-2 decoder and simple bigram language model that are both causal language modeling. After which, I regurgitated my understand by rewriting it from scratch, and refactored it into a more organized folder structure that is more modular. Making it modular makes it easier for experimentation and adjustments.

In my second phase, I implemented a byte-pair encoder (BPE) tokenizer called ‘simplebpe’. As with the initial phase, this was learnt from Andrej Karparthy’s tutorial on building the GPT tokenizer. Simultaneously, I created 2 more new tokenizers that are both a ‘new’ version of the initial ‘character’ and ‘simplebpe’ tokenizer. These 2 ‘new’ versions are created to handle special tokens like ‘<pad>’, ‘<mask>’, etc. The ability to handle special tokens comes in usefully during masked language modeling. 

In my third phase, I implemented masked language modelling (MLM) with the GPT2 decoder language model. During this phase, I explored the key questions of 1) how is MLM implemented, and 2) how to handle batch of inputs with different lengths. Here, I had to rely on various resources to ensure my understanding and insertion of masked language modelling is correct. These sources are [5,6,7,8]. The implementation of MLM with characternew tokenizer gave reasonable accuracies. However, the use of simplebpenew generated poor results.

In my fourth phase, and as a follow up from the poor results of the MLM with simplebpenew, I made the tokenizer more optimal by factoring in the use of regex to first chunk up the input text to ensure that byte pairs do not overrun into the next words, before running the BPE algorithm. This is to address the likely suboptimal token pairs that Radford et al (2019) cautions of. This did lead to better loss values when the same parameters were applied. However, the BPE was still underperforming. So, I’ll have to deep dive on this issue.

### Next steps

- Deep dive on the poor results for MLM when using BPE tokenizer
- Adjust the ADAM optimizer function to one that uses weight decay for regularization.
- Create an FFN with SwiGLU (as used in Llama)

### **References**

[1] Karparthy, A. (2024). Let’s build the GPT tokenizer. Retrieved from https://youtu.be/zduSFxRajkE?si=QIVyM_tgpHQ3T5RH

[2] Karparthy, A. (2023). Let's build GPT: from scratch, in code, spelled out. Retrieved from https://youtu.be/kCc8FmEb1nY?si=cyIGm83Kb26eBsmD

[3] Warner, B. (2023). Creating a Transformer From Scratch - Part One: The Attention Mechanism. Retrieved from https://benjaminwarner.dev/2023/07/01/attention-mechanism

[4] Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners.

[5] Jurafsky, D., & Martin, J. H. Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition.

[6] Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Ma, C., Jernite, Y., Plu, J., Xu, C., Le Scao, T., Gugger, S., Drame, M., Lhoest, Q., & Rush, A. M. (2020). Transformers: State-of-the-Art Natural Language Processing [Conference paper]. 38–45. https://www.aclweb.org/anthology/2020.emnlp-demos.6

[7] Masked language modeling. (n.d.). https://huggingface.co/docs/transformers/main/en/tasks/masked_language_modeling

[8] Shreya, G. (2024, April 18). Comprehensive Guide to BERT. Analytics Vidhya. https://www.analyticsvidhya.com/blog/2022/11/comprehensive-guide-to-bert/