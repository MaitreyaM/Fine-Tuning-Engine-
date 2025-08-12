---
description: >-
  Train your own model with Unsloth, an open-source framework for LLM
  fine-tuning and reinforcement learning.
---

# Unsloth Docs

At [Unsloth](https://app.gitbook.com/o/HpyELzcNe0topgVLGCZY/s/xhOjnexMCB3dmuQFQ2Zq/), our mission is to make AI as accurate and accessible as possible. Train, run, evaluate and save gpt-oss, Llama, DeepSeek, TTS, Qwen, Mistral, Gemma LLMs 2x faster with 70% less VRAM.

Our docs will guide you through running & training your own model locally.

<a href="beginner-start-here" class="button primary">Get started</a> <a href="https://github.com/unslothai/unsloth" class="button secondary">Our GitHub</a>

<table data-card-size="large" data-view="cards"><thead><tr><th></th><th></th><th data-hidden data-card-cover data-type="files"></th><th data-hidden data-card-target data-type="content-ref"></th></tr></thead><tbody><tr><td><strong>gpt-oss</strong></td><td>Run OpenAI's new open-source models with Unsloth's bug fixes!</td><td><a href="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FX0pJKFv8zDMf4TJomAts%2Fgpt-oss%20image.png?alt=media&#x26;token=60c73c0d-cf83-4269-9619-f4b71e25767a">gpt-oss image.png</a></td><td><a href="../basics/gpt-oss-how-to-run-and-fine-tune">gpt-oss-how-to-run-and-fine-tune</a></td></tr><tr><td><strong>Finetune gpt-oss</strong></td><td>Train gpt-oss locally or for free using our notebooks.</td><td><a href="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FdUKxTDoQUFZPpOixP1Cx%2Fsloth%20with%20comp.png?alt=media&#x26;token=16fbc4a3-3d03-4e6c-bc74-75cf1121c797">sloth with comp.png</a></td><td><a href="../../basics/gpt-oss-how-to-run-and-fine-tune#fine-tuning-gpt-oss-with-unsloth">#fine-tuning-gpt-oss-with-unsloth</a></td></tr></tbody></table>

{% columns %}
{% column %}
{% content-ref url="fine-tuning-llms-guide" %}
[fine-tuning-llms-guide](fine-tuning-llms-guide)
{% endcontent-ref %}

{% content-ref url="unsloth-notebooks" %}
[unsloth-notebooks](unsloth-notebooks)
{% endcontent-ref %}


{% endcolumn %}

{% column %}
{% content-ref url="all-our-models" %}
[all-our-models](all-our-models)
{% endcontent-ref %}

{% content-ref url="../basics/tutorials-how-to-fine-tune-and-run-llms" %}
[tutorials-how-to-fine-tune-and-run-llms](../basics/tutorials-how-to-fine-tune-and-run-llms)
{% endcontent-ref %}
{% endcolumn %}
{% endcolumns %}

<table data-view="cards"><thead><tr><th></th><th></th><th data-hidden data-card-cover data-type="files"></th><th data-hidden data-card-target data-type="content-ref"></th></tr></thead><tbody><tr><td><strong>Qwen3-2507</strong></td><td>Run the new SOTA Thinking &#x26; Instruct LLMs.</td><td><a href="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FEj2zfXu3PPd39PvAmQtx%2Fqwen3-2507.png?alt=media&#x26;token=c070db7b-bfe9-4a7f-9e75-bbd0b0a01a4d">qwen3-2507.png</a></td><td><a href="../basics/qwen3-how-to-run-and-fine-tune/qwen3-2507">qwen3-2507</a></td></tr><tr><td><strong>Qwen3-Coder</strong></td><td>Run Qwen's coding and agentic models.</td><td><a href="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FeDz30Gy6kQ8zzdMaxr5m%2Fqwen3-coder%201920.png?alt=media&#x26;token=efad8f53-6d06-48bd-98e6-96bde543702d">qwen3-coder 1920.png</a></td><td><a href="../basics/qwen3-coder-how-to-run-locally">qwen3-coder-how-to-run-locally</a></td></tr><tr><td><strong>Gemma 3n</strong></td><td>Fine-tune &#x26; run Google's new multimodal models.</td><td><a href="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FBszehKqh4ex9879rI5jv%2FGemma%203%20text%20only.png?alt=media&#x26;token=b66212ab-409b-4603-80fa-337bea439531">Gemma 3 text only.png</a></td><td><a href="../basics/gemma-3n-how-to-run-and-fine-tune">gemma-3n-how-to-run-and-fine-tune</a></td></tr></tbody></table>

### 🦥 Why Unsloth?

* Unsloth simplifies model training locally and on platforms like Google Colab and Kaggle. Our streamlined workflow handles everything from model loading and quantization to training, evaluation, saving, exporting, and integration with inference engines like Ollama, llama.cpp and vLLM.
* The **key advantage** of Unsloth is our active role in _**fixing critical bugs**_ in major models. We've collaborated directly with teams behind [Qwen3](https://www.reddit.com/r/LocalLLaMA/comments/1kaodxu/qwen3_unsloth_dynamic_ggufs_128k_context_bug_fixes/), [Meta (Llama 4)](https://github.com/ggml-org/llama.cpp/pull/12889), [Mistral (Devstral)](https://app.gitbook.com/o/HpyELzcNe0topgVLGCZY/s/xhOjnexMCB3dmuQFQ2Zq/~/changes/618/basics/tutorials-how-to-fine-tune-and-run-llms/devstral-how-to-run-and-fine-tune), [Google (Gemma 1–3)](https://news.ycombinator.com/item?id=39671146) and [Microsoft (Phi-3/4)](https://simonwillison.net/2025/Jan/11/phi-4-bug-fixes), contributing essential fixes that significantly boost accuracy.
* Unsloth is highly customizable, allowing modifications in chat templates, dataset formatting and more. We provide support and notebooks for various methods, including [vision](../basics/vision-fine-tuning), [text-to-speech (TTS)](../basics/text-to-speech-tts-fine-tuning), BERT, [reinforcement learning (RL)](../basics/reinforcement-learning-rl-guide), and all transformer-based models.

### ⭐ Key Features

* Supports **full-finetuning**, pretraining, 4-bit, 16-bit and **8-bit** training.
* [MultiGPU](../basics/multi-gpu-training-with-unsloth) is in the works and soon to come!
* Supports **all transformer-style models** including [TTS, STT](https://docs.unsloth.ai/basics/text-to-speech-tts-fine-tuning), multimodal, diffusion, [BERT](https://docs.unsloth.ai/get-started/unsloth-notebooks#other-important-notebooks) and more!
* All kernels written in [OpenAI's Triton](https://openai.com/index/triton/) language. **Manual backprop engine**.
* **0% loss in accuracy** - no approximation methods - all exact.
* Unsloth Supports **Linux, Windows,** Google Colab, Kaggle **NVIDIA** and soon **AMD** & **Intel setups**. Most use Unsloth through Colab which provides a free GPU to train with. See:

{% content-ref url="beginner-start-here/unsloth-requirements" %}
[unsloth-requirements](beginner-start-here/unsloth-requirements)
{% endcontent-ref %}

### Quickstart

**Install locally with pip (recommended)** for Linux devices:

```
pip install unsloth
```

For Windows install instructions, see [here](installing-+-updating/windows-installation).

{% content-ref url="installing-+-updating" %}
[installing-+-updating](installing-+-updating)
{% endcontent-ref %}

### What is Fine-tuning and RL? Why?

**Fine-tuning** an LLM customizes its behavior, enhances domain knowledge, and optimizes performance for specific tasks. By fine-tuning a pre-trained model (e.g. Llama-3.1-8B) on a dataset, you can:

* **Update Knowledge**: Introduce new domain-specific information.
* **Customize Behavior**: Adjust the model’s tone, personality, or response style.
* **Optimize for Tasks**: Improve accuracy and relevance for specific use cases.

[**Reinforcement Learning (RL)**](../basics/reinforcement-learning-rl-guide) is where an "agent" learns to make decisions by interacting with an environment and receiving **feedback** in the form of **rewards** or **penalties**.

* **Action:** What the model generates (e.g., a sentence).
* **Reward:** A signal indicating how good or bad the model's action was (e.g., did the response follow instructions? was it helpful?).
* **Environment:** The scenario or task the model is working on (e.g., answering a user’s question).

**Example usecases of fine-tuning or RL:**

* Train LLM to predict if a headline impacts a company positively or negatively.
* Use historical customer interactions for more accurate and custom responses.
* Train LLM on legal texts for contract analysis, case law research, and compliance.

You can think of a fine-tuned model as a specialized agent designed to do specific tasks more effectively and efficiently. **Fine-tuning can replicate all of RAG's capabilities**, but not vice versa.&#x20;

{% content-ref url="beginner-start-here/faq-+-is-fine-tuning-right-for-me" %}
[faq-+-is-fine-tuning-right-for-me](beginner-start-here/faq-+-is-fine-tuning-right-for-me)
{% endcontent-ref %}

{% content-ref url="../basics/reinforcement-learning-rl-guide" %}
[reinforcement-learning-rl-guide](../basics/reinforcement-learning-rl-guide)
{% endcontent-ref %}

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FLrqITvuoKyiMl8mqfu5B%2Flarge%20sloth%20wave.png?alt=media&#x26;token=3077792b-90ff-459d-aa52-57abcf219adf" alt="" width="188"><figcaption></figcaption></figure>


---
description: Here are Unsloth's requirements including system and GPU VRAM requirements.
---

# Unsloth Requirements

## System Requirements

* **Operating System**: Works on Linux and Windows.
* Supports NVIDIA GPUs since 2018+ including [Blackwell RTX 50](../../basics/training-llms-with-blackwell-rtx-50-series-and-unsloth) series. Minimum CUDA Capability 7.0 (V100, T4, Titan V, RTX 20, 30, 40, A100, H100, L40 etc) [Check your GPU!](https://developer.nvidia.com/cuda-gpus) GTX 1070, 1080 works, but is slow.
* Unsloth is soon going to support [AMD](https://github.com/unslothai/unsloth/pull/2520) and [Intel](https://github.com/unslothai/unsloth/pull/2621) GPUs! Apple/Silicon/MLX is in the works.
* If you have different versions of torch, transformers etc., `pip install unsloth` will automatically install all the latest versions of those libraries so you don't need to worry about version compatibility.
* Your device must have `xformers`, `torch`, `BitsandBytes` and `triton` support.

## Fine-tuning VRAM requirements:

How much GPU memory do I need for LLM fine-tuning using Unsloth?

{% hint style="info" %}
A common issue when you OOM or run out of memory is because you set your batch size too high. Set it to 1, 2, or 3 to use less VRAM.

**For context length benchmarks, see** [**here**](../../../basics/unsloth-benchmarks#context-length-benchmarks)**.**
{% endhint %}

Check this table for VRAM requirements sorted by model parameters and fine-tuning method. QLoRA uses 4-bit, LoRA uses 16-bit. Keep in mind that sometimes more VRAM is required depending on the model so these numbers are the absolute minimum:

| Model parameters | QLoRA (4-bit) VRAM | LoRA (16-bit) VRAM |
| ---------------- | ------------------ | ------------------ |
| 3B               | 3.5 GB             | 8 GB               |
| 7B               | 5 GB               | 19 GB              |
| 8B               | 6 GB               | 22 GB              |
| 9B               | 6.5 GB             | 24 GB              |
| 11B              | 7.5 GB             | 29 GB              |
| 14B              | 8.5 GB             | 33 GB              |
| 27B              | 22GB               | 64GB               |
| 32B              | 26 GB              | 76 GB              |
| 40B              | 30GB               | 96GB               |
| 70B              | 41 GB              | 164 GB             |
| 81B              | 48GB               | 192GB              |
| 90B              | 53GB               | 212GB              |
| 405B             | 237 GB             | 950 GB             |


---
description: >-
  If you're stuck on if fine-tuning is right for you, see here! Learn about
  fine-tuning misconceptions, how it compared to RAG and more:
---

# FAQ + Is Fine-tuning Right For Me?

## Understanding Fine-Tuning

Fine-tuning an LLM customizes its behavior, deepens its domain expertise, and optimizes its performance for specific tasks. By refining a pre-trained model (e.g. _Llama-3.1-8B_) with specialized data, you can:

* **Update Knowledge** – Introduce new, domain-specific information that the base model didn’t originally include.
* **Customize Behavior** – Adjust the model’s tone, personality, or response style to fit specific needs or a brand voice.
* **Optimize for Tasks** – Improve accuracy and relevance on particular tasks or queries your use-case requires.

Think of fine-tuning as creating a specialized expert out of a generalist model. Some debate whether to use Retrieval-Augmented Generation (RAG) instead of fine-tuning, but fine-tuning can incorporate knowledge and behaviors directly into the model in ways RAG cannot. In practice, combining both approaches yields the best results - leading to greater accuracy, better usability, and fewer hallucinations.

### Real-World Applications of Fine-Tuning

Fine-tuning can be applied across various domains and needs. Here are a few practical examples of how it makes a difference:

* **Sentiment Analysis for Finance** – Train an LLM to determine if a news headline impacts a company positively or negatively, tailoring its understanding to financial context.
* **Customer Support Chatbots** – Fine-tune on past customer interactions to provide more accurate and personalized responses in a company’s style and terminology.
* **Legal Document Assistance** – Fine-tune on legal texts (contracts, case law, regulations) for tasks like contract analysis, case law research, or compliance support, ensuring the model uses precise legal language.

## The Benefits of Fine-Tuning

Fine-tuning offers several notable benefits beyond what a base model or a purely retrieval-based system can provide:

#### Fine-Tuning vs. RAG: What’s the Difference?

Fine-tuning can do mostly everything RAG can - but not the other way around. During training, fine-tuning embeds external knowledge directly into the model. This allows the model to handle niche queries, summarize documents, and maintain context without relying on an outside retrieval system. That’s not to say RAG lacks advantages as it is excels at accessing up-to-date information from external databases. It is in fact possible to retrieve fresh data with fine-tuning as well, however it is better to combine RAG with fine-tuning for efficiency.

#### Task-Specific Mastery

Fine-tuning deeply integrates domain knowledge into the model. This makes it highly effective at handling structured, repetitive, or nuanced queries, scenarios where RAG-alone systems often struggle. In other words, a fine-tuned model becomes a specialist in the tasks or content it was trained on.

#### Independence from Retrieval

A fine-tuned model has no dependency on external data sources at inference time. It remains reliable even if a connected retrieval system fails or is incomplete, because all needed information is already within the model’s own parameters. This self-sufficiency means fewer points of failure in production.

#### Faster Responses

Fine-tuned models don’t need to call out to an external knowledge base during generation. Skipping the retrieval step means they can produce answers much more quickly. This speed makes fine-tuned models ideal for time-sensitive applications where every second counts.

#### Custom Behavior and Tone

Fine-tuning allows precise control over how the model communicates. This ensures the model’s responses stay consistent with a brand’s voice, adhere to regulatory requirements, or match specific tone preferences. You get a model that not only knows _what_ to say, but _how_ to say it in the desired style.

#### Reliable Performance

Even in a hybrid setup that uses both fine-tuning and RAG, the fine-tuned model provides a reliable fallback. If the retrieval component fails to find the right information or returns incorrect data, the model’s built-in knowledge can still generate a useful answer. This guarantees more consistent and robust performance for your system.

## Common Misconceptions

Despite fine-tuning’s advantages, a few myths persist. Let’s address two of the most common misconceptions about fine-tuning:

### Does Fine-Tuning Add New Knowledge to a Model?

**Yes - it absolutely can.** A common myth suggests that fine-tuning doesn’t introduce new knowledge, but in reality it does. If your fine-tuning dataset contains new domain-specific information, the model will learn that content during training and incorporate it into its responses. In effect, fine-tuning _can and does_ teach the model new facts and patterns from scratch.

### Is RAG Always Better Than Fine-Tuning?

**Not necessarily.** Many assume RAG will consistently outperform a fine-tuned model, but that’s not the case when fine-tuning is done properly. In fact, a well-tuned model often matches or even surpasses RAG-based systems on specialized tasks. Claims that “RAG is always better” usually stem from fine-tuning attempts that weren’t optimally configured - for example, using incorrect [LoRA parameters](../fine-tuning-llms-guide/lora-hyperparameters-guide) or insufficient training.

Unsloth takes care of these complexities by automatically selecting the best parameter configurations for you. All you need is a good-quality dataset, and you'll get a fine-tuned model that performs to its fullest potential.

### Is Fine-Tuning Expensive?

**Not at all!** While full fine-tuning or pretraining can be costly, these are not necessary (pretraining is especially not necessary). In most cases, LoRA or QLoRA fine-tuning can be done for minimal cost. In fact, with Unsloth’s [free notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks) for Colab or Kaggle, you can fine-tune models without spending a dime. Better yet, you can even fine-tune locally on your own device.

## FAQ:

### Why You Should Combine RAG & Fine-Tuning

Instead of choosing between RAG and fine-tuning, consider using **both** together for the best results. Combining a retrieval system with a fine-tuned model brings out the strengths of each approach. Here’s why:

* **Task-Specific Expertise** – Fine-tuning excels at specialized tasks or formats (making the model an expert in a specific area), while RAG keeps the model up-to-date with the latest external knowledge.
* **Better Adaptability** – A fine-tuned model can still give useful answers even if the retrieval component fails or returns incomplete information. Meanwhile, RAG ensures the system stays current without requiring you to retrain the model for every new piece of data.
* **Efficiency** – Fine-tuning provides a strong foundational knowledge base within the model, and RAG handles dynamic or quickly-changing details without the need for exhaustive re-training from scratch. This balance yields an efficient workflow and reduces overall compute costs.

### LoRA vs. QLoRA: Which One to Use?

When it comes to implementing fine-tuning, two popular techniques can dramatically cut down the compute and memory requirements: **LoRA** and **QLoRA**. Here’s a quick comparison of each:

* **LoRA (Low-Rank Adaptation)** – Fine-tunes only a small set of additional “adapter” weight matrices (in 16-bit precision), while leaving most of the original model unchanged. This significantly reduces the number of parameters that need updating during training.
* **QLoRA (Quantized LoRA)** – Combines LoRA with 4-bit quantization of the model weights, enabling efficient fine-tuning of very large models on minimal hardware. By using 4-bit precision where possible, it dramatically lowers memory usage and compute overhead.

We recommend starting with **QLoRA**, as it’s one of the most efficient and accessible methods available. Thanks to Unsloth’s [dynamic 4-bit](https://unsloth.ai/blog/dynamic-4bit) quants, the accuracy loss compared to standard 16-bit LoRA fine-tuning is now negligible.

### Experimentation is Key

There’s no single “best” approach to fine-tuning - only best practices for different scenarios. It’s important to experiment with different methods and configurations to find what works best for your dataset and use case. A great starting point is **QLoRA (4-bit)**, which offers a very cost-effective, resource-friendly way to fine-tune models without heavy computational requirements.

{% content-ref url="../fine-tuning-llms-guide/lora-hyperparameters-guide" %}
[lora-hyperparameters-guide](../fine-tuning-llms-guide/lora-hyperparameters-guide)
{% endcontent-ref %}

---
description: 'Explore our catalog of Unsloth notebooks:'
---

# Unsloth Notebooks

Also see our GitHub repo for our notebooks: [github.com/unslothai/notebooks](https://github.com/unslothai/notebooks/)

<a href="#grpo-reasoning-rl-notebooks" class="button secondary">GRPO (RL)</a><a href="#text-to-speech-tts-notebooks" class="button secondary">Text-to-speech (TTS)</a><a href="#vision-multimodal-notebooks" class="button secondary">Vision</a><a href="#other-important-notebooks" class="button secondary">Use-case</a>

{% tabs %}
{% tab title="• Google Colab" %}
#### Standard notebooks:

* [**gpt-oss (20b)**](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-\(20B\)-Fine-tuning.ipynb) **- new**
* [Gemma 3n (E4B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3N_\(4B\)-Conversational.ipynb) • [Text](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3N_\(4B\)-Conversational.ipynb) • [Vision](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3N_\(4B\)-Vision.ipynb) • [Audio](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3N_\(4B\)-Audio.ipynb)
* [Qwen3 (14B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_\(14B\)-Reasoning-Conversational.ipynb)
* [Gemma 3 (4B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_\(4B\).ipynb) • [Text](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_\(4B\).ipynb) • [Vision](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_\(4B\)-Vision.ipynb)
* [Phi-4 (14B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb)&#x20;
* [Llama 3.1 (8B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_\(8B\)-Alpaca.ipynb)
* [Llama 3.2 (1B + 3B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_\(1B_and_3B\)-Conversational.ipynb)
* [Qwen 2.5 (7B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_\(7B\)-Alpaca.ipynb)

#### GRPO (Reasoning RL) notebooks:

* [**Qwen3 (4B)**](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_\(4B\)-GRPO.ipynb) **-** Advanced GRPO LoRA
* [**DeepSeek-R1-0528-Qwen3 (8B)**](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/DeepSeek_R1_0528_Qwen3_\(8B\)_GRPO.ipynb) (for multilingual usecase)
* [Gemma 3 (1B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_\(1B\)-GRPO.ipynb)
* [Llama 3.2 (3B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Advanced_Llama3_2_\(3B\)_GRPO_LoRA.ipynb) - Advanced GRPO LoRA
* [Llama 3.1 (8B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_\(8B\)-GRPO.ipynb)
* [Phi-4 (14B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4_\(14B\)-GRPO.ipynb)&#x20;
* [Mistral v0.3 (7B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_\(7B\)-GRPO.ipynb)
* [Qwen2.5 (3B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_\(3B\)-GRPO.ipynb)

#### Text-to-Speech (TTS) notebooks:

* [Sesame-CSM (1B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Sesame_CSM_\(1B\)-TTS.ipynb) - new
* [Orpheus-TTS (3B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Orpheus_\(3B\)-TTS.ipynb)
* [Whisper Large V3](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Whisper.ipynb) - Speech-to-Text (STT)
* [Llasa-TTS (1B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llasa_TTS_\(1B\).ipynb)
* [Spark-TTS (0.5B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Spark_TTS_\(0_5B\).ipynb)
* [Oute-TTS (1B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Oute_TTS_\(1B\).ipynb)

#### Vision (Multimodal) notebooks:

* [Gemma 3 (4B) vision](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_\(4B\)-Vision.ipynb) - new
* [Llama 3.2 Vision (11B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_\(11B\)-Vision.ipynb)
* [Qwen2.5-VL (7B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_VL_\(7B\)-Vision.ipynb)
* [Pixtral (12B) 2409](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Pixtral_\(12B\)-Vision.ipynb)

#### Other important notebooks:

* [**Synthetic Data Generation Llama 3.2 (3B)**](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Meta_Synthetic_Data_Llama3_2_\(3B\).ipynb) - new
* [**Tool Calling**](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_Coder_\(1.5B\)-Tool_Calling.ipynb) **- new**
* [**ModernBERT-large**](https://colab.research.google.com/github/timothelaborie/text_classification_scripts/blob/main/bert_classification.ipynb) **- new**
* [Mistral v0.3 Instruct (7B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_\(7B\)-Conversational.ipynb)
* [Ollama](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_\(8B\)-Ollama.ipynb)
* [ORPO](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_\(8B\)-ORPO.ipynb)
* [Continued Pretraining](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_\(7B\)-CPT.ipynb)
* [DPO Zephyr](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Zephyr_\(7B\)-DPO.ipynb)
* [_**Inference only**_](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_\(8B\)-Inference.ipynb)
* [Llama 3 (8B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_\(8B\)-Alpaca.ipynb)

#### Specific use-case notebooks:

* [DPO Zephyr](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Zephyr_\(7B\)-DPO.ipynb)
* [**Text Classification**](https://colab.research.google.com/github/timothelaborie/text_classification_scripts/blob/main/unsloth_classification.ipynb) **- new (by Timotheeee)**
* [Ollama](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_\(8B\)-Ollama.ipynb)
* [**Tool Calling**](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_Coder_\(1.5B\)-Tool_Calling.ipynb) **- new**
* [Continued Pretraining (CPT)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_\(7B\)-CPT.ipynb)
* [Multiple Datasets](https://colab.research.google.com/drive/1njCCbE1YVal9xC83hjdo2hiGItpY_D6t?usp=sharing) by Flail
* [KTO](https://colab.research.google.com/drive/1MRgGtLWuZX4ypSfGguFgC-IblTvO2ivM?usp=sharing) by Jeffrey
* [Inference chat UI](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Unsloth_Studio.ipynb)
* [Conversational](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_\(1B_and_3B\)-Conversational.ipynb)
* [ChatML](https://colab.research.google.com/drive/15F1xyn8497_dUbxZP4zWmPZ3PJx1Oymv?usp=sharing)
* [Text Completion](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_\(7B\)-Text_Completion.ipynb)

#### Rest of notebooks:

* [Gemma 2 (9B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma2_\(9B\)-Alpaca.ipynb)
* [Mistral NeMo (12B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_Nemo_\(12B\)-Alpaca.ipynb)
* [Phi-3.5 (mini)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_3.5_Mini-Conversational.ipynb)
* [Phi-3 (medium)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_3_Medium-Conversational.ipynb)
* [Gemma 2 (2B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma2_\(2B\)-Alpaca.ipynb)
* [Qwen 2.5 Coder (14B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_Coder_\(14B\)-Conversational.ipynb)
* [Mistral Small (22B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_Small_\(22B\)-Alpaca.ipynb)
* [TinyLlama](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/TinyLlama_\(1.1B\)-Alpaca.ipynb)
* [CodeGemma (7B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/CodeGemma_\(7B\)-Conversational.ipynb)
* [Mistral v0.3 (7B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_\(7B\)-Alpaca.ipynb)
* [Qwen2 (7B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2_\(7B\)-Alpaca.ipynb)
{% endtab %}

{% tab title="• Kaggle" %}
#### Standard notebooks:

* [**gpt-oss (20B)**](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-\(20B\)-Fine-tuning.ipynb) **- new**
* [Gemma 3n (E4B)](https://www.kaggle.com/code/danielhanchen/gemma-3n-4b-multimodal-finetuning-inference)
* [Qwen3 (14B)](https://www.kaggle.com/notebooks/welcome?src=https%3A%2F%2Fgithub.com%2Funslothai/notebooks/blob/main/nb/Kaggle-Qwen3_\(14B\).ipynb)
* [Gemma 3 (4B)](https://www.kaggle.com/notebooks/welcome?src=https%3A%2F%2Fgithub.com%2Funslothai/notebooks/blob/main/nb/Kaggle-Gemma3_\(4B\).ipynb)
* [Phi-4 (14B)](https://www.kaggle.com/notebooks/welcome?src=https%3A%2F%2Fgithub.com%2Funslothai/notebooks/blob/main/nb/Kaggle-Phi_4-Conversational.ipynb)
* [Llama 3.1 (8B)](https://www.kaggle.com/notebooks/welcome?src=https%3A%2F%2Fgithub.com%2Funslothai/notebooks/blob/main/nb/Kaggle-Llama3.1_\(8B\)-Alpaca.ipynb)
* [Llama 3.2 (1B + 3B)](https://www.kaggle.com/notebooks/welcome?src=https%3A%2F%2Fgithub.com%2Funslothai/notebooks/blob/main/nb/Kaggle-Llama3.2_\(1B_and_3B\)-Conversational.ipynb)
* [Qwen 2.5 (7B)](https://www.kaggle.com/notebooks/welcome?src=https%3A%2F%2Fgithub.com%2Funslothai/notebooks/blob/main/nb/Kaggle-Qwen2.5_\(7B\)-Alpaca.ipynb)

#### GRPO (Reasoning) notebooks:

* [Qwen3 (4B)](https://www.kaggle.com/notebooks/welcome?src=https://github.com/unslothai/notebooks/blob/main/nb/Kaggle-Qwen3_\(4B\)-GRPO.ipynb\&accelerator=nvidiaTeslaT4)
* [Gemma 3 (1B)](https://www.kaggle.com/notebooks/welcome?src=https%3A%2F%2Fgithub.com%2Funslothai/notebooks/blob/main/nb/Kaggle-Gemma3_\(1B\)-GRPO.ipynb)
* [Llama 3.1 (8B)](https://www.kaggle.com/notebooks/welcome?src=https%3A%2F%2Fgithub.com%2Funslothai/notebooks/blob/main/nb/Kaggle-Llama3.1_\(8B\)-GRPO.ipynb)
* [Phi-4 (14B)](https://www.kaggle.com/notebooks/welcome?src=https%3A%2F%2Fgithub.com%2Funslothai/notebooks/blob/main/nb/Kaggle-Phi_4_\(14B\)-GRPO.ipynb)
* [Qwen 2.5 (3B)](https://www.kaggle.com/notebooks/welcome?src=https%3A%2F%2Fgithub.com%2Funslothai/notebooks/blob/main/nb/Kaggle-Qwen2.5_\(3B\)-GRPO.ipynb)

#### Text-to-Speech (TTS) notebooks:

* [Sesame-CSM (1B)](https://www.kaggle.com/notebooks/welcome?src=https%3A%2F%2Fgithub.com%2Funslothai/notebooks/blob/main/nb/Kaggle-Sesame_CSM_\(1B\)-TTS.ipynb)
* [Orpheus-TTS (3B)](https://www.kaggle.com/notebooks/welcome?src=https%3A%2F%2Fgithub.com%2Funslothai/notebooks/blob/main/nb/Kaggle-Orpheus_\(3B\)-TTS.ipynb)
* [Whisper Large V3](https://www.kaggle.com/notebooks/welcome?src=https%3A%2F%2Fgithub.com%2Funslothai/notebooks/blob/main/nb/Kaggle-Whisper.ipynb) – Speech-to-Text
* [Llasa-TTS (1B)](https://www.kaggle.com/notebooks/welcome?src=https%3A%2F%2Fgithub.com%2Funslothai/notebooks/blob/main/nb/Kaggle-Llasa_TTS_\(1B\).ipynb)
* [Spark-TTS (0.5B)](https://www.kaggle.com/notebooks/welcome?src=https%3A%2F%2Fgithub.com%2Funslothai/notebooks/blob/main/nb/Kaggle-Spark_TTS_\(0_5B\).ipynb)
* [Oute-TTS (1B)](https://www.kaggle.com/notebooks/welcome?src=https%3A%2F%2Fgithub.com%2Funslothai/notebooks/blob/main/nb/Kaggle-Oute_TTS_\(1B\).ipynb)

#### Vision (Multimodal) notebooks:

* [Llama 3.2 Vision (11B)](https://www.kaggle.com/notebooks/welcome?src=https%3A%2F%2Fgithub.com%2Funslothai/notebooks/blob/main/nb/Kaggle-Llama3.2_\(11B\)-Vision.ipynb)
* [Qwen 2.5-VL (7B)](https://www.kaggle.com/notebooks/welcome?src=https%3A%2F%2Fgithub.com%2Funslothai/notebooks/blob/main/nb/Kaggle-Qwen2.5_VL_\(7B\)-Vision.ipynb)
* [Pixtral (12B) 2409](https://www.kaggle.com/notebooks/welcome?src=https%3A%2F%2Fgithub.com%2Funslothai/notebooks/blob/main/nb/Kaggle-Pixtral_\(12B\)-Vision.ipynb)

#### Specific use-case notebooks:

* [Tool Calling](https://www.kaggle.com/notebooks/welcome?src=https%3A%2F%2Fgithub.com%2Funslothai/notebooks/blob/main/nb/Kaggle-Qwen2.5_Coder_\(14B\)-Tool_Calling.ipynb)
* [ORPO](https://www.kaggle.com/notebooks/welcome?src=https%3A%2F%2Fgithub.com%2Funslothai/notebooks/blob/main/nb/Kaggle-Llama3_\(8B\)-ORPO.ipynb)
* [Continued Pretraining](https://www.kaggle.com/notebooks/welcome?src=https%3A%2F%2Fgithub.com%2Funslothai/notebooks/blob/main/nb/Kaggle-Mistral_v0.3_\(7B\)-CPT.ipynb)
* [DPO Zephyr](https://www.kaggle.com/notebooks/welcome?src=https%3A%2F%2Fgithub.com%2Funslothai/notebooks/blob/main/nb/Kaggle-Zephyr_\(7B\)-DPO.ipynb)
* [Inference only](https://www.kaggle.com/notebooks/welcome?src=https%3A%2F%2Fgithub.com%2Funslothai/notebooks/blob/main/nb/Kaggle-Llama3.1_\(8B\)-Inference.ipynb)
* [Ollama](https://www.kaggle.com/notebooks/welcome?src=https%3A%2F%2Fgithub.com%2Funslothai/notebooks/blob/main/nb/Kaggle-Llama3_\(8B\)-Ollama.ipynb)
* [Text Completion](https://www.kaggle.com/notebooks/welcome?src=https%3A%2F%2Fgithub.com%2Funslothai/notebooks/blob/main/nb/Kaggle-Mistral_\(7B\)-Text_Completion.ipynb)
* [CodeForces-cot (Reasoning)](https://www.kaggle.com/notebooks/welcome?src=https%3A%2F%2Fgithub.com%2Funslothai/notebooks/blob/main/nb/Kaggle-CodeForces-cot-Finetune_for_Reasoning_on_CodeForces.ipynb)
* [Unsloth Studio (chat UI)](https://www.kaggle.com/notebooks/welcome?src=https%3A%2F%2Fgithub.com%2Funslothai/notebooks/blob/main/nb/Kaggle-Unsloth_Studio.ipynb)

#### Rest of notebooks:

* [Gemma 2 (9B)](https://www.kaggle.com/notebooks/welcome?src=https%3A%2F%2Fgithub.com%2Funslothai/notebooks/blob/main/nb/Kaggle-Gemma2_\(9B\)-Alpaca.ipynb)
* [Gemma 2 (2B)](https://www.kaggle.com/notebooks/welcome?src=https%3A%2F%2Fgithub.com%2Funslothai/notebooks/blob/main/nb/Kaggle-Gemma2_\(2B\)-Alpaca.ipynb)
* [CodeGemma (7B)](https://www.kaggle.com/notebooks/welcome?src=https%3A%2F%2Fgithub.com%2Funslothai/notebooks/blob/main/nb/Kaggle-CodeGemma_\(7B\)-Conversational.ipynb)
* [Mistral NeMo (12B)](https://www.kaggle.com/notebooks/welcome?src=https%3A%2F%2Fgithub.com%2Funslothai/notebooks/blob/main/nb/Kaggle-Mistral_Nemo_\(12B\)-Alpaca.ipynb)
* [Mistral Small (22B)](https://www.kaggle.com/notebooks/welcome?src=https%3A%2F%2Fgithub.com%2Funslothai/notebooks/blob/main/nb/Kaggle-Mistral_Small_\(22B\)-Alpaca.ipynb)
* [TinyLlama (1.1B)](https://www.kaggle.com/notebooks/welcome?src=https%3A%2F%2Fgithub.com%2Funslothai/notebooks/blob/main/nb/Kaggle-TinyLlama_\(1.1B\)-Alpaca.ipynb)



To view a complete list of all our Kaggle notebooks, [click here](https://github.com/unslothai/notebooks#-kaggle-notebooks).
{% endtab %}
{% endtabs %}

{% hint style="info" %}
Feel free to contribute to the notebooks by visiting our [repo](https://github.com/unslothai/notebooks)!
{% endhint %}

# All Our Models

Unsloth model catalog for all our [Dynamic](../basics/unsloth-dynamic-2.0-ggufs) GGUF, 4-bit, 16-bit models on Hugging Face.

{% tabs %}
{% tab title="• GGUF + 4-bit" %}
<a href="#deepseek-models" class="button secondary">DeepSeek</a><a href="#llama-models" class="button secondary">Llama</a><a href="#gemma-models" class="button secondary">Gemma</a><a href="#qwen-models" class="button secondary">Qwen</a><a href="#mistral-models" class="button secondary">Mistral</a><a href="#phi-models" class="button secondary">Phi</a>

**GGUFs** let you run models in tools like Ollama, Open WebUI, and llama.cpp.\
**Instruct (4-bit)** safetensors can be used for inference or fine-tuning.

### New & recommended models:

| Model                | Variant                | GGUF                                                                            | Instruct (4-bit)                                                                            |
| -------------------- | ---------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **gpt-oss (new)**    | 120b                   | [link](https://huggingface.co/unsloth/gpt-oss-120b-GGUF)                        | [link](https://huggingface.co/unsloth/gpt-oss-120b-unsloth-bnb-4bit)                        |
|                      | 20b                    | [link](https://huggingface.co/unsloth/gpt-oss-20b-GGUF)                         | [link](https://app.gitbook.com/o/HpyELzcNe0topgVLGCZY/s/xhOjnexMCB3dmuQFQ2Zq/)              |
| **Qwen3-2507**       | 30B-A3B-Instruct       | [link](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF)         | —                                                                                           |
|                      | 30B-A3B-Thinking       | [link](https://huggingface.co/unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF)         | —                                                                                           |
|                      | 235B-A22B-Thinking     | [link](https://huggingface.co/unsloth/Qwen3-235B-A22B-Thinking-2507-GGUF/)      | —                                                                                           |
|                      | 235B-A22B-Instruct     | [link](https://huggingface.co/unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF/)      | —                                                                                           |
| **Qwen3-Coder**      | 30B-A3B                | [link](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF)        | —                                                                                           |
|                      | 480B-A35B              | [link](https://huggingface.co/unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF)      | —                                                                                           |
| **Kimi K2**          | 1T                     | [link](https://huggingface.co/unsloth/Kimi-K2-Instruct-GGUF)                    | —                                                                                           |
| **Gemma 3n**         | E2B                    | [link](https://huggingface.co/unsloth/gemma-3n-E2B-it-GGUF)                     | [link](https://huggingface.co/unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit)                     |
|                      | E4B                    | [link](https://huggingface.co/unsloth/gemma-3n-E4B-it-GGUF)                     | [link](https://huggingface.co/unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit)                     |
| **DeepSeek-R1-0528** | R1-0528-Qwen3-8B       | [link](https://huggingface.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF)           | [link](https://huggingface.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-unsloth-bnb-4bit)           |
|                      | R1-0528                | [link](https://huggingface.co/unsloth/DeepSeek-R1-0528-GGUF)                    | —                                                                                           |
| **Mistral**          | Magistral Small (2507) | [link](https://huggingface.co/unsloth/Magistral-Small-2507-GGUF)                | [link](https://huggingface.co/unsloth/Magistral-Small-2507-unsloth-bnb-4bit)                |
|                      | Small 3.2 24B (2506)   | [link](https://huggingface.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF) | [link](https://huggingface.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-unsloth-bnb-4bit) |
| FLUX.1               | Kontext-dev            | [link](https://huggingface.co/unsloth/FLUX.1-Kontext-dev-GGUF)                  | —                                                                                           |
| **Qwen3**            | 0.6 B                  | [link](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF)                          | [link](https://huggingface.co/unsloth/Qwen3-0.6B-unsloth-bnb-4bit)                          |
|                      | 1.7 B                  | [link](https://huggingface.co/unsloth/Qwen3-1.7B-GGUF)                          | [link](https://huggingface.co/unsloth/Qwen3-1.7B-unsloth-bnb-4bit)                          |
|                      | 4 B                    | [link](https://huggingface.co/unsloth/Qwen3-4B-GGUF)                            | [link](https://huggingface.co/unsloth/Qwen3-4B-unsloth-bnb-4bit)                            |
|                      | 8 B                    | [link](https://huggingface.co/unsloth/Qwen3-8B-GGUF)                            | [link](https://huggingface.co/unsloth/Qwen3-8B-unsloth-bnb-4bit)                            |
|                      | 14 B                   | [link](https://huggingface.co/unsloth/Qwen3-14B-GGUF)                           | [link](https://huggingface.co/unsloth/Qwen3-14B-unsloth-bnb-4bit)                           |
|                      | 30B-A3B                | [link](https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF)                       | [link](https://huggingface.co/unsloth/Qwen3-30B-A3B-bnb-4bit)                               |
|                      | 32 B                   | [link](https://huggingface.co/unsloth/Qwen3-32B-GGUF)                           | [link](https://huggingface.co/unsloth/Qwen3-32B-unsloth-bnb-4bit)                           |
|                      | 235B-A22B              | [link](https://huggingface.co/unsloth/Qwen3-235B-A22B-GGUF)                     | —                                                                                           |
| **Llama 4**          | Scout 17B 16E          | [link](https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF)      | [link](https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit)      |
|                      | Maverick 17B 128E      | [link](https://huggingface.co/unsloth/Llama-4-Maverick-17B-128E-Instruct-GGUF)  | —                                                                                           |
| **Qwen-2.5 Omni**    | 3 B                    | [link](https://huggingface.co/unsloth/Qwen2.5-Omni-3B-GGUF)                     | —                                                                                           |
|                      | 7 B                    | [link](https://huggingface.co/unsloth/Qwen2.5-Omni-7B-GGUF)                     | —                                                                                           |
| **Phi-4**            | Reasoning-plus         | [link](https://huggingface.co/unsloth/Phi-4-reasoning-plus-GGUF)                | [link](https://huggingface.co/unsloth/Phi-4-reasoning-plus-unsloth-bnb-4bit)                |
|                      | Reasoning              | [link](https://huggingface.co/unsloth/Phi-4-reasoning-GGUF)                     | [link](https://huggingface.co/unsloth/phi-4-reasoning-unsloth-bnb-4bit)                     |

### DeepSeek models:

| Model           | Variant                | GGUF                                                                      | Instruct (4-bit)                                                                      |
| --------------- | ---------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| **DeepSeek-V3** | V3-0324                | [link](https://huggingface.co/unsloth/DeepSeek-V3-0324-GGUF)              | —                                                                                     |
|                 | V3                     | [link](https://huggingface.co/unsloth/DeepSeek-V3-GGUF)                   | —                                                                                     |
| **DeepSeek-R1** | R1-0528                | [link](https://huggingface.co/unsloth/DeepSeek-R1-0528-GGUF)              | —                                                                                     |
|                 | R1-0528-Qwen3-8B       | [link](https://huggingface.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF)     | [link](https://huggingface.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-unsloth-bnb-4bit)     |
|                 | R1                     | [link](https://huggingface.co/unsloth/DeepSeek-R1-GGUF)                   | —                                                                                     |
|                 | R1 Zero                | [link](https://huggingface.co/unsloth/DeepSeek-R1-Zero-GGUF)              | —                                                                                     |
|                 | Distill Llama 3 8 B    | [link](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF)  | [link](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit)  |
|                 | Distill Llama 3.3 70 B | [link](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF) | [link](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-70B-bnb-4bit)         |
|                 | Distill Qwen 2.5 1.5 B | [link](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF) | [link](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit) |
|                 | Distill Qwen 2.5 7 B   | [link](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF)   | [link](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit)   |
|                 | Distill Qwen 2.5 14 B  | [link](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF)  | [link](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit)  |
|                 | Distill Qwen 2.5 32 B  | [link](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF)  | [link](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit)          |

### Llama models:

| Model         | Variant             | GGUF                                                                           | Instruct (4-bit)                                                                       |
| ------------- | ------------------- | ------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------- |
| **Llama 4**   | Scout 17 B-16 E     | [link](https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF)     | [link](https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit) |
|               | Maverick 17 B-128 E | [link](https://huggingface.co/unsloth/Llama-4-Maverick-17B-128E-Instruct-GGUF) | —                                                                                      |
| **Llama 3.3** | 70 B                | [link](https://huggingface.co/unsloth/Llama-3.3-70B-Instruct-GGUF)             | [link](https://huggingface.co/unsloth/Llama-3.3-70B-Instruct-bnb-4bit)                 |
| **Llama 3.2** | 1 B                 | [link](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF)              | [link](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-bnb-4bit)                  |
|               | 3 B                 | [link](https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF)              | [link](https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-bnb-4bit)                  |
|               | 11 B Vision         | —                                                                              | [link](https://huggingface.co/unsloth/Llama-3.2-11B-Vision-Instruct-unsloth-bnb-4bit)  |
|               | 90 B Vision         | —                                                                              | [link](https://huggingface.co/unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit)          |
| **Llama 3.1** | 8 B                 | [link](https://huggingface.co/unsloth/Llama-3.1-8B-Instruct-GGUF)              | [link](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit)             |
|               | 70 B                | —                                                                              | [link](https://huggingface.co/unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit)            |
|               | 405 B               | —                                                                              | [link](https://huggingface.co/unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit)           |
| **Llama 3**   | 8 B                 | —                                                                              | [link](https://huggingface.co/unsloth/llama-3-8b-Instruct-bnb-4bit)                    |
|               | 70 B                | —                                                                              | [link](https://huggingface.co/unsloth/llama-3-70b-bnb-4bit)                            |
| **Llama 2**   | 7 B                 | —                                                                              | [link](https://huggingface.co/unsloth/llama-2-7b-chat-bnb-4bit)                        |
|               | 13 B                | —                                                                              | [link](https://huggingface.co/unsloth/llama-2-13b-bnb-4bit)                            |
| **CodeLlama** | 7 B                 | —                                                                              | [link](https://huggingface.co/unsloth/codellama-7b-bnb-4bit)                           |
|               | 13 B                | —                                                                              | [link](https://huggingface.co/unsloth/codellama-13b-bnb-4bit)                          |
|               | 34 B                | —                                                                              | [link](https://huggingface.co/unsloth/codellama-34b-bnb-4bit)                          |

### Gemma models:

| Model        | Variant       | GGUF                                                         | Instruct (4-bit)                                                             |
| ------------ | ------------- | ------------------------------------------------------------ | ---------------------------------------------------------------------------- |
| **Gemma 3n** | E2B           | ​[link](https://huggingface.co/unsloth/gemma-3n-E2B-it-GGUF) | [link](https://huggingface.co/unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit)      |
|              | E4B           | [link](https://huggingface.co/unsloth/gemma-3n-E4B-it-GGUF)  | [link](https://huggingface.co/unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit)      |
| **Gemma 3**  | 1 B           | [link](https://huggingface.co/unsloth/gemma-3-1b-it-GGUF)    | [link](https://huggingface.co/unsloth/gemma-3-1b-it-unsloth-bnb-4bit)        |
|              | 4 B           | [link](https://huggingface.co/unsloth/gemma-3-4b-it-GGUF)    | [link](https://huggingface.co/unsloth/gemma-3-4b-it-unsloth-bnb-4bit)        |
|              | 12 B          | [link](https://huggingface.co/unsloth/gemma-3-12b-it-GGUF)   | [link](https://huggingface.co/unsloth/gemma-3-12b-it-unsloth-bnb-4bit)       |
|              | 27 B          | [link](https://huggingface.co/unsloth/gemma-3-27b-it-GGUF)   | [link](https://huggingface.co/unsloth/gemma-3-27b-it-unsloth-bnb-4bit)       |
| **MedGemma** | 4 B (vision)  | [link](https://huggingface.co/unsloth/medgemma-4b-it-GGUF)   | [link](https://huggingface.co/unsloth/medgemma-4b-it-unsloth-bnb-4bit)       |
|              | 27 B (vision) | [link](https://huggingface.co/unsloth/medgemma-27b-it-GGUF)  | [link](https://huggingface.co/unsloth/medgemma-27b-text-it-unsloth-bnb-4bit) |
| **Gemma 2**  | 2 B           | [link](https://huggingface.co/unsloth/gemma-2-it-GGUF)       | [link](https://huggingface.co/unsloth/gemma-2-2b-it-bnb-4bit)                |
|              | 9 B           | —                                                            | [link](https://huggingface.co/unsloth/gemma-2-9b-it-bnb-4bit)                |
|              | 27 B          | —                                                            | [link](https://huggingface.co/unsloth/gemma-2-27b-it-bnb-4bit)               |

### Qwen models:

| Model                      | Variant    | GGUF                                                                         | Instruct (4-bit)                                                                |
| -------------------------- | ---------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| **Qwen 3**                 | 0.6 B      | [link](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF)                       | [link](https://huggingface.co/unsloth/Qwen3-0.6B-unsloth-bnb-4bit)              |
|                            | 1.7 B      | [link](https://huggingface.co/unsloth/Qwen3-1.7B-GGUF)                       | [link](https://huggingface.co/unsloth/Qwen3-1.7B-unsloth-bnb-4bit)              |
|                            | 4 B        | [link](https://huggingface.co/unsloth/Qwen3-4B-GGUF)                         | [link](https://huggingface.co/unsloth/Qwen3-4B-unsloth-bnb-4bit)                |
|                            | 8 B        | [link](https://huggingface.co/unsloth/Qwen3-8B-GGUF)                         | [link](https://huggingface.co/unsloth/Qwen3-8B-unsloth-bnb-4bit)                |
|                            | 14 B       | [link](https://huggingface.co/unsloth/Qwen3-14B-GGUF)                        | [link](https://huggingface.co/unsloth/Qwen3-14B-unsloth-bnb-4bit)               |
|                            | 30 B-A3B   | [link](https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF)                    | [link](https://huggingface.co/unsloth/Qwen3-30B-A3B-bnb-4bit)                   |
|                            | 32 B       | [link](https://huggingface.co/unsloth/Qwen3-32B-GGUF)                        | [link](https://huggingface.co/unsloth/Qwen3-32B-unsloth-bnb-4bit)               |
|                            | 235 B-A22B | [link](https://huggingface.co/unsloth/Qwen3-235B-A22B-GGUF)                  | —                                                                               |
| **Qwen 2.5 Omni**          | 3 B        | [link](https://huggingface.co/unsloth/Qwen2.5-Omni-3B-GGUF)                  | —                                                                               |
|                            | 7 B        | [link](https://huggingface.co/unsloth/Qwen2.5-Omni-7B-GGUF)                  | —                                                                               |
| **Qwen 2.5 VL**            | 3 B        | [link](https://huggingface.co/unsloth/Qwen2.5-VL-3B-Instruct-GGUF)           | [link](https://huggingface.co/unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit)  |
|                            | 7 B        | [link](https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF)           | [link](https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit)  |
|                            | 32 B       | [link](https://huggingface.co/unsloth/Qwen2.5-VL-32B-Instruct-GGUF)          | [link](https://huggingface.co/unsloth/Qwen2.5-VL-32B-Instruct-unsloth-bnb-4bit) |
|                            | 72 B       | [link](https://huggingface.co/unsloth/Qwen2.5-VL-72B-Instruct-GGUF)          | [link](https://huggingface.co/unsloth/Qwen2.5-VL-72B-Instruct-unsloth-bnb-4bit) |
| **Qwen 2.5**               | 0.5 B      | —                                                                            | [link](https://huggingface.co/unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit)           |
|                            | 1.5 B      | —                                                                            | [link](https://huggingface.co/unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit)           |
|                            | 3 B        | —                                                                            | [link](https://huggingface.co/unsloth/Qwen2.5-3B-Instruct-bnb-4bit)             |
|                            | 7 B        | —                                                                            | [link](https://huggingface.co/unsloth/Qwen2.5-7B-Instruct-bnb-4bit)             |
|                            | 14 B       | —                                                                            | [link](https://huggingface.co/unsloth/Qwen2.5-14B-Instruct-bnb-4bit)            |
|                            | 32 B       | —                                                                            | [link](https://huggingface.co/unsloth/Qwen2.5-32B-Instruct-bnb-4bit)            |
|                            | 72 B       | —                                                                            | [link](https://huggingface.co/unsloth/Qwen2.5-72B-Instruct-bnb-4bit)            |
| **Qwen 2.5 Coder (128 K)** | 0.5 B      | [link](https://huggingface.co/unsloth/Qwen2.5-Coder-0.5B-Instruct-128K-GGUF) | [link](https://huggingface.co/unsloth/Qwen2.5-Coder-0.5B-Instruct-bnb-4bit)     |
|                            | 1.5 B      | [link](https://huggingface.co/unsloth/Qwen2.5-Coder-1.5B-Instruct-128K-GGUF) | [link](https://huggingface.co/unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit)     |
|                            | 3 B        | [link](https://huggingface.co/unsloth/Qwen2.5-Coder-3B-Instruct-128K-GGUF)   | [link](https://huggingface.co/unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit)       |
|                            | 7 B        | [link](https://huggingface.co/unsloth/Qwen2.5-Coder-7B-Instruct-128K-GGUF)   | [link](https://huggingface.co/unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit)       |
|                            | 14 B       | [link](https://huggingface.co/unsloth/Qwen2.5-Coder-14B-Instruct-128K-GGUF)  | [link](https://huggingface.co/unsloth/Qwen2.5-Coder-14B-Instruct-bnb-4bit)      |
|                            | 32 B       | [link](https://huggingface.co/unsloth/Qwen2.5-Coder-32B-Instruct-128K-GGUF)  | [link](https://huggingface.co/unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit)      |
| **QwQ**                    | 32 B       | [link](https://huggingface.co/unsloth/QwQ-32B-GGUF)                          | [link](https://huggingface.co/unsloth/QwQ-32B-unsloth-bnb-4bit)                 |
| **QVQ (preview)**          | 72 B       | —                                                                            | [link](https://huggingface.co/unsloth/QVQ-72B-Preview-bnb-4bit)                 |
| **Qwen 2 (chat)**          | 1.5 B      | —                                                                            | [link](https://huggingface.co/unsloth/Qwen2-1.5B-Instruct-bnb-4bit)             |
|                            | 7 B        | —                                                                            | [link](https://huggingface.co/unsloth/Qwen2-7B-Instruct-bnb-4bit)               |
|                            | 72 B       | —                                                                            | [link](https://huggingface.co/unsloth/Qwen2-72B-Instruct-bnb-4bit)              |
| **Qwen 2 VL**              | 2 B        | —                                                                            | [link](https://huggingface.co/unsloth/Qwen2-VL-2B-Instruct-unsloth-bnb-4bit)    |
|                            | 7 B        | —                                                                            | [link](https://huggingface.co/unsloth/Qwen2-VL-7B-Instruct-unsloth-bnb-4bit)    |
|                            | 72 B       | —                                                                            | [link](https://huggingface.co/unsloth/Qwen2-VL-72B-Instruct-bnb-4bit)           |

### Mistral models:

<table><thead><tr><th width="174">Model</th><th>Variant</th><th>GGUF</th><th>Instruct (4-bit)</th></tr></thead><tbody><tr><td><strong>Mistral Small</strong></td><td>3.2-24 B (2506)</td><td><a href="https://huggingface.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF">link</a></td><td><a href="https://huggingface.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-unsloth-bnb-4bit">link</a></td></tr><tr><td></td><td>3.1-24 B (2503)</td><td><a href="https://huggingface.co/unsloth/Mistral-Small-3.1-24B-Instruct-2503-GGUF">link</a></td><td><a href="https://huggingface.co/unsloth/Mistral-Small-3.1-24B-Instruct-2503-unsloth-bnb-4bit">link</a></td></tr><tr><td></td><td>3-24 B (2501)</td><td><a href="https://huggingface.co/unsloth/Mistral-Small-24B-Instruct-2501-GGUF">link</a></td><td><a href="https://huggingface.co/unsloth/Mistral-Small-24B-Instruct-2501-unsloth-bnb-4bit">link</a></td></tr><tr><td><strong>Magistral</strong></td><td>Small-24 B (2506)</td><td><a href="https://huggingface.co/unsloth/Magistral-Small-2506-GGUF">link</a></td><td><a href="https://huggingface.co/unsloth/Magistral-Small-2506-unsloth-bnb-4bit">link</a></td></tr><tr><td><strong>Devstral</strong></td><td>Small-24 B (2507)</td><td><a href="https://huggingface.co/unsloth/Devstral-Small-2507-GGUF">link</a></td><td><a href="https://huggingface.co/unsloth/Devstral-Small-2507-unsloth-bnb-4bit">link</a></td></tr><tr><td></td><td>Small-24 B (2505)</td><td><a href="https://huggingface.co/unsloth/Devstral-Small-2505-GGUF">link</a></td><td><a href="https://huggingface.co/unsloth/Devstral-Small-2505-unsloth-bnb-4bit">link</a></td></tr><tr><td><strong>Pixtral</strong></td><td>12 B (2409)</td><td>—</td><td><a href="https://huggingface.co/unsloth/Pixtral-12B-2409-bnb-4bit">link</a></td></tr><tr><td>Mistral <strong>Small</strong></td><td>2409-22 B</td><td>—</td><td><a href="https://huggingface.co/unsloth/Mistral-Small-Instruct-2409-bnb-4bit">link</a></td></tr><tr><td>Mistral <strong>NeMo</strong></td><td>12 B (2407)</td><td><a href="https://huggingface.co/unsloth/Mistral-Nemo-Instruct-2407-GGUF">link</a></td><td><a href="https://huggingface.co/unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit">link</a></td></tr><tr><td>Mistral <strong>Large</strong></td><td>2407</td><td>—</td><td><a href="https://huggingface.co/unsloth/Mistral-Large-Instruct-2407-bnb-4bit">link</a></td></tr><tr><td><strong>Mistral 7 B</strong></td><td>v0.3</td><td>—</td><td><a href="https://huggingface.co/unsloth/mistral-7b-instruct-v0.3-bnb-4bit">link</a></td></tr><tr><td></td><td>v0.2</td><td>—</td><td><a href="https://huggingface.co/unsloth/mistral-7b-instruct-v0.2-bnb-4bit">link</a></td></tr><tr><td><strong>Mixtral</strong></td><td>8 × 7 B</td><td>—</td><td><a href="https://huggingface.co/unsloth/Mixtral-8x7B-Instruct-v0.1-unsloth-bnb-4bit">link</a></td></tr></tbody></table>

### Phi models:

| Model       | Variant          | GGUF                                                             | Instruct (4-bit)                                                             |
| ----------- | ---------------- | ---------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **Phi-4**   | Reasoning-plus   | [link](https://huggingface.co/unsloth/Phi-4-reasoning-plus-GGUF) | [link](https://huggingface.co/unsloth/Phi-4-reasoning-plus-unsloth-bnb-4bit) |
|             | Reasoning        | [link](https://huggingface.co/unsloth/Phi-4-reasoning-GGUF)      | [link](https://huggingface.co/unsloth/phi-4-reasoning-unsloth-bnb-4bit)      |
|             | Mini-Reasoning   | [link](https://huggingface.co/unsloth/Phi-4-mini-reasoning-GGUF) | [link](https://huggingface.co/unsloth/Phi-4-mini-reasoning-unsloth-bnb-4bit) |
|             | Phi-4 (instruct) | [link](https://huggingface.co/unsloth/phi-4-GGUF)                | [link](https://huggingface.co/unsloth/phi-4-unsloth-bnb-4bit)                |
|             | mini (instruct)  | [link](https://huggingface.co/unsloth/Phi-4-mini-instruct-GGUF)  | [link](https://huggingface.co/unsloth/Phi-4-mini-instruct-unsloth-bnb-4bit)  |
| **Phi-3.5** | mini             | —                                                                | [link](https://huggingface.co/unsloth/Phi-3.5-mini-instruct-bnb-4bit)        |
| **Phi-3**   | mini             | —                                                                | [link](https://huggingface.co/unsloth/Phi-3-mini-4k-instruct-bnb-4bit)       |
|             | medium           | —                                                                | [link](https://huggingface.co/unsloth/Phi-3-medium-4k-instruct-bnb-4bit)     |

### Other (GLM, Orpheus, Smol, Llava etc.) models:

| Model          | Variant           | GGUF                                                                           | Instruct (4-bit)                                                          |
| -------------- | ----------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------- |
| GLM            | 4.5-Air           | [link](https://huggingface.co/unsloth/GLM-4.5-Air-GGUF)                        |                                                                           |
|                | 4.5               | [4.5](https://huggingface.co/unsloth/GLM-4.5-GGUF)                             |                                                                           |
|                | 4-32B-0414        | [4-32B-0414](https://huggingface.co/unsloth/GLM-4-32B-0414-GGUF)               |                                                                           |
| Hunyuan        | A13B              | [link](https://huggingface.co/unsloth/Hunyuan-A13B-Instruct-GGUF)              | —                                                                         |
| Orpheus        | 0.1-ft (3B)       | [link](https://app.gitbook.com/o/HpyELzcNe0topgVLGCZY/s/xhOjnexMCB3dmuQFQ2Zq/) | [link](https://huggingface.co/unsloth/orpheus-3b-0.1-ft-unsloth-bnb-4bit) |
| **LLava**      | 1.5 (7 B)         | —                                                                              | [link](https://huggingface.co/unsloth/llava-1.5-7b-hf-bnb-4bit)           |
|                | 1.6 Mistral (7 B) | —                                                                              | [link](https://huggingface.co/unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit)  |
| **TinyLlama**  | Chat              | —                                                                              | [link](https://huggingface.co/unsloth/tinyllama-chat-bnb-4bit)            |
| **SmolLM 2**   | 135 M             | [link](https://huggingface.co/unsloth/SmolLM2-135M-Instruct-GGUF)              | [link](https://huggingface.co/unsloth/SmolLM2-135M-Instruct-bnb-4bit)     |
|                | 360 M             | [link](https://huggingface.co/unsloth/SmolLM2-360M-Instruct-GGUF)              | [link](https://huggingface.co/unsloth/SmolLM2-360M-Instruct-bnb-4bit)     |
|                | 1.7 B             | [link](https://huggingface.co/unsloth/SmolLM2-1.7B-Instruct-GGUF)              | [link](https://huggingface.co/unsloth/SmolLM2-1.7B-Instruct-bnb-4bit)     |
| **Zephyr-SFT** | 7 B               | —                                                                              | [link](https://huggingface.co/unsloth/zephyr-sft-bnb-4bit)                |
| **Yi**         | 6 B (v1.5)        | —                                                                              | [link](https://huggingface.co/unsloth/Yi-1.5-6B-bnb-4bit)                 |
|                | 6 B (v1.0)        | —                                                                              | [link](https://huggingface.co/unsloth/yi-6b-bnb-4bit)                     |
|                | 34 B (chat)       | —                                                                              | [link](https://huggingface.co/unsloth/yi-34b-chat-bnb-4bit)               |
|                | 34 B (base)       | —                                                                              | [link](https://huggingface.co/unsloth/yi-34b-bnb-4bit)                    |
{% endtab %}

{% tab title="• Instruct 16-bit" %}
16-bit and 8-bit Instruct models are used for inference or fine-tuning:

### New models:

| Model                | Variant                | Instruct (16-bit)                                                          |
| -------------------- | ---------------------- | -------------------------------------------------------------------------- |
| **gpt-oss** (new)    | 20b                    | [link](https://huggingface.co/unsloth/gpt-oss-20b)                         |
|                      | 120b                   | [link](https://huggingface.co/unsloth/gpt-oss-120b)                        |
| **Gemma 3n**         | E2B                    | [link](https://huggingface.co/unsloth/gemma-3n-E4B-it)                     |
|                      | E4B                    | [link](https://huggingface.co/unsloth/gemma-3n-E2B-it)                     |
| **DeepSeek-R1-0528** | R1-0528-Qwen3-8B       | [link](https://huggingface.co/unsloth/DeepSeek-R1-0528-Qwen3-8B)           |
|                      | R1-0528                | [link](https://huggingface.co/unsloth/DeepSeek-R1-0528)                    |
| **Mistral**          | Small 3.2 24B (2506)   | [link](https://huggingface.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506) |
|                      | Small 3.1 24B (2503)   | [link](https://huggingface.co/unsloth/Mistral-Small-3.1-24B-Instruct-2503) |
|                      | Small 3.0 24B (2501)   | [link](https://huggingface.co/unsloth/Mistral-Small-24B-Instruct-2501)     |
|                      | Magistral Small (2506) | [link](https://huggingface.co/unsloth/Magistral-Small-2506)                |
| **Qwen 3**           | 0.6 B                  | [link](https://huggingface.co/unsloth/Qwen3-0.6B)                          |
|                      | 1.7 B                  | [link](https://huggingface.co/unsloth/Qwen3-1.7B)                          |
|                      | 4 B                    | [link](https://huggingface.co/unsloth/Qwen3-4B)                            |
|                      | 8 B                    | [link](https://huggingface.co/unsloth/Qwen3-8B)                            |
|                      | 14 B                   | [link](https://huggingface.co/unsloth/Qwen3-14B)                           |
|                      | 30B-A3B                | [link](https://huggingface.co/unsloth/Qwen3-30B-A3B)                       |
|                      | 32 B                   | [link](https://huggingface.co/unsloth/Qwen3-32B)                           |
|                      | 235B-A22B              | [link](https://huggingface.co/unsloth/Qwen3-235B-A22B)                     |
| **Llama 4**          | Scout 17B-16E          | [link](https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct)      |
|                      | Maverick 17B-128E      | [link](https://huggingface.co/unsloth/Llama-4-Maverick-17B-128E-Instruct)  |
| **Qwen 2.5 Omni**    | 3 B                    | [link](https://huggingface.co/unsloth/Qwen2.5-Omni-3B)                     |
|                      | 7 B                    | [link](https://huggingface.co/unsloth/Qwen2.5-Omni-7B)                     |
| **Phi-4**            | Reasoning-plus         | [link](https://huggingface.co/unsloth/Phi-4-reasoning-plus)                |
|                      | Reasoning              | [link](https://huggingface.co/unsloth/Phi-4-reasoning)                     |

### DeepSeek models

| Model           | Variant               | Instruct (16-bit)                                                    |
| --------------- | --------------------- | -------------------------------------------------------------------- |
| **DeepSeek-V3** | V3-0324               | [link](https://huggingface.co/unsloth/DeepSeek-V3-0324)              |
|                 | V3                    | [link](https://huggingface.co/unsloth/DeepSeek-V3)                   |
| **DeepSeek-R1** | R1-0528               | [link](https://huggingface.co/unsloth/DeepSeek-R1-0528)              |
|                 | R1-0528-Qwen3-8B      | [link](https://huggingface.co/unsloth/DeepSeek-R1-0528-Qwen3-8B)     |
|                 | R1                    | [link](https://huggingface.co/unsloth/DeepSeek-R1)                   |
|                 | R1 Zero               | [link](https://huggingface.co/unsloth/DeepSeek-R1-Zero)              |
|                 | Distill Llama 3 8B    | [link](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B)  |
|                 | Distill Llama 3.3 70B | [link](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-70B) |
|                 | Distill Qwen 2.5 1.5B | [link](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-1.5B) |
|                 | Distill Qwen 2.5 7B   | [link](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-7B)   |
|                 | Distill Qwen 2.5 14B  | [link](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-14B)  |
|                 | Distill Qwen 2.5 32B  | [link](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-32B)  |

### Llama models

| Family        | Variant           | Instruct (16-bit)                                                         |
| ------------- | ----------------- | ------------------------------------------------------------------------- |
| **Llama 4**   | Scout 17B-16E     | [link](https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct)     |
|               | Maverick 17B-128E | [link](https://huggingface.co/unsloth/Llama-4-Maverick-17B-128E-Instruct) |
| **Llama 3.3** | 70 B              | [link](https://huggingface.co/unsloth/Llama-3.3-70B-Instruct)             |
| **Llama 3.2** | 1 B               | [link](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct)              |
|               | 3 B               | [link](https://huggingface.co/unsloth/Llama-3.2-3B-Instruct)              |
|               | 11 B Vision       | [link](https://huggingface.co/unsloth/Llama-3.2-11B-Vision-Instruct)      |
|               | 90 B Vision       | [link](https://huggingface.co/unsloth/Llama-3.2-90B-Vision-Instruct)      |
| **Llama 3.1** | 8 B               | [link](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct)         |
|               | 70 B              | [link](https://huggingface.co/unsloth/Meta-Llama-3.1-70B-Instruct)        |
|               | 405 B             | [link](https://huggingface.co/unsloth/Meta-Llama-3.1-405B-Instruct)       |
| **Llama 3**   | 8 B               | [link](https://huggingface.co/unsloth/llama-3-8b-Instruct)                |
|               | 70 B              | [link](https://huggingface.co/unsloth/llama-3-70b-Instruct)               |
| **Llama 2**   | 7 B               | [link](https://huggingface.co/unsloth/llama-2-7b-chat)                    |

### Gemma models:

| Model        | Variant | Instruct (16-bit)                                      |
| ------------ | ------- | ------------------------------------------------------ |
| **Gemma 3n** | E2B     | [link](https://huggingface.co/unsloth/gemma-3n-E4B-it) |
|              | E4B     | [link](https://huggingface.co/unsloth/gemma-3n-E2B-it) |
| **Gemma 3**  | 1 B     | [link](https://huggingface.co/unsloth/gemma-3-1b-it)   |
|              | 4 B     | [link](https://huggingface.co/unsloth/gemma-3-4b-it)   |
|              | 12 B    | [link](https://huggingface.co/unsloth/gemma-3-12b-it)  |
|              | 27 B    | [link](https://huggingface.co/unsloth/gemma-3-27b-it)  |
| **Gemma 2**  | 2 B     | [link](https://huggingface.co/unsloth/gemma-2b-it)     |
|              | 9 B     | [link](https://huggingface.co/unsloth/gemma-9b-it)     |
|              | 27 B    | [link](https://huggingface.co/unsloth/gemma-27b-it)    |

### Qwen models:

| Family                   | Variant   | Instruct (16-bit)                                                       |
| ------------------------ | --------- | ----------------------------------------------------------------------- |
| **Qwen 3**               | 0.6 B     | [link](https://huggingface.co/unsloth/Qwen3-0.6B)                       |
|                          | 1.7 B     | [link](https://huggingface.co/unsloth/Qwen3-1.7B)                       |
|                          | 4 B       | [link](https://huggingface.co/unsloth/Qwen3-4B)                         |
|                          | 8 B       | [link](https://huggingface.co/unsloth/Qwen3-8B)                         |
|                          | 14 B      | [link](https://huggingface.co/unsloth/Qwen3-14B)                        |
|                          | 30B-A3B   | [link](https://huggingface.co/unsloth/Qwen3-30B-A3B)                    |
|                          | 32 B      | [link](https://huggingface.co/unsloth/Qwen3-32B)                        |
|                          | 235B-A22B | [link](https://huggingface.co/unsloth/Qwen3-235B-A22B)                  |
| **Qwen 2.5 Omni**        | 3 B       | [link](https://huggingface.co/unsloth/Qwen2.5-Omni-3B)                  |
|                          | 7 B       | [link](https://huggingface.co/unsloth/Qwen2.5-Omni-7B)                  |
| **Qwen 2.5 VL**          | 3 B       | [link](https://huggingface.co/unsloth/Qwen2.5-VL-3B-Instruct)           |
|                          | 7 B       | [link](https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct)           |
|                          | 32 B      | [link](https://huggingface.co/unsloth/Qwen2.5-VL-32B-Instruct)          |
|                          | 72 B      | [link](https://huggingface.co/unsloth/Qwen2.5-VL-72B-Instruct)          |
| **Qwen 2.5**             | 0.5 B     | [link](https://huggingface.co/unsloth/Qwen2.5-0.5B-Instruct)            |
|                          | 1.5 B     | [link](https://huggingface.co/unsloth/Qwen2.5-1.5B-Instruct)            |
|                          | 3 B       | [link](https://huggingface.co/unsloth/Qwen2.5-3B-Instruct)              |
|                          | 7 B       | [link](https://huggingface.co/unsloth/Qwen2.5-7B-Instruct)              |
|                          | 14 B      | [link](https://huggingface.co/unsloth/Qwen2.5-14B-Instruct)             |
|                          | 32 B      | [link](https://huggingface.co/unsloth/Qwen2.5-32B-Instruct)             |
|                          | 72 B      | [link](https://huggingface.co/unsloth/Qwen2.5-72B-Instruct)             |
| **Qwen 2.5 Coder 128 K** | 0.5 B     | [link](https://huggingface.co/unsloth/Qwen2.5-Coder-0.5B-Instruct-128K) |
|                          | 1.5 B     | [link](https://huggingface.co/unsloth/Qwen2.5-Coder-1.5B-Instruct-128K) |
|                          | 3 B       | [link](https://huggingface.co/unsloth/Qwen2.5-Coder-3B-Instruct-128K)   |
|                          | 7 B       | [link](https://huggingface.co/unsloth/Qwen2.5-Coder-7B-Instruct-128K)   |
|                          | 14 B      | [link](https://huggingface.co/unsloth/Qwen2.5-Coder-14B-Instruct-128K)  |
|                          | 32 B      | [link](https://huggingface.co/unsloth/Qwen2.5-Coder-32B-Instruct-128K)  |
| **QwQ**                  | 32 B      | [link](https://huggingface.co/unsloth/QwQ-32B)                          |
| **QVQ (preview)**        | 72 B      | —                                                                       |
| **Qwen 2 (Chat)**        | 1.5 B     | [link](https://huggingface.co/unsloth/Qwen2-1.5B-Instruct)              |
|                          | 7 B       | [link](https://huggingface.co/unsloth/Qwen2-7B-Instruct)                |
|                          | 72 B      | [link](https://huggingface.co/unsloth/Qwen2-72B-Instruct)               |
| **Qwen 2 VL**            | 2 B       | [link](https://huggingface.co/unsloth/Qwen2-VL-2B-Instruct)             |
|                          | 7 B       | [link](https://huggingface.co/unsloth/Qwen2-VL-7B-Instruct)             |
|                          | 72 B      | [link](https://huggingface.co/unsloth/Qwen2-VL-72B-Instruct)            |

### Mistral models:

| Model            | Variant        | Instruct (16-bit)                                                  |
| ---------------- | -------------- | ------------------------------------------------------------------ |
| **Mistral**      | Small 2409-22B | [link](https://huggingface.co/unsloth/Mistral-Small-Instruct-2409) |
| **Mistral**      | Large 2407     | [link](https://huggingface.co/unsloth/Mistral-Large-Instruct-2407) |
| **Mistral**      | 7B v0.3        | [link](https://huggingface.co/unsloth/mistral-7b-instruct-v0.3)    |
| **Mistral**      | 7B v0.2        | [link](https://huggingface.co/unsloth/mistral-7b-instruct-v0.2)    |
| **Pixtral**      | 12B 2409       | [link](https://huggingface.co/unsloth/Pixtral-12B-2409)            |
| **Mixtral**      | 8×7B           | [link](https://huggingface.co/unsloth/Mixtral-8x7B-Instruct-v0.1)  |
| **Mistral NeMo** | 12B 2407       | [link](https://huggingface.co/unsloth/Mistral-Nemo-Instruct-2407)  |
| **Devstral**     | Small 2505     | [link](https://huggingface.co/unsloth/Devstral-Small-2505)         |

### Phi models:

| Model       | Variant        | Instruct (16-bit)                                               |
| ----------- | -------------- | --------------------------------------------------------------- |
| **Phi-4**   | Reasoning-plus | [link](https://huggingface.co/unsloth/Phi-4-reasoning-plus)     |
|             | Reasoning      | [link](https://huggingface.co/unsloth/Phi-4-reasoning)          |
|             | Phi-4 (core)   | [link](https://huggingface.co/unsloth/Phi-4)                    |
|             | Mini-Reasoning | [link](https://huggingface.co/unsloth/Phi-4-mini-reasoning)     |
|             | Mini           | [link](https://huggingface.co/unsloth/Phi-4-mini)               |
| **Phi-3.5** | Mini           | [link](https://huggingface.co/unsloth/Phi-3.5-mini-instruct)    |
| **Phi-3**   | Mini           | [link](https://huggingface.co/unsloth/Phi-3-mini-4k-instruct)   |
|             | Medium         | [link](https://huggingface.co/unsloth/Phi-3-medium-4k-instruct) |

### Text-to-Speech (TTS) models:

| Model                  | Instruct (16-bit)                                                |
| ---------------------- | ---------------------------------------------------------------- |
| Orpheus-3B (v0.1 ft)   | [link](https://huggingface.co/unsloth/orpheus-3b-0.1-ft)         |
| Orpheus-3B (v0.1 pt)   | [link](https://huggingface.co/unsloth/orpheus-3b-0.1-pretrained) |
| Sesame-CSM 1B          | [link](https://huggingface.co/unsloth/csm-1b)                    |
| Whisper Large V3 (STT) | [link](https://huggingface.co/unsloth/whisper-large-v3)          |
| Llasa-TTS 1B           | [link](https://huggingface.co/unsloth/Llasa-1B)                  |
| Spark-TTS 0.5B         | [link](https://huggingface.co/unsloth/Spark-TTS-0.5B)            |
| Oute-TTS 1B            | [link](https://huggingface.co/unsloth/Llama-OuteTTS-1.0-1B)      |
{% endtab %}

{% tab title="• Base 4 + 16-bit" %}
Base models are usually used for fine-tuning purposes:

### New models:

| Model        | Variant           | Base (16-bit)                                                    | Base (4-bit)                                                                           |
| ------------ | ----------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| **Gemma 3n** | E2B               | [link](https://huggingface.co/unsloth/gemma-3n-E2B)              | [link](https://huggingface.co/unsloth/gemma-3n-E2B-unsloth-bnb-4bit)                   |
|              | E4B               | [link](https://huggingface.co/unsloth/gemma-3n-E4B)              | [link](https://huggingface.co/unsloth/gemma-3n-E4B-unsloth-bnb-4bit)                   |
| **Qwen 3**   | 0.6 B             | [link](https://huggingface.co/unsloth/Qwen3-0.6B-Base)           | [link](https://huggingface.co/unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit)                |
|              | 1.7 B             | [link](https://huggingface.co/unsloth/Qwen3-1.7B-Base)           | [link](https://huggingface.co/unsloth/Qwen3-1.7B-Base-unsloth-bnb-4bit)                |
|              | 4 B               | [link](https://huggingface.co/unsloth/Qwen3-4B-Base)             | [link](https://huggingface.co/unsloth/Qwen3-4B-Base-unsloth-bnb-4bit)                  |
|              | 8 B               | [link](https://huggingface.co/unsloth/Qwen3-8B-Base)             | [link](https://huggingface.co/unsloth/Qwen3-8B-Base-unsloth-bnb-4bit)                  |
|              | 14 B              | [link](https://huggingface.co/unsloth/Qwen3-14B-Base)            | [link](https://huggingface.co/unsloth/Qwen3-14B-Base-unsloth-bnb-4bit)                 |
|              | 30B-A3B           | [link](https://huggingface.co/unsloth/Qwen3-30B-A3B-Base)        | [link](https://huggingface.co/unsloth/Qwen3-30B-A3B-Base-bnb-4bit)                     |
| **Llama 4**  | Scout 17B 16E     | [link](https://huggingface.co/unsloth/Llama-4-Scout-17B-16E)     | [link](https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit) |
|              | Maverick 17B 128E | [link](https://huggingface.co/unsloth/Llama-4-Maverick-17B-128E) | —                                                                                      |

### **Llama models:**

| Model         | Variant           | Base (16-bit)                                                    | Base (4-bit)                                                |
| ------------- | ----------------- | ---------------------------------------------------------------- | ----------------------------------------------------------- |
| **Llama 4**   | Scout 17B 16E     | [link](https://huggingface.co/unsloth/Llama-4-Scout-17B-16E)     | —                                                           |
|               | Maverick 17B 128E | [link](https://huggingface.co/unsloth/Llama-4-Maverick-17B-128E) | —                                                           |
| **Llama 3.3** | 70 B              | [link](https://huggingface.co/unsloth/Llama-3.3-70B)             | —                                                           |
| **Llama 3.2** | 1 B               | [link](https://huggingface.co/unsloth/Llama-3.2-1B)              | —                                                           |
|               | 3 B               | [link](https://huggingface.co/unsloth/Llama-3.2-3B)              | —                                                           |
|               | 11 B Vision       | [link](https://huggingface.co/unsloth/Llama-3.2-11B-Vision)      | —                                                           |
|               | 90 B Vision       | [link](https://huggingface.co/unsloth/Llama-3.2-90B-Vision)      | —                                                           |
| **Llama 3.1** | 8 B               | [link](https://huggingface.co/unsloth/Meta-Llama-3.1-8B)         | —                                                           |
|               | 70 B              | [link](https://huggingface.co/unsloth/Meta-Llama-3.1-70B)        | —                                                           |
| **Llama 3**   | 8 B               | [link](https://huggingface.co/unsloth/llama-3-8b)                | [link](https://huggingface.co/unsloth/llama-3-8b-bnb-4bit)  |
| **Llama 2**   | 7 B               | [link](https://huggingface.co/unsloth/llama-2-7b)                | [link](https://huggingface.co/unsloth/llama-2-7b-bnb-4bit)  |
|               | 13 B              | [link](https://huggingface.co/unsloth/llama-2-13b)               | [link](https://huggingface.co/unsloth/llama-2-13b-bnb-4bit) |

### **Qwen models:**

| Model        | Variant | Base (16-bit)                                             | Base (4-bit)                                                               |
| ------------ | ------- | --------------------------------------------------------- | -------------------------------------------------------------------------- |
| **Qwen 3**   | 0.6 B   | [link](https://huggingface.co/unsloth/Qwen3-0.6B-Base)    | [link](https://huggingface.co/unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit)    |
|              | 1.7 B   | [link](https://huggingface.co/unsloth/Qwen3-1.7B-Base)    | [link](https://huggingface.co/unsloth/Qwen3-1.7B-Base-unsloth-bnb-4bit)    |
|              | 4 B     | [link](https://huggingface.co/unsloth/Qwen3-4B-Base)      | [link](https://huggingface.co/unsloth/Qwen3-4B-Base-unsloth-bnb-4bit)      |
|              | 8 B     | [link](https://huggingface.co/unsloth/Qwen3-8B-Base)      | [link](https://huggingface.co/unsloth/Qwen3-8B-Base-unsloth-bnb-4bit)      |
|              | 14 B    | [link](https://huggingface.co/unsloth/Qwen3-14B-Base)     | [link](https://huggingface.co/unsloth/Qwen3-14B-Base-unsloth-bnb-4bit)     |
|              | 30B-A3B | [link](https://huggingface.co/unsloth/Qwen3-30B-A3B-Base) | [link](https://huggingface.co/unsloth/Qwen3-30B-A3B-Base-unsloth-bnb-4bit) |
| **Qwen 2.5** | 0.5 B   | [link](https://huggingface.co/unsloth/Qwen2.5-0.5B)       | [link](https://huggingface.co/unsloth/Qwen2.5-0.5B-bnb-4bit)               |
|              | 1.5 B   | [link](https://huggingface.co/unsloth/Qwen2.5-1.5B)       | [link](https://huggingface.co/unsloth/Qwen2.5-1.5B-bnb-4bit)               |
|              | 3 B     | [link](https://huggingface.co/unsloth/Qwen2.5-3B)         | [link](https://huggingface.co/unsloth/Qwen2.5-3B-bnb-4bit)                 |
|              | 7 B     | [link](https://huggingface.co/unsloth/Qwen2.5-7B)         | [link](https://huggingface.co/unsloth/Qwen2.5-7B-bnb-4bit)                 |
|              | 14 B    | [link](https://huggingface.co/unsloth/Qwen2.5-14B)        | [link](https://huggingface.co/unsloth/Qwen2.5-14B-bnb-4bit)                |
|              | 32 B    | [link](https://huggingface.co/unsloth/Qwen2.5-32B)        | [link](https://huggingface.co/unsloth/Qwen2.5-32B-bnb-4bit)                |
|              | 72 B    | [link](https://huggingface.co/unsloth/Qwen2.5-72B)        | [link](https://huggingface.co/unsloth/Qwen2.5-72B-bnb-4bit)                |
| **Qwen 2**   | 1.5 B   | [link](https://huggingface.co/unsloth/Qwen2-1.5B)         | [link](https://huggingface.co/unsloth/Qwen2-1.5B-bnb-4bit)                 |
|              | 7 B     | [link](https://huggingface.co/unsloth/Qwen2-7B)           | [link](https://huggingface.co/unsloth/Qwen2-7B-bnb-4bit)                   |

### **Llama models:**

| Model         | Variant           | Base (16-bit)                                                    | Base (4-bit)                                                |
| ------------- | ----------------- | ---------------------------------------------------------------- | ----------------------------------------------------------- |
| **Llama 4**   | Scout 17B 16E     | [link](https://huggingface.co/unsloth/Llama-4-Scout-17B-16E)     | —                                                           |
|               | Maverick 17B 128E | [link](https://huggingface.co/unsloth/Llama-4-Maverick-17B-128E) | —                                                           |
| **Llama 3.3** | 70 B              | [link](https://huggingface.co/unsloth/Llama-3.3-70B)             | —                                                           |
| **Llama 3.2** | 1 B               | [link](https://huggingface.co/unsloth/Llama-3.2-1B)              | —                                                           |
|               | 3 B               | [link](https://huggingface.co/unsloth/Llama-3.2-3B)              | —                                                           |
|               | 11 B Vision       | [link](https://huggingface.co/unsloth/Llama-3.2-11B-Vision)      | —                                                           |
|               | 90 B Vision       | [link](https://huggingface.co/unsloth/Llama-3.2-90B-Vision)      | —                                                           |
| **Llama 3.1** | 8 B               | [link](https://huggingface.co/unsloth/Meta-Llama-3.1-8B)         | —                                                           |
|               | 70 B              | [link](https://huggingface.co/unsloth/Meta-Llama-3.1-70B)        | —                                                           |
| **Llama 3**   | 8 B               | [link](https://huggingface.co/unsloth/llama-3-8b)                | [link](https://huggingface.co/unsloth/llama-3-8b-bnb-4bit)  |
| **Llama 2**   | 7 B               | [link](https://huggingface.co/unsloth/llama-2-7b)                | [link](https://huggingface.co/unsloth/llama-2-7b-bnb-4bit)  |
|               | 13 B              | [link](https://huggingface.co/unsloth/llama-2-13b)               | [link](https://huggingface.co/unsloth/llama-2-13b-bnb-4bit) |

### **Gemma models**

| Model       | Variant | Base (16-bit)                                         | Base (4-bit)                                                           |
| ----------- | ------- | ----------------------------------------------------- | ---------------------------------------------------------------------- |
| **Gemma 3** | 1 B     | [link](https://huggingface.co/unsloth/gemma-3-1b-pt)  | [link](https://huggingface.co/unsloth/gemma-3-1b-pt-unsloth-bnb-4bit)  |
|             | 4 B     | [link](https://huggingface.co/unsloth/gemma-3-4b-pt)  | [link](https://huggingface.co/unsloth/gemma-3-4b-pt-unsloth-bnb-4bit)  |
|             | 12 B    | [link](https://huggingface.co/unsloth/gemma-3-12b-pt) | [link](https://huggingface.co/unsloth/gemma-3-12b-pt-unsloth-bnb-4bit) |
|             | 27 B    | [link](https://huggingface.co/unsloth/gemma-3-27b-pt) | [link](https://huggingface.co/unsloth/gemma-3-27b-pt-unsloth-bnb-4bit) |
| **Gemma 2** | 2 B     | [link](https://huggingface.co/unsloth/gemma-2-2b)     | —                                                                      |
|             | 9 B     | [link](https://huggingface.co/unsloth/gemma-2-9b)     | —                                                                      |
|             | 27 B    | [link](https://huggingface.co/unsloth/gemma-2-27b)    | —                                                                      |

### **Mistral models:**

| Model       | Variant          | Base (16-bit)                                                      | Base (4-bit)                                                    |
| ----------- | ---------------- | ------------------------------------------------------------------ | --------------------------------------------------------------- |
| **Mistral** | Small 24B 2501   | [link](https://huggingface.co/unsloth/Mistral-Small-24B-Base-2501) | —                                                               |
|             | NeMo 12B 2407    | [link](https://huggingface.co/unsloth/Mistral-Nemo-Base-2407)      | —                                                               |
|             | 7B v0.3          | [link](https://huggingface.co/unsloth/mistral-7b-v0.3)             | [link](https://huggingface.co/unsloth/mistral-7b-v0.3-bnb-4bit) |
|             | 7B v0.2          | [link](https://huggingface.co/unsloth/mistral-7b-v0.2)             | [link](https://huggingface.co/unsloth/mistral-7b-v0.2-bnb-4bit) |
|             | Pixtral 12B 2409 | [link](https://huggingface.co/unsloth/Pixtral-12B-Base-2409)       | —                                                               |

### **Other (TTS, TinyLlama) models:**

| Model          | Variant        | Base (16-bit)                                                    | Base (4-bit)                                                                      |
| -------------- | -------------- | ---------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| **TinyLlama**  | 1.1 B (Base)   | [link](https://huggingface.co/unsloth/tinyllama)                 | [link](https://huggingface.co/unsloth/tinyllama-bnb-4bit)                         |
| **Orpheus-3b** | 0.1-pretrained | [link](https://huggingface.co/unsloth/orpheus-3b-0.1-pretrained) | [link](https://huggingface.co/unsloth/orpheus-3b-0.1-pretrained-unsloth-bnb-4bit) |
{% endtab %}
{% endtabs %}

---
description: Learn to install Unsloth locally or online.
---

# Installing + Updating

Unsloth works on Linux, Windows directly, Kaggle, Google Colab and more. See our [system requirements](beginner-start-here/unsloth-requirements).

**Recommended installation method:**

```
pip install unsloth
```

<table data-view="cards"><thead><tr><th data-type="content-ref"></th><th data-hidden data-card-target data-type="content-ref"></th></tr></thead><tbody><tr><td><a href="installing-+-updating/pip-install">pip-install</a></td><td><a href="installing-+-updating/pip-install">pip-install</a></td></tr><tr><td><a href="installing-+-updating/windows-installation">windows-installation</a></td><td></td></tr><tr><td><a href="installing-+-updating/updating">updating</a></td><td><a href="installing-+-updating/updating">updating</a></td></tr><tr><td><a href="installing-+-updating/conda-install">conda-install</a></td><td><a href="installing-+-updating/conda-install">conda-install</a></td></tr><tr><td><a href="installing-+-updating/google-colab">google-colab</a></td><td><a href="installing-+-updating/google-colab">google-colab</a></td></tr></tbody></table>

---
description: 'To update or use an old version of Unsloth, follow the steps below:'
---

# Updating

## Standard Updating  (recommended):

```
pip install --upgrade unsloth unsloth_zoo
```

### Updating without dependency updates:

<pre><code>pip install --upgrade --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
<strong>pip install --upgrade --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth-zoo.git
</strong></code></pre>

## To use an old version of Unsloth:

```
pip install --force-reinstall --no-cache-dir --no-deps unsloth==2025.1.5
```

'2025.1.5' is one of the previous old versions of Unsloth. Change it to a specific release listed on our [Github here](https://github.com/unslothai/unsloth/releases).


---
description: 'To install Unsloth locally via Pip, follow the steps below:'
---

# Pip Install

## **Recommended installation:**

**Install with pip (recommended) for the latest pip release:**

```
pip install unsloth
```

{% hint style="info" %}
Python 3.13 does not support a lot of packages including Unsloth and vLLM. Use 3.12, 3.11, 3.10 or 3.90
{% endhint %}

**To install the latest main branch of Unsloth:**

```bash
pip uninstall unsloth unsloth_zoo -y && pip install --no-deps git+https://github.com/unslothai/unsloth_zoo.git && pip install --no-deps git+https://github.com/unslothai/unsloth.git
```

If you're installing Unsloth in Jupyter, Colab, or other notebooks, be sure to prefix the command with `!`. This isn't necessary when using a terminal

## Uninstall + Reinstall

If you're still encountering dependency issues with Unsloth, many users have resolved them by forcing uninstalling and reinstalling Unsloth:

```bash
pip install --upgrade --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
pip install --upgrade --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth-zoo.git
```

***

## Advanced Pip Installation

{% hint style="warning" %}
Do **NOT** use this if you have [Conda](conda-install).
{% endhint %}

Pip is a bit more complex since there are dependency issues. The pip command is different for `torch 2.2,2.3,2.4,2.5` and CUDA versions.

For other torch versions, we support `torch211`, `torch212`, `torch220`, `torch230`, `torch240` and for CUDA versions, we support `cu118` and `cu121` and `cu124`. For Ampere devices (A100, H100, RTX3090) and above, use `cu118-ampere` or `cu121-ampere` or `cu124-ampere`.

For example, if you have `torch 2.4` and `CUDA 12.1`, use:

```bash
pip install --upgrade pip
pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"
```

Another example, if you have `torch 2.5` and `CUDA 12.4`, use:

```bash
pip install --upgrade pip
pip install "unsloth[cu124-torch250] @ git+https://github.com/unslothai/unsloth.git"
```

And other examples:

```bash
pip install "unsloth[cu121-ampere-torch240] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu118-ampere-torch240] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu118-torch240] @ git+https://github.com/unslothai/unsloth.git"

pip install "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu121-ampere-torch230] @ git+https://github.com/unslothai/unsloth.git"

pip install "unsloth[cu121-torch250] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu124-ampere-torch250] @ git+https://github.com/unslothai/unsloth.git"
```

Or, run the below in a terminal to get the **optimal** pip installation command:

```bash
wget -qO- https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/_auto_install.py | python -
```

Or, run the below manually in a Python REPL:

```python
try: import torch
except: raise ImportError('Install torch via `pip install torch`')
from packaging.version import Version as V
v = V(torch.__version__)
cuda = str(torch.version.cuda)
is_ampere = torch.cuda.get_device_capability()[0] >= 8
if cuda != "12.1" and cuda != "11.8" and cuda != "12.4": raise RuntimeError(f"CUDA = {cuda} not supported!")
if   v <= V('2.1.0'): raise RuntimeError(f"Torch = {v} too old!")
elif v <= V('2.1.1'): x = 'cu{}{}-torch211'
elif v <= V('2.1.2'): x = 'cu{}{}-torch212'
elif v  < V('2.3.0'): x = 'cu{}{}-torch220'
elif v  < V('2.4.0'): x = 'cu{}{}-torch230'
elif v  < V('2.5.0'): x = 'cu{}{}-torch240'
elif v  < V('2.6.0'): x = 'cu{}{}-torch250'
else: raise RuntimeError(f"Torch = {v} too new!")
x = x.format(cuda.replace(".", ""), "-ampere" if is_ampere else "")
print(f'pip install --upgrade pip && pip install "unsloth[{x}] @ git+https://github.com/unslothai/unsloth.git"')
```
---
description: See how to install Unsloth on Windows with or without WSL.
---

# Windows Installation

## Method #1 - Windows directly:

{% hint style="info" %}
Python 3.13 does not support Unsloth. Use 3.12, 3.11 or 3.10.

Need help or experiencing an error? Ask on our [GitHub Discussions](https://github.com/unslothai/unsloth/discussions/1849) thread for Windows support!
{% endhint %}

{% stepper %}
{% step %}
**Install NVIDIA Video Driver**

You should install the latest version of your GPUs driver. Download drivers here: [NVIDIA GPU Drive](https://www.nvidia.com/Download/index.aspx)
{% endstep %}

{% step %}
**Install Visual Studio C++**

You will need Visual Studio, with C++ installed. By default, C++ is not installed with Visual Studio, so make sure you select all of the C++ options. Also select options for Windows 10/11 SDK.

* Launch the Installer here:  [Visual Studio Community Edition](https://visualstudio.microsoft.com/vs/community/)
* In the installer, navigate to individual components and select all the options listed here:
  * **.NET Framework 4.8 SDK**
  * **.NET Framework 4.7.2 targeting pack**
  * **C# and Visual Basic Roslyn compilers**
  * **MSBuild**
  * **MSVC v143 - VS 2022 C++ x64/x86 build tools**
  * **C++ 2022 Redistributable Update**
  * **C++ CMake tools for Windows**
  * **C++/CLI support for v143 build tools (Latest)**
  * **MSBuild support for LLVM (clang-cl) toolset**
  * **C++ Clang Compiler for Windows (19.1.1)**
  * **Windows 11 SDK (10.0.22621.0)**
  * **Windows Universal CRT SDK**
  * **C++ 2022 Redistributable MSMs**

**Easier method:** Or you can open an elevated Command Prompt or PowerShell:

* Search for "cmd" or "PowerShell", right-click it, and choose "Run as administrator."
* Paste and run this command (update the Visual Studio path if necessary):

```
"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vs_installer.exe" modify ^
--installPath "C:\Program Files\Microsoft Visual Studio\2022\Community" ^
--add Microsoft.Net.Component.4.8.SDK ^
--add Microsoft.Net.Component.4.7.2.TargetingPack ^
--add Microsoft.VisualStudio.Component.Roslyn.Compiler ^
--add Microsoft.Component.MSBuild ^
--add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 ^
--add Microsoft.VisualStudio.Component.VC.Redist.14.Latest ^
--add Microsoft.VisualStudio.Component.VC.CMake.Project ^
--add Microsoft.VisualStudio.Component.VC.CLI.Support ^
--add Microsoft.VisualStudio.Component.VC.Llvm.Clang ^
--add Microsoft.VisualStudio.ComponentGroup.ClangCL ^
--add Microsoft.VisualStudio.Component.Windows11SDK.22621 ^
--add Microsoft.VisualStudio.Component.Windows10SDK.19041 ^
--add Microsoft.VisualStudio.Component.UniversalCRT.SDK ^
--add Microsoft.VisualStudio.Component.VC.Redist.MSM
```
{% endstep %}

{% step %}
**Install Python and CUDA Toolkit**

Follow the instructions to install [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive).

Then install Miniconda (which has Python) here: [https://www.anaconda.com/docs/getting-started/miniconda/install](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions)
{% endstep %}

{% step %}
**Install PyTorch**

You will need the correct version of PyTorch that is compatible with your CUDA drivers, so make sure to select them carefully. [Install PyTorch](https://pytorch.org/get-started/locally/)
{% endstep %}

{% step %}
**Install Unsloth**

Open Conda command prompt or your terminal with Python and run the command:

```
pip install "unsloth[windows] @ git+https://github.com/unslothai/unsloth.git"
```
{% endstep %}
{% endstepper %}

{% hint style="warning" %}
If you're using GRPO or plan to use vLLM, currently vLLM does not support Windows directly but only via WSL or Linux.
{% endhint %}

### **Notes**

To run Unsloth directly on Windows:

* Install Triton from this Windows fork and follow the instructions [here](https://github.com/woct0rdho/triton-windows) (be aware that the Windows fork requires PyTorch >= 2.4 and CUDA 12)
* In the SFTTrainer, set `dataset_num_proc=1` to avoid a crashing issue:

```python
trainer = SFTTrainer(
    dataset_num_proc=1,
    ...
)
```

### **Advanced/Troubleshooting**

For **advanced installation instructions** or if you see weird errors during installations:

1. Install `torch` and `triton`. Go to https://pytorch.org to install it. For example `pip install torch torchvision torchaudio triton`
2. Confirm if CUDA is installated correctly. Try `nvcc`. If that fails, you need to install `cudatoolkit` or CUDA drivers.
3. Install `xformers` manually. You can try installing `vllm` and seeing if `vllm` succeeds. Check if `xformers` succeeded with `python -m xformers.info` Go to https://github.com/facebookresearch/xformers. Another option is to install `flash-attn` for Ampere GPUs.
4. Double check that your versions of Python, CUDA, CUDNN, `torch`, `triton`, and `xformers` are compatible with one another. The [PyTorch Compatibility Matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix) may be useful.
5. Finally, install `bitsandbytes` and check it with `python -m bitsandbytes`

***

## Method #2 - Windows using PowerShell:

#### **Step 1: Install Prerequisites**

1. **Install NVIDIA CUDA Toolkit**:
   * Download and install the appropriate version of the **NVIDIA CUDA Toolkit** from [CUDA Downloads](https://developer.nvidia.com/cuda-downloads).
   * Reboot your system after installation if prompted.
   * **Note**: No additional setup is required after installation for Unsloth.
2. **Install Microsoft C++ Build Tools**:
   * Download and install **Microsoft Build Tools for Visual Studio** from the [official website](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
   * During installation, select the **C++ build tools** workload.\
     Ensure the **MSVC compiler toolset** is included.
3. **Set Environment Variables for the C++ Compiler**:
   * Open the **System Properties** window (search for "Environment Variables" in the Start menu).
   * Click **"Environment Variables…"**.
   * Add or update the following under **System variables**:
     *   **CC**:\
         Path to the `cl.exe` C++ compiler.\
         Example (adjust if your version differs):

         ```plaintext
         C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.34.31933\bin\Hostx64\x64\cl.exe
         ```
     * **CXX**:\
       Same path as `CC`.
   * Click **OK** to save changes.
   * Verify: Open a new terminal and type `cl`. It should show version info.
4. **Install Conda**
   1. Download and install **Miniconda** from the [official website](https://docs.anaconda.com/miniconda/install/#quick-command-line-install)
   2. Follow installation instruction from the website
   3. To check whether `conda` is already installed, you can test it with `conda` in your PowerShell

#### **Step 2: Run the Unsloth Installation Script**

1. **Download the** [**unsloth\_windows.ps1**](https://github.com/unslothai/notebooks/blob/main/unsloth_windows.ps1) **PowerShell script by going through this link**.
2. **Open PowerShell as Administrator**:
   * Right-click Start and select **"Windows PowerShell (Admin)"**.
3.  **Navigate to the script’s location** using `cd`:

    ```powershell
    cd path\to\script\folder
    ```
4.  **Run the script**:

    ```powershell
    powershell.exe -ExecutionPolicy Bypass -File .\unsloth_windows.ps1
    ```

#### **Step 3: Using Unsloth**

Activate the environment after the installation completes:

```powershell
conda activate unsloth_env
```

**Unsloth and its dependencies are now ready!**

***

## Method #3 - Windows via WSL:

WSL is Window's subsystem for Linux.

1. Install python though [Python's official site](https://www.python.org/downloads/windows/).
2. Start WSL (Should already be preinstalled). Open command prompt as admin then run:

```
wsl -d ubuntu
```

Optional: If WSL is not preinstalled, go to the Microsoft store and search "Ubuntu" and the app that says Ubuntu will be WSL. Install it and run it and continue from there.

3. Update WSL:

```
sudo apt update && sudo apt upgrade -y
```

4. Install pip:

```
sudo apt install python3-pip
```

5. Install unsloth:

```
pip install unsloth
```

6. Optional: Install Jupyter Notebook to run in a Colab like environment:

```
pip3 install notebook
```

7. Launch Jupyter Notebook:

<pre><code><strong>jupyter notebook
</strong></code></pre>

8. Download any Colab notebook from Unsloth, import it into your Jupyter Notebook, adjust the parameters as needed, and execute the script.

---
description: 'To install Unsloth locally on Conda, follow the steps below:'
---

# Conda Install

{% hint style="warning" %}
Only use Conda if you have it. If not, use [Pip](pip-install).
{% endhint %}

Select either `pytorch-cuda=11.8,12.1` for CUDA 11.8 or CUDA 12.1. We support `python=3.10,3.11,3.12`.

```bash
conda create --name unsloth_env \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y
conda activate unsloth_env

pip install unsloth
```

If you're looking to install Conda in a Linux environment, [read here](https://docs.anaconda.com/miniconda/), or run the below:

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```
---
description: 'To install and run Unsloth on Google Colab, follow the steps below:'
---

# Google Colab

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FQzuUQL60uFWHpaAvDPYD%2FColab%20Options.png?alt=media&#x26;token=fb808ec5-20c5-4f42-949e-14ed26a44987" alt=""><figcaption></figcaption></figure>

If you have never used a Colab notebook, a quick primer on the notebook itself:

1. **Play Button at each "cell".** Click on this to run that cell's code. You must not skip any cells and you must run every cell in chronological order. If you encounter errors, simply rerun the cell you did not run. Another option is to click CTRL + ENTER if you don't want to click the play button.
2. **Runtime Button in the top toolbar.** You can also use this button and hit "Run all" to run the entire notebook in 1 go. This will skip all the customization steps, but is a good first try.
3. **Connect / Reconnect T4 button.** T4 is the free GPU Google is providing. It's quite powerful!

The first installation cell looks like below: Remember to click the PLAY button in the brackets \[  ]. We grab our open source Github package, and install some other packages.

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FIz2XUXhcmjheDtxfvbLA%2Fimage.png?alt=media&#x26;token=b9da0e5c-075c-48f8-8abb-5db6fdf9866b" alt=""><figcaption></figcaption></figure>

---
description: Learn all the basics and best practices of fine-tuning. Beginner-friendly.
---

# Fine-tuning LLMs Guide

## 1. Understand Fine-tuning

Fine-tuning an LLM customizes its behavior, enhances + injects knowledge, and optimizes performance for domains/specific tasks. For example:

* **GPT-4** serves as a base model; however, OpenAI fine-tuned it to better comprehend instructions and prompts, leading to the creation of ChatGPT-4 which everyone uses today.
* ​**DeepSeek-R1-Distill-Llama-8B** is a fine-tuned version of Llama-3.1-8B. DeepSeek utilized data generated by DeepSeek-R1, to fine-tune Llama-3.1-8B. This process, known as distillation (a subcategory of fine-tuning), injects the data into the Llama model to learn reasoning capabilities.

With [Unsloth](https://github.com/unslothai/unsloth), you can fine-tune for free on Colab, Kaggle, or locally with just 3GB VRAM by using our [notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks). By fine-tuning a pre-trained model (e.g. Llama-3.1-8B) on a specialized dataset, you can:

* **Update + Learn New Knowledge**: Inject and learn new domain-specific information.
* **Customize Behavior**: Adjust the model’s tone, personality, or response style.
* **Optimize for Tasks**: Improve accuracy and relevance for specific use cases.

**Example usecases**:

* Train LLM to predict if a headline impacts a company positively or negatively.
* Use historical customer interactions for more accurate and custom responses.
* Fine-tune LLM on legal texts for contract analysis, case law research, and compliance.

You can think of a fine-tuned model as a specialized agent designed to do specific tasks more effectively and efficiently. **Fine-tuning can replicate all of RAG's capabilities**, but not vice versa.

#### Fine-tuning misconceptions:

You may have heard that fine-tuning does not make a model learn new knowledge or RAG performs better than fine-tuning. That is **false**. Read more FAQ + misconceptions [here](../beginner-start-here/faq-+-is-fine-tuning-right-for-me#fine-tuning-vs.-rag-whats-the-difference):

{% content-ref url="beginner-start-here/faq-+-is-fine-tuning-right-for-me" %}
[faq-+-is-fine-tuning-right-for-me](beginner-start-here/faq-+-is-fine-tuning-right-for-me)
{% endcontent-ref %}

## 2. Choose the Right Model + Method

If you're a beginner, it is best to start with a small instruct model like Llama 3.1 (8B) and experiment from there. You'll also need to decide between QLoRA and LoRA training:

* **LoRA:** Fine-tunes small, trainable matrices in 16-bit without updating all model weights. &#x20;
* **QLoRA:** Combines LoRA with 4-bit quantization to handle very large models with minimal resources.&#x20;

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FDpWv59wCNJUR38sVMjT6%2Fmodel%20name%20change.png?alt=media&#x26;token=1283a92d-9df7-4de0-b1a1-9fc7cc483381" alt="" width="563"><figcaption></figcaption></figure>

You can change the model name to whichever model you like by matching it with model's name on Hugging Face e.g. 'unsloth/llama-3.1-8b-unsloth-bnb-4bit'.

We recommend starting with **Instruct models**, as they allow direct fine-tuning using conversational chat templates (ChatML, ShareGPT etc.) and require less data compared to **Base models** (which uses Alpaca, Vicuna etc). Learn more about the differences between [instruct and base models here](what-model-should-i-use#instruct-or-base-model).

* Model names ending in **`unsloth-bnb-4bit`** indicate they are [**Unsloth dynamic 4-bit**](https://unsloth.ai/blog/dynamic-4bit) **quants**. These models consume slightly more VRAM than standard BitsAndBytes 4-bit models but offer significantly higher accuracy.
* If a model name ends with just **`bnb-4bit`**, without "unsloth", it refers to a standard BitsAndBytes 4-bit quantization.
* Models with **no suffix** are in their original **16-bit or 8-bit formats**. While they are the original models from the official model creators, we sometimes include important fixes - such as chat template or tokenizer fixes. So it's recommended to use our versions when available.

There are other settings which you can toggle:

* **`max_seq_length = 2048`** – Controls context length. While Llama-3 supports 8192, we recommend 2048 for testing. Unsloth enables 4× longer context fine-tuning.
* **`dtype = None`** – Defaults to None; use `torch.float16` or `torch.bfloat16` for newer GPUs.
* **`load_in_4bit = True`** – Enables 4-bit quantization, reducing memory use 4× for fine-tuning. Disabling it allows for LoRA 16-bit fine-tuning to be enabled.
* To enable full fine-tuning (FFT), set `full_finetuning = True`. For 8-bit fine-tuning, set `load_in_8bit = True`. **Note:** Only one training method can be set to `True` at a time.

We recommend starting with QLoRA, as it is one of the most accessible and effective methods for training models. Our [dynamic 4-bit](https://unsloth.ai/blog/dynamic-4bit) quants, the accuracy loss for QLoRA compared to LoRA is now largely recovered.

You can also do [Text-to-speech (TTS)](../basics/text-to-speech-tts-fine-tuning), [reasoning (GRPO)](../basics/reinforcement-learning-rl-guide), [vision](../basics/vision-fine-tuning), [reinforcement learning](../basics/reinforcement-learning-rl-guide/reinforcement-learning-dpo-orpo-and-kto) (DPO, ORPO, KTO), [continued pretraining](../basics/continued-pretraining), text completion and other training methodologies with Unsloth.

Read our detailed guide on choosing the right model:

{% content-ref url="fine-tuning-llms-guide/what-model-should-i-use" %}
[what-model-should-i-use](fine-tuning-llms-guide/what-model-should-i-use)
{% endcontent-ref %}

## 3. Your Dataset

For LLMs, datasets are collections of data that can be used to train our models. In order to be useful for training, text data needs to be in a format that can be tokenized.

* You will need to create a dataset usually with 2 columns - question and answer. The quality and amount will largely reflect the end result of your fine-tune so it's imperative to get this part right.
* You can [synthetically generate data](../../basics/datasets-guide#synthetic-data-generation) and structure your dataset (into QA pairs) using ChatGPT or local LLMs.
* You can also use our new Synthetic Dataset notebook which automatically parses documents (PDFs, videos etc.), generates QA pairs and auto cleans data using local models like Llama 3.2. [Access the notebook here.](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Meta_Synthetic_Data_Llama3_2_\(3B\).ipynb)
* Fine-tuning can learn from an existing repository of documents and continuously expand its knowledge base, but just dumping data alone won’t work as well. For optimal results, curate a well-structured dataset, ideally as question-answer pairs. This enhances learning, understanding, and response accuracy.
* But, that's not always the case, e.g. if you are fine-tuning a LLM for code, just dumping all your code data can actually enable your model to yield significant performance improvements, even without structured formatting. So it really depends on your use case.

_**Read more about creating your dataset:**_

{% content-ref url="../basics/datasets-guide" %}
[datasets-guide](../basics/datasets-guide)
{% endcontent-ref %}

For most of our notebook examples, we utilize the [Alpaca dataset](https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama#id-6.-alpaca-dataset) however other notebooks like Vision will use different datasets which may need images in the answer ouput as well.

## 4. Understand Training Hyperparameters

Learn how to choose the right hyperparameters using best practices from research and real-world experiments — and understand how each one affects your model's performance.

**For a complete guide on how hyperparameters affect training, see:**

{% content-ref url="fine-tuning-llms-guide/lora-hyperparameters-guide" %}
[lora-hyperparameters-guide](fine-tuning-llms-guide/lora-hyperparameters-guide)
{% endcontent-ref %}

## 5. Installing + Requirements

We would recommend beginners to utilise our pre-made [notebooks](unsloth-notebooks) first as it's the easiest way to get started with guided steps. However, if installing locally is a must, you can install and use Unsloth - just make sure you have all the right requirements necessary. Also depending on the model and quantization you're using, you'll need enough VRAM and resources. See all the details here:

{% content-ref url="beginner-start-here/unsloth-requirements" %}
[unsloth-requirements](beginner-start-here/unsloth-requirements)
{% endcontent-ref %}

Next, you'll need to install Unsloth. Unsloth currently only supports Windows and Linux devices. Once you install Unsloth, you can copy and paste our notebooks and use them in your own local environment. We have many installation methods:

{% content-ref url="installing-+-updating" %}
[installing-+-updating](installing-+-updating)
{% endcontent-ref %}

## 6. Training + Evaluation

Once you have everything set, it's time to train! If something's not working, remember you can always change hyperparameters, your dataset etc.&#x20;

You will see a log of some numbers whilst training! This is the training loss, and your job is to set parameters to make this go to as close to 0.5 as possible! If your finetune is not reaching 1, 0.8 or 0.5, you might have to adjust some numbers. If your loss goes to 0, that's probably not a good sign as well!

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FxwOA09mtcimcQOCjP4PG%2Fimage.png?alt=media&#x26;token=39a0f525-6d4e-4c3b-af0d-82d8960d87be" alt="" width="375"><figcaption><p>The training loss will appear as numbers</p></figcaption></figure>

We generally recommend keeping the default settings unless you need longer training or larger batch sizes.

* **`per_device_train_batch_size = 2`** – Increase for better GPU utilization but beware of slower training due to padding. Instead, increase `gradient_accumulation_steps` for smoother training.
* **`gradient_accumulation_steps = 4`** – Simulates a larger batch size without increasing memory usage.
* **`max_steps = 60`** – Speeds up training. For full runs, replace with `num_train_epochs = 1` (1–3 epochs recommended to avoid overfitting).
* **`learning_rate = 2e-4`** – Lower for slower but more precise fine-tuning. Try values like `1e-4`, `5e-5`, or `2e-5`.

### Evaluation

In order to evaluate, you could do manually evaluation by just chatting with the model and see if it's to your liking.  You can also enable evaluation for Unsloth, but keep in mind it can be time-consuming depending on the dataset size. To speed up evaluation you can: reduce the evaluation dataset size or set `evaluation_steps = 100`.

For testing, you can also  take 20% of your training data and use that for testing. If you already used all of the training data, then you have to manually evaluate it. You can also use automatic eval tools like EleutherAI’s [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). Keep in mind that automated tools may not perfectly align with your evaluation criteria.

## 7. Running + Saving the model

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FRX9Byv1hlSpvmonT1PLw%2Fimage.png?alt=media&#x26;token=6043cd8c-c6a3-4cc5-a019-48baeed3b5a2" alt=""><figcaption></figcaption></figure>

Now let's run the model after we completed the training process! You can edit the yellow underlined part! In fact, because we created a multi turn chatbot, we can now also call the model as if it saw some conversations in the past like below:

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2F6DXSlsHkN8cZiiAxAV0Z%2Fimage.png?alt=media&#x26;token=846307de-7386-4bbe-894e-7d9e572244fe" alt=""><figcaption></figcaption></figure>

Reminder Unsloth itself provides **2x faster inference** natively as well, so always do not forget to call `FastLanguageModel.for_inference(model)`. If you want the model to output longer responses, set `max_new_tokens = 128` to some larger number like 256 or 1024. Notice you will have to wait longer for the result as well!

### Saving the model

For saving and using your model in desired inference engines like Ollama, vLLM, Open WebUI, we can have more information here:

{% content-ref url="../basics/running-and-saving-models" %}
[running-and-saving-models](../basics/running-and-saving-models)
{% endcontent-ref %}

We can now save the finetuned model as a small 100MB file called a LoRA adapter like below. You can instead push to the Hugging Face hub as well if you want to upload your model! Remember to get a Hugging Face token via: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and add your token!

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FBz0YDi6Sc2oEP5QWXgSz%2Fimage.png?alt=media&#x26;token=33d9e4fd-e7dc-4714-92c5-bfa3b00f86c4" alt=""><figcaption></figcaption></figure>

After saving the model, we can again use Unsloth to run the model itself! Use `FastLanguageModel` again to call it for inference!

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FzymBQrqwt4GUmCIN0Iec%2Fimage.png?alt=media&#x26;token=41a110e4-8263-426f-8fa7-cdc295cc8210" alt=""><figcaption></figcaption></figure>

## 8. We're done!

You've successfully finetuned a language model and exported it to your desired inference engine with Unsloth!

To learn more about finetuning tips and tricks, head over to our blogs which provide tremendous and educational value: [https://unsloth.ai/blog/](https://unsloth.ai/blog/)

If you need any help on finetuning, you can also join our Discord server [here](https://discord.gg/unsloth). Thanks for reading and hopefully this was helpful!

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FPEvp4xsbVObJZ1lawDj8%2Fsloth%20sparkling%20square.png?alt=media&#x26;token=876bf67d-7470-4977-a6cc-3ee02cc9440b" alt="" width="188"><figcaption></figcaption></figure>


# What Model Should I Use?

## Llama, Qwen, Mistral, Phi or?

When preparing for fine-tuning, one of the first decisions you'll face is selecting the right model. Here's a step-by-step guide to help you choose:

{% stepper %}
{% step %}
#### Choose a model that aligns with your usecase

* E.g. For image-based training, select a vision model such as _Llama 3.2 Vision_. For code datasets, opt for a specialized model like _Qwen Coder 2.5_.
* **Licensing and Requirements**: Different models may have specific licensing terms and [system requirements](../../beginner-start-here/unsloth-requirements#system-requirements). Be sure to review these carefully to avoid compatibility issues.
{% endstep %}

{% step %}
#### **Assess your storage, compute capacity and dataset**

* Use our [VRAM guideline](../../beginner-start-here/unsloth-requirements#approximate-vram-requirements-based-on-model-parameters) to determine the VRAM requirements for the model you’re considering.
* Your dataset will reflect the type of model you will use and amount of time it will take to train
{% endstep %}

{% step %}
#### **Select a Model and Parameters**

* We recommend using the latest model for the best performance and capabilities. For instance, as of January 2025, the leading 70B model is _Llama 3.3_.
* You can stay up to date by exploring our [model catalog](../all-our-models) to find the newest and relevant options.
{% endstep %}

{% step %}
#### **Choose Between Base and Instruct Models**

Further details below:
{% endstep %}
{% endstepper %}

## Instruct or Base Model?

When preparing for fine-tuning, one of the first decisions you'll face is whether to use an instruct model or a base model.

### Instruct Models

Instruct models are pre-trained with built-in instructions, making them ready to use without any fine-tuning. These models, including GGUFs and others commonly available, are optimized for direct usage and respond effectively to prompts right out of the box. Instruct models work with conversational chat templates like ChatML or ShareGPT.

### **Base Models**

Base models, on the other hand, are the original pre-trained versions without instruction fine-tuning. These are specifically designed for customization through fine-tuning, allowing you to adapt them to your unique needs. Base models are compatible with instruction-style templates like [Alpaca or Vicuna](../../basics/chat-templates), but they generally do not support conversational chat templates out of the box.

### Should I Choose Instruct or Base?

The decision often depends on the quantity, quality, and type of your data:

* **1,000+ Rows of Data**: If you have a large dataset with over 1,000 rows, it's generally best to fine-tune the base model.
* **300–1,000 Rows of High-Quality Data**: With a medium-sized, high-quality dataset, fine-tuning the base or instruct model are both viable options.
* **Less than 300 Rows**: For smaller datasets, the instruct model is typically the better choice. Fine-tuning the instruct model enables it to align with specific needs while preserving its built-in instructional capabilities. This ensures it can follow general instructions without additional input unless you intend to significantly alter its functionality.
* For information how how big your dataset should be, [see here](../../../basics/datasets-guide#how-big-should-my-dataset-be)

## Fine-tuning models with Unsloth

You can change the model name to whichever model you like by matching it with model's name on Hugging Face e.g. 'unsloth/llama-3.1-8b-unsloth-bnb-4bit'.

We recommend starting with **Instruct models**, as they allow direct fine-tuning using conversational chat templates (ChatML, ShareGPT etc.) and require less data compared to **Base models** (which uses Alpaca, Vicuna etc). Learn more about the differences between [instruct and base models here](#instruct-or-base-model).

* Model names ending in **`unsloth-bnb-4bit`** indicate they are [**Unsloth dynamic 4-bit**](https://unsloth.ai/blog/dynamic-4bit) **quants**. These models consume slightly more VRAM than standard BitsAndBytes 4-bit models but offer significantly higher accuracy.
* If a model name ends with just **`bnb-4bit`**, without "unsloth", it refers to a standard BitsAndBytes 4-bit quantization.
* Models with **no suffix** are in their original **16-bit or 8-bit formats**. While they are the original models from the official model creators, we sometimes include important fixes - such as chat template or tokenizer fixes. So it's recommended to use our versions when available.

### Experimentation is Key

{% hint style="info" %}
We recommend experimenting with both models when possible. Fine-tune each one and evaluate the outputs to see which aligns better with your goals.
{% endhint %}

---
description: >-
  Optimal lora rank. alpha, number of epochs, batch size & gradient
  accumulation, QLoRA vs LoRA, target modules and more!
---

# LoRA Hyperparameters Guide

LoRA hyperparameters are adjustable parameters that control how Low-Rank Adaptation (LoRA) fine-tunes LLMs. With many options (such as learning rate and epochs) and millions of possible combinations, selecting the right values is crucial for achieving accuracy, stability, quality, and fewer hallucinations during fine-tuning.

You'll learn the best practices for these parameters, based on insights from hundreds of research papers and experiments, and see how they impact the model. **While we recommend using Unsloth's defaults**, understanding these concepts will give you full control.\
\
The goal is to change hyperparameter numbers to increase accuracy while counteracting [**overfitting or underfitting**](#overfitting-poor-generalization-too-specialized). Overfitting occurs when the model memorizes the training data, harming its ability to generalize to new, unseen inputs. The objective is a model that generalizes well, not one that simply memorizes.

{% columns %}
{% column %}
### :question:But what is LoRA?

In LLMs, we have model weights. Llama 70B has 70 billion numbers. Instead of changing all 70b numbers, we instead add thin matrices A and B to each weight, and optimize those. This means we only optimize 1% of weights.
{% endcolumn %}

{% column %}
<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2Fx6UtLPuzEudHY7SjLDAm%2Fimage.png?alt=media&#x26;token=ca891bda-e67e-4219-b74e-4a3a9c137700" alt=""><figcaption><p>Instead of optimizing Model Weights (yellow), we optimize 2 thin matrices A and B.</p></figcaption></figure>
{% endcolumn %}
{% endcolumns %}

## :1234: Key Fine-tuning Hyperparameters

### **Learning Rate**

Defines how much the model’s weights are adjusted during each training step.

* **Higher Learning Rates**: Lead to faster initial convergence but can cause training to become unstable or fail to find an optimal minimum if set too high.
* **Lower Learning Rates**: Result in more stable and precise training but may require more epochs to converge, increasing overall training time. While low learning rates are often thought to cause underfitting, they actually can lead to **overfitting** or even prevent the model from learning.
* **Typical Range**: `2e-4` (0.0002) to `5e-6` (0.000005).  \
  :green\_square: _**For normal LoRA/QLoRA Fine-tuning**_, _we recommend_ **`2e-4`** _as a starting point._ \
  :blue\_square: _**For Reinforcement Learning** (DPO, GRPO etc.), we recommend_ **`5e-6` .** \
  :white\_large\_square: _**For Full Fine-tuning,** lower learning rates are generally more appropriate._

### **Epochs**

The number of times the model sees the full training dataset.

* **More Epochs:** Can help the model learn better, but a high number can cause it to **memorize the training data**, hurting its performance on new tasks.
* **Fewer Epochs:** Reduces training time and can prevent overfitting, but may result in an undertrained model if the number is insufficient for the model to learn the dataset's underlying patterns.
* **Recommended:** 1-3 epochs. For most instruction-based datasets, training for more than 3 epochs offers diminishing returns and increases the risk of overfitting.

### **LoRA or QLoRA**

LoRA uses 16-bit precision, while QLoRA is a 4-bit fine-tuning method.

* **LoRA:** 16-bit fine-tuning. It's slightly faster and slightly more accurate, but consumes significantly more VRAM (4× more than QLoRA). Recommended for 16-bit environments and scenarios where maximum accuracy is required.
* **QLoRA:** 4-bit fine-tuning. Slightly slower and marginally less accurate, but uses much less VRAM (4× less). \
  :sloth: _70B LLaMA fits in <48GB VRAM with QLoRA in Unsloth -_ [_more details here_](https://unsloth.ai/blog/llama3-3)_._

### Hyperparameters & Recommendations:

<table><thead><tr><th width="154.39678955078125">Hyperparameter</th><th width="383.6192626953125">Function</th><th>Recommended Settings</th></tr></thead><tbody><tr><td><strong>LoRA Rank</strong> (<code>r</code>)</td><td>Controls the number of trainable parameters in the LoRA adapter matrices. A higher rank increases model capacity but also memory usage.</td><td>8, 16, 32, 64, 128<br><br>Choose 16 or 32</td></tr><tr><td><strong>LoRA Alpha</strong> (<code>lora_alpha</code>)</td><td>Scales the strength of the fine-tuned adjustments in relation to the rank (<code>r</code>).</td><td><code>r</code> (standard) or <code>r * 2</code> (common heuristic). <a href="#lora-alpha-and-rank-relationship">More details here</a>.</td></tr><tr><td><strong>LoRA Dropout</strong></td><td>A regularization technique that randomly sets a fraction of LoRA activations to zero during training to prevent overfitting. <strong>Not that useful</strong>, so we default set it to 0. </td><td>0 (default) to 0.1</td></tr><tr><td><strong>Weight Decay</strong></td><td>A regularization term that penalizes large weights to prevent overfitting and improve generalization. Don't use too large numbers!</td><td>0.01 (recommended) - 0.1</td></tr><tr><td><strong>Warmup Steps</strong></td><td>Gradually increases the learning rate at the start of training.</td><td>5-10% of total steps</td></tr><tr><td><strong>Scheduler Type</strong></td><td>Adjusts the learning rate dynamically during training.</td><td><code>linear</code> or <code>cosine</code></td></tr><tr><td><strong>Seed (<code>random_state</code>)</strong></td><td>A fixed number to ensure reproducibility of results.</td><td>Any integer (e.g., <code>42</code>, <code>3407</code>)</td></tr><tr><td><strong>Target Modules</strong></td><td><p>Specify which parts of the model you want to apply LoRA adapters to — either the attention, the MLP, or both.</p><p><br>Attention: <code>q_proj, k_proj, v_proj, o_proj</code><br><br>MLP: <code>gate_proj, up_proj, down_proj</code></p></td><td>Recommended to target all major linear layers: <code>q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj</code>.</td></tr></tbody></table>

## :deciduous\_tree: Gradient Accumulation and Batch Size equivalency

### Effective Batch Size

Correctly configuring your batch size is critical for balancing training stability with your GPU's VRAM limitations. This is managed by two parameters whose product is the **Effective Batch Size**.\
\
**Effective Batch Size** = `batch_size * gradient_accumulation_steps`

* A **larger Effective Batch Size** generally leads to smoother, more stable training.
* A **smaller Effective Batch Size** may introduce more variance.

While every task is different, the following configuration provides a great starting point for achieving a stable **Effective Batch Size** of 16, which works well for most fine-tuning tasks on modern GPUs.

| Parameter                                                 | Description                                                                                                                                                                                                                                                                    | Recommended Setting                            |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------- |
| **Batch Size** (`batch_size`)                             | <p>The number of samples processed in a single forward/backward pass on one GPU. <br><br><strong>Primary Driver of VRAM Usage</strong>. Higher values can improve hardware utilization and speed up training, but only if they fit in memory.</p>                              | 2                                              |
| **Gradient Accumulation** (`gradient_accumulation_steps`) | <p>The number of micro-batches to process before performing a single model weight update.<br><br><strong>Primary Driver of Training Time.</strong> Allows simulation of a larger <code>batch_size</code> to conserve VRAM. Higher values increase training time per epoch.</p> | 8                                              |
| **Effective Batch Size** (Calculated)                     | The true batch size used for each gradient update. It directly influences training stability, quality, and final model performance.                                                                                                                                            | <p>4 to 16<br>Recommended: 16 (from 2 * 8)</p> |

### The VRAM & Performance Trade-off

Assume you want 32 samples of data per training step. Then you can use any of the following configurations:

* `batch_size = 32,  gradient_accumulation_steps = 1`
* `batch_size = 16,  gradient_accumulation_steps = 2`
* `batch_size = 8,   gradient_accumulation_steps = 4`
* `batch_size = 4,   gradient_accumulation_steps = 8`
* `batch_size = 2,   gradient_accumulation_steps = 16`
* `batch_size = 1,   gradient_accumulation_steps = 32`

While all of these are equivalent for the model's weight updates, they have vastly different hardware requirements.

The first configuration (`batch_size = 32`) uses the **most VRAM** and will likely fail on most GPUs.  The last configuration (`batch_size = 1`) uses the **least VRAM,** but at the cost of slightly slower trainin&#x67;**.** To avoid OOM (out of memory) errors, always prefer to set a smaller `batch_size` and increase `gradient_accumulation_steps` to reach your target **Effective Batch Size**.

### :sloth: Unsloth Gradient Accumulation Fix

Gradient accumulation and batch sizes <mark style="color:green;">**are now fully equivalent in Unsloth**</mark> due to our bug fixes for gradient accumulation. We have implemented specific bug fixes for gradient accumulation that resolve a common issue where the two methods did not produce the same results. This was a known challenge in the wider community, but for Unsloth users, the two methods are now interchangeable.

[Read our blog post](https://unsloth.ai/blog/gradient) for more details.

Prior to our fixes, combinations of `batch_size` and `gradient_accumulation_steps` that yielded the same **Effective Batch Size** (i.e., `batch_size × gradient_accumulation_steps = 16`) did not result in equivalent training behavior. For example, configurations like `b1/g16`, `b2/g8`, `b4/g4`, `b8/g2`, and `b16/g1` all have an **Effective Batch Size** of 16, but as shown in the graph, the loss curves did not align when using standard gradient accumulation:

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FfbTkE4kv2tVwCIdyxWKe%2FBefore_-_Standard_gradient_accumulation_UQOFkUggudXuV9dzrh8MA.svg?alt=media&#x26;token=c3297fd4-a96b-45d0-9925-0010165d85c6" alt=""><figcaption><p>(Before - Standard Gradient Accumulation)</p></figcaption></figure>

After applying our fixes, the loss curves now align correctly, regardless of how the **Effective Batch Size** of 16 is achieved:

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FBtwCpRAye5yq1Yvhlwn2%2FAfter_-_Unsloth_gradient_accumulation_6Y4pJdJF0vruzradUpymY.svg?alt=media&#x26;token=3b53d4ca-44f2-45b2-af41-cbf6b24fc80b" alt=""><figcaption><p>(After - 🦥 <mark style="color:green;">Unsloth Gradient Accumulation</mark>)</p></figcaption></figure>

## 🦥 **LoRA Hyperparameters in Unsloth**

The following demonstrates a standard configuration. **While Unsloth provides optimized defaults**, understanding these parameters is key to manual tuning.

<div data-full-width="false"><figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FmxdGwpEiv0XReahK4zDf%2Fnotebook_parameter_screenshott.png?alt=media&#x26;token=2e11c53c-9a23-4132-8c6e-cb81f3d78172" alt=""><figcaption></figcaption></figure></div>

1.  ```python
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    ```

    The rank (`r`) of the fine-tuning process. A larger rank uses more memory and will be slower, but can increase accuracy on complex tasks. We suggest ranks like 8 or 16 (for fast fine-tunes) and up to 128. Using a rank that is too large can cause overfitting and harm your model's quality.\

2.  ```python
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    ```

    For optimal performance, <mark style="background-color:blue;">**LoRA should be applied to all major linear layers**</mark>. [Research has shown](#lora-target-modules-and-qlora-vs-lora) that targeting all major layers is crucial for matching the performance of full fine-tuning. While it's possible to remove modules to reduce memory usage, we strongly advise against it to preserve maximum quality as the savings are minimal.\

3.  ```python
    lora_alpha = 16,
    ```

    A scaling factor that controls the strength of the fine-tuned adjustments. Setting it equal to the rank (`r`) is a reliable baseline. A popular and effective heuristic is to set it to double the rank (`r * 2`), which makes the model learn more aggressively by giving more weight to the LoRA updates. [More details here](#lora-alpha-and-rank-relationship).\

4.  ```python
    lora_dropout = 0, # Supports any, but = 0 is optimized
    ```

    A regularization technique that helps [prevent overfitting](#overfitting-poor-generalization-too-specialized) by randomly setting a fraction of the LoRA activations to zero during each training step. [Recent research suggests](https://arxiv.org/abs/2410.09692) that for **the short training runs** common in fine-tuning, `lora_dropout` may be an unreliable regularizer.\
    🦥 _Unsloth's internal code can optimize training when_ `lora_dropout = 0`_, making it slightly faster, but we recommend a non-zero value if you suspect overfitting._\

5.  ```python
    bias = "none",    # Supports any, but = "none" is optimized
    ```

    Leave this as `"none"` for faster training and reduced memory usage. This setting avoids training the bias terms in the linear layers, which adds trainable parameters for little to no practical gain.\

6.  ```python
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    ```

    Options are `True`, `False`, and `"unsloth"`. \
    🦥 _We recommend_ `"unsloth"` _as it reduces memory usage by an extra 30% and supports extremely long context fine-tunes. You can read more on_ [_our blog post about long context training_](https://unsloth.ai/blog/long-context)_._\

7.  ```python
    random_state = 3407,
    ```

    The seed to ensure deterministic, reproducible runs. Training involves random numbers, so setting a fixed seed is essential for consistent experiments.\

8.  ```python
    use_rslora = False,  # We support rank stabilized LoRA
    ```

    An advanced feature that implements [**Rank-Stabilized LoRA**](https://arxiv.org/abs/2312.03732). If set to `True`, the effective scaling becomes `lora_alpha / sqrt(r)` instead of the standard `lora_alpha / r`. This can sometimes improve stability, particularly for higher ranks. [More details here](#lora-alpha-and-rank-relationship).\

9.  ```python
    loftq_config = None, # And LoftQ
    ```

    An advanced technique, as proposed in [**LoftQ**](https://arxiv.org/abs/2310.08659), initializes LoRA matrices with the top 'r' singular vectors from the pretrained weights. This can improve accuracy but may cause a significant memory spike at the start of training.

### **Verifying LoRA Weight Updates:**

When validating that **LoRA** adapter weights have been updated after fine-tuning, avoid using **np.allclose()** for comparison. This method can miss subtle but meaningful changes, particularly in **LoRA A**, which is initialized with small Gaussian values. These changes may not register as significant under loose numerical tolerances. Thanks to [contributors](https://github.com/unslothai/unsloth/issues/3035) for this section.

To reliably confirm weight updates, we recommend:

* Using **checksum or hash comparisons** (e.g., MD5)
* Computing the **sum of absolute differences** between tensors
* Inspecting t**ensor statistics** (e.g., mean, variance) manually
* Or using **np.array\_equal()** if exact equality is expected

## :triangular\_ruler:LoRA Alpha and Rank relationship

{% hint style="success" %}
It's best to set `lora_alpha = 2 * lora_rank` or `lora_alpha = lora_rank`&#x20;
{% endhint %}

{% columns %}
{% column width="50%" %}
$$
\hat{W} = W + \frac{\alpha}{\text{rank}} \times AB
$$

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FfrlYmBPuCMy1GaXVYpIp%2Fimage.png?alt=media&#x26;token=b4cdfb81-8117-4852-a552-4869d27ea141" alt=""><figcaption><p>rsLoRA other scaling options. sqrt(r) is the best.</p></figcaption></figure>

$$
\hat{W}_{\text{rslora}} = W + \frac{\alpha}{\sqrt{\text{rank}}} \times AB
$$
{% endcolumn %}

{% column %}
The formula for LoRA is on the left. We need to scale the thin matrices A and B by alpha divided by the rank. <mark style="background-color:blue;">**This means we should keep alpha/rank at least = 1**</mark>.

According to the [rsLoRA (rank stabilized lora) paper](https://arxiv.org/abs/2312.03732), we should instead scale alpha by the sqrt of the rank. Other options exist, but theoretically this is the optimum. The left plot shows other ranks and their perplexities (lower is better). To enable this, set `use_rslora = True` in Unsloth.

Our recommendation is to set the <mark style="background-color:green;">**alpha to equal to the rank, or at least 2 times the rank.**</mark> This means alpha/rank = 1 or 2.
{% endcolumn %}
{% endcolumns %}

## :dart: LoRA Target Modules and QLoRA vs LoRA

{% hint style="success" %}
Use:\
`target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",]` to target both **MLP** and **attention** layers to increase accuracy.

**QLoRA uses 4-bit precision**, reducing VRAM usage by over 75%.

**LoRA (16-bit)** is slightly more accurate and faster.
{% endhint %}

According to empirical experiments and research papers like the original [QLoRA paper](https://arxiv.org/pdf/2305.14314), it's best to apply LoRA to both attention and MLP layers.

{% columns %}
{% column %}
<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FeTeDWK5yQhRv1YxmKyQ5%2Fimage.png?alt=media&#x26;token=a4d21361-9128-46e0-bc17-a31d212d16a1" alt=""><figcaption></figcaption></figure>
{% endcolumn %}

{% column %}
The chart shows RougeL scores (higher is better) for different target module configurations, comparing LoRA vs QLoRA.

The first 3 dots show:

1. **QLoRA-All:** LoRA applied to all FFN/MLP and Attention layers. \
   :fire: _This performs best overall._
2. **QLoRA-FFN**: LoRA only on FFN. \
   Equivalent to: `gate_proj`, `up_proj`, `down_proj.`
3. **QLoRA-Attention**: LoRA applied only to Attention layers. \
   Equivalent to: `q_proj`, `k_proj`, `v_proj`, `o_proj`.
{% endcolumn %}
{% endcolumns %}

## :sunglasses: Training on completions only, masking out inputs

The [QLoRA paper](https://arxiv.org/pdf/2305.14314) shows that masking out inputs and **training only on completions** (outputs or assistant messages) can further **increase accuracy** by a few percentage points (_1%_). Below demonstrates how this is done in Unsloth:

{% columns %}
{% column %}
**NOT** training on completions only:

**USER:** <mark style="background-color:green;">Hello what is 2+2?</mark>\
**ASSISTANT:** <mark style="background-color:green;">The answer is 4.</mark>\
**USER:** <mark style="background-color:green;">Hello what is 3+3?</mark>\
**ASSISTANT:** <mark style="background-color:green;">The answer is 6.</mark>


{% endcolumn %}

{% column %}
**Training** on completions only:

**USER:** ~~Hello what is 2+2?~~\
**ASSISTANT:** <mark style="background-color:green;">The answer is 4.</mark>\
**USER:** ~~Hello what is 3+3?~~\
**ASSISTANT:** <mark style="background-color:green;">The answer is 6</mark><mark style="background-color:green;">**.**</mark>
{% endcolumn %}
{% endcolumns %}

The QLoRA paper states that **training on completions only** increases accuracy by quite a bit, especially for multi-turn conversational finetunes! We do this in our [conversational notebooks here](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_\(1B_and_3B\)-Conversational.ipynb).

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2Fe8oeF4J6Pe2kpDE4hosL%2Fimage.png?alt=media&#x26;token=7e59cb98-10d4-4563-9e25-26d3f3fb35cb" alt=""><figcaption></figcaption></figure>

To enable **training on completions** in Unsloth, you will need to define the instruction and assistant parts. :sloth: _We plan to further automate this for you in the future!_

For Llama 3, 3.1, 3.2, 3.3 and 4 models, you define the parts as follows:

```python
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)
```

For Gemma 2, 3, 3n models, you define the parts as follows:

```python
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)
```

## :key: **Avoiding Overfitting & Underfitting**

### **Overfitting** (Poor Generalization/Too Specialized)

The model memorizes the training data, including its statistical noise, and consequently fails to generalize to unseen data.

{% hint style="success" %}
If your training loss drops below 0.2, your model is likely **overfitting** — meaning it may perform poorly on unseen tasks.

One simple trick is LoRA alpha scaling — just multiply the alpha value of each LoRA matrix by 0.5. This effectively scales down the impact of fine-tuning.

**This is closely related to merging / averaging weights.** \
You can take the original base (or instruct) model, add the LoRA weights, then divide the result by 2. This gives you an averaged model — which is functionally equivalent to reducing the `alpha` by half.
{% endhint %}

**Solution:**

* **Adjust the learning rate:** A high learning rate often leads to overfitting, especially during short training runs. For longer training, a higher learning rate may work better. It’s best to experiment with both to see which performs best.
* **Reduce the number of training epochs**. Stop training after 1, 2, or 3 epochs.
* **Increase** `weight_decay`. A value of `0.01` or `0.1` is a good starting point.
* **Increase** `lora_dropout`. Use a value like `0.1` to add regularization.
* **Increase batch size or gradient accumulation steps**.
* **Dataset expansion** - make your dataset larger by combining or concatenating open source datasets with your dataset. Choose higher quality ones.
* **Evaluation early stopping** - enable evaluation and stop when the evaluation loss increases for a few steps.
* **LoRA Alpha Scaling** - scale the alpha down after training and during inference - this will make the finetune less pronounced.
* **Weight averaging** - literally add the original instruct model and the finetune and divide the weights by 2.

### **Underfitting** (Too Generic)

The model fails to capture the underlying patterns in the training data, often due to insufficient complexity or training duration.

**Solution:**

* **Adjust the Learning Rate:** If the current rate is too low, increasing it may speed up convergence, especially for short training runs. For longer runs, try lowering the learning rate instead. Test both approaches to see which works best.
* **Increase Training Epochs:** Train for more epochs, but monitor validation loss to avoid overfitting.
* **Increase LoRA Rank** (`r`) and alpha: Rank should at least equal to the alpha number, and rank should be bigger for smaller models/more complex datasets; it usually is between 4 and 64.
* **Use a More Domain-Relevant Dataset**: Ensure the training data is high-quality and directly relevant to the target task.
* **Decrease batch size to 1**. This will cause the model to update more vigorously.

{% hint style="success" %}
Fine-tuning has no single "best" approach, only best practices. Experimentation is key to finding what works for your specific needs. Our notebooks automatically set optimal parameters based on many papers research and our experiments, giving you a great starting point. Happy fine-tuning!
{% endhint %}

_**Acknowledgements:** A huge thank you to_ [_Eyera_](https://huggingface.co/Orenguteng) _for contributing to this guide!_


---
description: Run & fine-tune OpenAI's new open-source models!
---

# gpt-oss: How to Run & Fine-tune

OpenAI releases '**gpt-oss-120b'** and '**gpt-oss-20b'**, two SOTA open language models under the Apache 2.0 license. Both 128k context models outperform similarly sized open models in reasoning, tool use, and agentic tasks. You can now run & fine-tune them locally with Unsloth!

It's best to train & use Unsloth quants due to our [fixes](#unsloth-fixes-for-gpt-oss) for the model.

> [**Fine-tune**](#fine-tuning-gpt-oss-with-unsloth) **gpt-oss-20b for free with our** [**Colab notebook**](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-\(20B\)-Fine-tuning.ipynb)

Trained with [RL](reinforcement-learning-rl-guide), **gpt-oss-120b** rivals o4-mini and **gpt-oss-20b** rivals o3-mini. Both excel at function calling and CoT reasoning, surpassing o1 and GPT-4o.

{% hint style="success" %}
**Includes Unsloth's** [**chat template fixes**](#unsloth-fixes-for-gpt-oss)**. For best results, use our uploads & train with Unsloth!**
{% endhint %}

<a href="#run-gpt-oss-20b" class="button secondary">Run gpt-oss-20b</a><a href="#run-gpt-oss-120b" class="button secondary">Run gpt-oss-120b</a><a href="#fine-tuning-gpt-oss-with-unsloth" class="button primary">Fine-tune gpt-oss</a>

#### **gpt-oss - Unsloth GGUFs:**

* 20B: [gpt-oss-**20B**](https://huggingface.co/unsloth/gpt-oss-20b-GGUF)
* 120B: [gpt-oss-**120B**](https://huggingface.co/unsloth/gpt-oss-120b-GGUF)

## :scroll:Unsloth fixes for gpt-oss

OpenAI released a standalone parsing and tokenization library called [Harmony](https://github.com/openai/harmony) which allows one to tokenize conversations to OpenAI's preferred format for gpt-oss. The official OpenAI [cookbook article](https://app.gitbook.com/o/HpyELzcNe0topgVLGCZY/s/xhOjnexMCB3dmuQFQ2Zq/) provides many more details on how to use the Harmony library.

Inference engines generally use the jinja chat template instead and not the Harmony package, and we found some issues with them after comparing with Harmony directly. If you see below, the top is the correct rendered form as from Harmony. The below is the one rendered by the current jinja chat template. There are quite a few differences!

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FFqIrmxJhFtJutzMn5wLx%2FScreenshot%202025-08-08%20at%2008-19-49%20Untitled151.ipynb%20-%20Colab.png?alt=media&#x26;token=e740b75f-1634-45ad-9be7-55370d13cd7e" alt=""><figcaption></figcaption></figure>

We also made some functions to directly allow you to use OpenAI's Harmony library directly without a jinja chat template if you desire - you can simply parse in normal conversations like below:

```python
messages = [
    {"role" : "user", "content" : "What is 1+1?"},
    {"role" : "assistant", "content" : "2"},
    {"role": "user",  "content": "What's the temperature in San Francisco now? How about tomorrow? Today's date is 2024-09-30."},
    {"role": "assistant",  "content": "User asks: 'What is the weather in San Francisco?' We need to use get_current_temperature tool.", "thinking" : ""},
    {"role": "assistant", "content": "", "tool_calls": [{"name": "get_current_temperature", "arguments": '{"location": "San Francisco, California, United States", "unit": "celsius"}'}]},
    {"role": "tool", "name": "get_current_temperature", "content": '{"temperature": 19.9, "location": "San Francisco, California, United States", "unit": "celsius"}'},
]
```

Then use the `encode_conversations_with_harmony` function from Unsloth:

```python
from unsloth_zoo import encode_conversations_with_harmony

def encode_conversations_with_harmony(
    messages,
    reasoning_effort = "medium",
    add_generation_prompt = True,
    tool_calls = None,
    developer_instructions = None,
    model_identity = "You are ChatGPT, a large language model trained by OpenAI.",
)
```

The harmony format includes multiple interesting things:

1. `reasoning_effort = "medium"` You can select low, medium or high, and this changes gpt-oss's reasoning budget - generally the higher the better the accuracy of the model.
2. `developer_instructions` is like a system prompt which you can add.
3. `model_identity` is best left alone - you can edit it, but we're unsure if custom ones will function.

We find multiple issues with current jinja chat templates (there exists multiple implementations across the ecosystem):

1. Function and tool calls are rendered with `tojson`, which is fine it's a dict, but if it's a string, speech marks and other **symbols become backslashed**.
2. There are some **extra new lines** in the jinja template on some boundaries.
3. Tool calling thoughts from the model should have the **`analysis` tag and not `final` tag**.
4. Other chat templates seem to not utilize `<|channel|>final` at all - one should use this for the final assistant message. You should not use this for thinking traces or tool calls.

Our chat templates for the GGUF, our BnB and BF16 uploads and all versions are fixed! For example when comparing both ours and Harmony's format, we get no different characters:

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2Fq3pLyJyjBA7MTENhEX8S%2FScreenshot%202025-08-08%20at%2008-20-00%20Untitled151.ipynb%20-%20Colab.png?alt=media&#x26;token=a02d2626-c535-4aa3-bd72-09bf5829ac8e" alt=""><figcaption></figcaption></figure>

## :1234: Precision issues

We found multiple precision issues in Tesla T4 and float16 machines primarily since the model was trained using BF16, and so outliers and overflows existed. MXFP4 is not actually supported on Ampere and older GPUs, so Triton provides `tl.dot_scaled` for MXFP4 matrix multiplication. It upcasts the matrices to BF16 internaly on the fly.

We made a [MXFP4 inference notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/GPT_OSS_MXFP4_\(20B\)-Inference.ipynb) as well in Tesla T4 Colab!

{% hint style="info" %}
[Software emulation](https://triton-lang.org/main/python-api/generated/triton.language.dot_scaled.html) enables targeting hardware architectures without native microscaling operation support. Right now for such case, microscaled lhs/rhs are upcasted to `bf16` element type beforehand for dot computation,
{% endhint %}

We found if you use float16 as the mixed precision autocast data-type, you will get infinities after some time. To counteract this, we found doing the MoE in bfloat16, then leaving it in either bfloat16 or float32 precision. If older GPUs don't even have bfloat16 support (like T4), then float32 is used.

We also change all precisions of operations (like the router) to float32 for float16 machines.

## 🖥️ **Running gpt-oss**

Below are guides for the [20B](#run-gpt-oss-20b) and [120B](#run-gpt-oss-120b) variants of the model.

{% hint style="info" %}
Any quant smaller than F16, including 2-bit has minimal accuracy loss, since only some parts (e.g., attention layers) are lower bit while most remain full-precision. That’s why sizes are close to the F16 model; for example, the 2-bit (11.5 GB) version performs nearly the same as the full 16-bit (14 GB) one. Once llama.cpp supports better quantization for these models, we'll upload them ASAP.
{% endhint %}

### :gear: Recommended Settings

OpenAI recommends these inference settings for both models:

`temperature=1.0`, `top_p=1.0`, `top_k=0`

* <mark style="background-color:green;">**Temperature of 1.0**</mark>
* Top\_K = 0 (or experiment with 100 for possible better results)
* Top\_P = 1.0
* Recommended minimum context: 16,384
* Maximum context length window: 131,072
* OpenAI now lets you adjust gpt-oss reasoning traces: low, medium, or high - if your tool supports it. Lower traces speed up responses but may slightly reduce answer quality.

**Chat template:**

```
<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2025-08-05\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>user<|message|>Hello<|end|><|start|>assistant<|channel|>final<|message|>Hi there!<|end|><|start|>user<|message|>What is 1+1?<|end|><|start|>assistant
```

The end of sentence/generation token: EOS is `<|return|>`

### Run gpt-oss-20B

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2F5uMxZIFbSS7976wghYcR%2Fgpt-oss-20b.svg?alt=media&#x26;token=43e2694c-317b-49ec-9723-2c08e1cc9dd3" alt=""><figcaption></figcaption></figure>

To achieve inference speeds of 6+ tokens per second for our Dynamic 4-bit quant, have at least **14GB of unified memory** (combined VRAM and RAM) or **14GB of system RAM** alone. As a rule of thumb, your available memory should match or exceed the size of the model you’re using. GGUF Link: [unsloth/gpt-oss-20b-GGUF](https://huggingface.co/unsloth/gpt-oss-20b-GGUF)

**NOTE:** The model can run on less memory than its total size, but this will slow down inference. Maximum memory is only needed for the fastest speeds.&#x20;

{% hint style="info" %}
Follow the [**best practices above**](#recommended-settings). They're the same as the 120B model.
{% endhint %}

You can run the model on Google Colab, Docker, LM Studio or llama.cpp for now. See below:

> **You can run gpt-oss-20b for free with our** [**Google Colab notebook**](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/GPT_OSS_MXFP4_\(20B\)-Inference.ipynb)

#### 🐋 Docker: Run gpt-oss-20b Tutorial

If you already have Docker desktop, all your need to do is run the command below and you're done:

```
docker model pull hf.co/unsloth/gpt-oss-20b-GGUF:F16
```

#### :sparkles: Llama.cpp: Run gpt-oss-20b Tutorial

1. Obtain the latest `llama.cpp` on [GitHub here](https://github.com/ggml-org/llama.cpp). You can follow the build instructions below as well. Change `-DGGML_CUDA=ON` to `-DGGML_CUDA=OFF` if you don't have a GPU or just want CPU inference.

```bash
apt-get update
apt-get install pciutils build-essential cmake curl libcurl4-openssl-dev -y
git clone https://github.com/ggml-org/llama.cpp
cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON
cmake --build llama.cpp/build --config Release -j --clean-first --target llama-cli llama-gguf-split
cp llama.cpp/build/bin/llama-* llama.cpp
```

2.  You can directly pull from Hugging Face via:

    ```
    ./llama.cpp/llama-cli \
        -hf unsloth/gpt-oss-20b-GGUF:F16 \
        --jinja -ngl 99 --threads -1 --ctx-size 16384 \
        --temp 1.0 --top-p 1.0 --top-k 0
    ```
3. Download the model via (after installing `pip install huggingface_hub hf_transfer` ).

```python
# !pip install huggingface_hub hf_transfer
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id = "unsloth/gpt-oss-20b-GGUF",
    local_dir = "unsloth/gpt-oss-20b-GGUF",
    allow_patterns = ["*F16*"],
)
```

### Run gpt-oss-120b:

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FuelT8du9Slmb40yhLN9g%2Fgpt-oss-120b.svg?alt=media&#x26;token=3447826e-78fc-4732-b321-70dfd513804c" alt=""><figcaption></figcaption></figure>

To achieve inference speeds of 6+ tokens per second for our 1-bit quant, we recommend at least **66GB of unified memory** (combined VRAM and RAM) or **66GB of system RAM** alone. As a rule of thumb, your available memory should match or exceed the size of the model you’re using. GGUF Link: [unsloth/gpt-oss-120b-GGUF](https://huggingface.co/unsloth/gpt-oss-120b-GGUF)

**NOTE:** The model can run on less memory than its total size, but this will slow down inference. Maximum memory is only needed for the fastest speeds.

{% hint style="info" %}
Follow the [**best practices above**](#recommended-settings).  They're the same as the 20B model.
{% endhint %}

#### 📖 Llama.cpp: Run gpt-oss-120b Tutorial

For gpt-oss-120b, we will specifically use Llama.cpp for optimized inference.

{% hint style="success" %}
If you want a **full precision unquantized version**, use our  `F16` versions!
{% endhint %}

1.  Obtain the latest `llama.cpp` on [GitHub here](https://github.com/ggml-org/llama.cpp). You can follow the build instructions below as well. Change `-DGGML_CUDA=ON` to `-DGGML_CUDA=OFF` if you don't have a GPU or just want CPU inference.

    ```bash
    apt-get update
    apt-get install pciutils build-essential cmake curl libcurl4-openssl-dev -y
    git clone https://github.com/ggml-org/llama.cpp
    cmake llama.cpp -B llama.cpp/build \
        -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON
    cmake --build llama.cpp/build --config Release -j --clean-first --target llama-cli llama-gguf-split
    cp llama.cpp/build/bin/llama-* llama.cpp
    ```
2.  You can directly use llama.cpp to download the model but I normally suggest using `huggingface_hub` To use llama.cpp directly, do:

    {% code overflow="wrap" %}
    ```bash
    ./llama.cpp/llama-cli \
        --hf unsloth/gpt-oss-120b-GGUF:F16 \
        --threads -1 \
        --ctx-size 16384 \
        --n-gpu-layers 99 \
        -ot ".ffn_.*_exps.=CPU" \
        --temp 1.0 \
        --min-p 0.0 \
        --top-p 1.0 \
        --top-k 0.0 \
    ```
    {% endcode %}
3.  Or, download the model via (after installing `pip install huggingface_hub hf_transfer` ). You can choose UD-Q2\_K\_XL, or other quantized versions..

    ```python
    # !pip install huggingface_hub hf_transfer
    import os
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0" # Can sometimes rate limit, so set to 0 to disable
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id = "unsloth/gpt-oss-120b-GGUF",
        local_dir = "unsloth/gpt-oss-120b-GGUF",
        allow_patterns = ["*F16*"],
    )
    ```


4. Run the model in conversation mode and try any prompt.
5. Edit `--threads -1` for the number of CPU threads, `--ctx-size` 262114 for context length, `--n-gpu-layers 99` for GPU offloading on how many layers. Try adjusting it if your GPU goes out of memory. Also remove it if you have CPU only inference.

{% hint style="success" %}
Use `-ot ".ffn_.*_exps.=CPU"` to offload all MoE layers to the CPU! This effectively allows you to fit all non MoE layers on 1  GPU, improving generation speeds. You can customize the regex expression to fit more layers if you have more GPU capacity. More options discussed [here](#improving-generation-speed).
{% endhint %}

{% code overflow="wrap" %}
```bash
./llama.cpp/llama-cli \
    --model unsloth/gpt-oss-120b-GGUF/gpt-oss-120b-F16.gguf \
    --threads -1 \
    --ctx-size 16384 \
    --n-gpu-layers 99 \
    -ot ".ffn_.*_exps.=CPU" \
    --temp 1.0 \
    --min-p 0.0 \
    --top-p 1.0 \
    --top-k 0.0 \
```
{% endcode %}

### :tools: Improving generation speed

If you have more VRAM, you can try offloading more MoE layers, or offloading whole layers themselves.

Normally, `-ot ".ffn_.*_exps.=CPU"`  offloads all MoE layers to the CPU! This effectively allows you to fit all non MoE layers on 1 GPU, improving generation speeds. You can customize the regex expression to fit more layers if you have more GPU capacity.

If you have a bit more GPU memory, try `-ot ".ffn_(up|down)_exps.=CPU"` This offloads up and down projection MoE layers.

Try `-ot ".ffn_(up)_exps.=CPU"` if you have even more GPU memory. This offloads only up projection MoE layers.

You can also customize the regex, for example `-ot "\.(6|7|8|9|[0-9][0-9]|[0-9][0-9][0-9])\.ffn_(gate|up|down)_exps.=CPU"` means to offload gate, up and down MoE layers but only from the 6th layer onwards.

The [latest llama.cpp release](https://github.com/ggml-org/llama.cpp/pull/14363) also introduces high throughput mode. Use `llama-parallel`. Read more about it [here](https://github.com/ggml-org/llama.cpp/tree/master/examples/parallel). You can also **quantize the KV cache to 4bits** for example to reduce VRAM / RAM movement, which can also make the generation process faster.

## 🦥 Fine-tuning gpt-oss with Unsloth

You can now fine-tune **gpt-oss-20b** for free on Kaggle using Unsloth and **16-bit LoRA.**

This was made possible by combining our brand new **Unsloth offloading techniques**, custom kernels, algorithms, and OpenAI’s Triton kernels, which speed up training and largely reduces memory use.

**Unsloth gpt-oss fine-tuning is 1.5x faster, uses 70% less VRAM, and supports 10x longer context lengths.** gpt-oss-20b LoRA training fits on a 14GB VRAM, and **gpt-oss-120b works on 65GB VRAM**.

Free Unsloth notebooks to fine-tune gpt-oss:

* gpt-oss-20b [Reasoning + Conversational notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-\(20B\)-Fine-tuning.ipynb) (recommended)
* GRPO notebooks coming soon! Stay tuned!

To fine-tune gpt-oss and leverage our latest updates, you must install the latest version of Unsloth:

```
pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo
```

### 💡Making efficient gpt-oss fine-tuning work

We found that while MXFP4 is highly efficient, it does not natively support training with gpt-oss. To overcome this limitation, we implemented custom training functions specifically for MXFP4 layers through mimicking it via `Bitsandbytes` NF4 quantization.

We utilized OpenAI's Triton Kernels library directly to allow MXFP4 inference. For finetuning / training however, the MXFP4 kernels do not yet support training, since the backwards pass is not yet implemented. We're actively working on implementing it in Triton! There is a flag called `W_TRANSPOSE` as mentioned [here](https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/matmul_ogs_details/_matmul_ogs.py#L39), which should be implemented. The derivative can be calculated by the transpose of the weight matrices, and so we have to implement the transpose operation.

If you want to train gpt-oss with any library other than Unsloth, you’ll need to upcast the weights to bf16 before training. This approach, however, **significantly increases** both VRAM usage and training time by as much as **300% more memory usage**! <mark style="background-color:green;">**ALL other training methods will require a minimum of 65GB VRAM to train the 20b model while Unsloth only requires 14GB VRAM (-80%).**</mark>

As both models use MoE architecture, the 20B model selects 4 experts out of 32, while the 120B model selects 4 out of 128 per token. During training and release, weights are stored in MXFP4 format as `nn.Parameter` objects, not as `nn.Linear` layers, which complicates quantization, especially since MoE/MLP experts make up about 19B of the 20B parameters.

To enable `BitsandBytes` quantization and memory-efficient fine-tuning, we converted these parameters into `nn.Linear` layers. Although this slightly slows down operations, it allows fine-tuning on GPUs with limited memory, a worthwhile trade-off.

### Datasets fine-tuning guide

Though gpt-oss supports only reasoning, you can still fine-tune it with a non-reasoning [dataset](datasets-guide), but this may affect its reasoning ability. If you want to maintain its reasoning capabilities (optional), you can use a mix of direct answers and chain-of-thought examples. Use at least <mark style="background-color:green;">75% reasoning</mark> and <mark style="background-color:green;">25% non-reasoning</mark> in your dataset to make the model retain its reasoning capabilities.

Our gpt-oss-20b Conversational notebook uses OpenAI's example which is Hugging Face's Multilingual-Thinking dataset.

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FQhnJE7SelxoTaAv6l8Ff%2Fwider%20gptoss%20image.png?alt=media&#x26;token=fd8d11f2-0159-44aa-a773-4cd2668f0a78" alt=""><figcaption></figcaption></figure>

---
description: >-
  Run Qwen3-Coder-30B-A3B-Instruct and 480B-A35B locally with Unsloth Dynamic
  quants.
---

# Qwen3-Coder: How to Run Locally

Qwen3-Coder is Qwen’s new series of coding agent models, available in 30B (**Qwen3-Coder-Flash**) and 480B parameters. **Qwen3-480B-A35B-Instruct** achieves SOTA coding performance rivalling Claude Sonnet-4, GPT-4.1, and [Kimi K2](kimi-k2-how-to-run-locally), with 61.8% on Aider Polygot and support for 256K (extendable to 1M) token context.

We also uploaded Qwen3-Coder with native <mark style="background-color:purple;">**1M context length**</mark> extended by YaRN and full-precision 8bit and 16bit versions. [Unsloth](https://github.com/unslothai/unsloth) also now supports fine-tuning and [RL](reinforcement-learning-rl-guide) of Qwen3-Coder.

{% hint style="success" %}
[**UPDATE:** We fixed tool-calling for Qwen3-Coder! ](#tool-calling-fixes)You can now use tool-calling seamlessly in llama.cpp, Ollama, LMStudio, Open WebUI, Jan etc. This issue was universal and affected all uploads (not just Unsloth), and we've communicated with the Qwen team about our fixes! [Read more](#tool-calling-fixes)
{% endhint %}

<a href="#run-qwen3-coder-30b-a3b-instruct" class="button secondary">Run 30B-A3B</a><a href="#run-qwen3-coder-480b-a35b-instruct" class="button secondary">Run 480B-A35B</a>

{% hint style="success" %}
**Does** [**Unsloth Dynamic Quants**](unsloth-dynamic-2.0-ggufs) **work?** Yes, and very well. In third-party testing on the Aider Polyglot benchmark, the **UD-Q4\_K\_XL (276GB)** dynamic quant nearly matched the **full bf16 (960GB)** Qwen3-coder model, scoring 60.9% vs 61.8%. [More details here.](https://huggingface.co/unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF/discussions/8)
{% endhint %}

#### **Qwen3 Coder - Unsloth Dynamic 2.0 GGUFs**:

| Dynamic 2.0 GGUF (to run)                                                                                                                                                                                                     | 1M Context Dynamic 2.0 GGUF                                                                                                                                                                                                         |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <ul><li><a href="https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF">30B-A3B-Instruct</a></li><li><a href="https://huggingface.co/unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF">480B-A35B-Instruct</a></li></ul> | <ul><li><a href="https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-1M-GGUF">30B-A3B-Instruct</a></li><li><a href="https://huggingface.co/unsloth/Qwen3-Coder-480B-A35B-Instruct-1M-GGUF">480B-A35B-Instruct</a></li></ul> |

## 🖥️ **Running Qwen3-Coder**

Below are guides for the [**30B-A3B**](#run-qwen3-coder-30b-a3b-instruct) and [**480B-A35B**](#run-qwen3-coder-480b-a35b-instruct) variants of the model.

### :gear: Recommended Settings

Qwen recommends these inference settings for both models:

`temperature=0.7`, `top_p=0.8`, `top_k=20`, `repetition_penalty=1.05`

* <mark style="background-color:green;">**Temperature of 0.7**</mark>
* Top\_K of 20
* Min\_P of 0.00 (optional, but 0.01 works well, llama.cpp default is 0.1)
* Top\_P of 0.8
* <mark style="background-color:green;">**Repetition Penalty of 1.05**</mark>
*   Chat template:&#x20;

    {% code overflow="wrap" %}
    ```
    <|im_start|>user
    Hey there!<|im_end|>
    <|im_start|>assistant
    What is 1+1?<|im_end|>
    <|im_start|>user
    2<|im_end|>
    <|im_start|>assistant
    ```
    {% endcode %}
* Recommended context output: 65,536 tokens (can be increased). Details here.

**Chat template/prompt format with newlines un-rendered**

{% code overflow="wrap" %}
```
<|im_start|>user\nHey there!<|im_end|>\n<|im_start|>assistant\nWhat is 1+1?<|im_end|>\n<|im_start|>user\n2<|im_end|>\n<|im_start|>assistant\n
```
{% endcode %}

<mark style="background-color:yellow;">**Chat template for tool calling**</mark> (Getting the current temperature for San Francisco). More details here for how to format tool calls.

```
<|im_start|>user
What's the temperature in San Francisco now? How about tomorrow?<|im_end|>
<|im_start|>assistant
<tool_call>\n<function=get_current_temperature>\n<parameter=location>\nSan Francisco, CA, USA
</parameter>\n</function>\n</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"temperature": 26.1, "location": "San Francisco, CA, USA", "unit": "celsius"}
</tool_response>\n<|im_end|>
```

{% hint style="info" %}
Reminder that this model supports only non-thinking mode and does not generate `<think></think>` blocks in its output. Meanwhile, specifying `enable_thinking=False` is no longer required.
{% endhint %}

### Run Qwen3-Coder-30B-A3B-Instruct:

To achieve inference speeds of 6+ tokens per second for our Dynamic 4-bit quant, have at least **18GB of unified memory** (combined VRAM and RAM) or **18GB of system RAM** alone. As a rule of thumb, your available memory should match or exceed the size of the model you’re using. E.g. the UD\_Q8\_K\_XL quant (full precision), which is 32.5GB, will require at least **33GB of unified memory** (VRAM + RAM) or **33GB of RAM** for optimal performance.

**NOTE:** The model can run on less memory than its total size, but this will slow down inference. Maximum memory is only needed for the fastest speeds.

Given that this is a non thinking model, there is no need to set `thinking=False` and the model does not generate `<think> </think>` blocks.

{% hint style="info" %}
Follow the [**best practices above**](#recommended-settings). They're the same as the 480B model.
{% endhint %}

#### 🦙 Ollama: Run Qwen3-Coder-30B-A3B-Instruct Tutorial

1. Install `ollama` if you haven't already! You can only run models up to 32B in size.

```bash
apt-get update
apt-get install pciutils -y
curl -fsSL https://ollama.com/install.sh | sh
```

2. Run the model! Note you can call `ollama serve`in another terminal if it fails! We include all our fixes and suggested parameters (temperature etc) in `params` in our Hugging Face upload!

```bash
ollama run hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:UD-Q4_K_XL
```

#### :sparkles: Llama.cpp: Run Qwen3-Coder-30B-A3B-Instruct Tutorial

1. Obtain the latest `llama.cpp` on [GitHub here](https://github.com/ggml-org/llama.cpp). You can follow the build instructions below as well. Change `-DGGML_CUDA=ON` to `-DGGML_CUDA=OFF` if you don't have a GPU or just want CPU inference.

```bash
apt-get update
apt-get install pciutils build-essential cmake curl libcurl4-openssl-dev -y
git clone https://github.com/ggml-org/llama.cpp
cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON
cmake --build llama.cpp/build --config Release -j --clean-first --target llama-cli llama-gguf-split
cp llama.cpp/build/bin/llama-* llama.cpp
```

2.  You can directly pull from HuggingFace via:

    ```
    ./llama.cpp/llama-cli \
        -hf unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_XL \
        --jinja -ngl 99 --threads -1 --ctx-size 32684 \
        --temp 0.7 --min-p 0.0 --top-p 0.80 --top-k 20 --repeat-penalty 1.05
    ```
3. Download the model via (after installing `pip install huggingface_hub hf_transfer` ). You can choose UD\_Q4\_K\_XL or other quantized versions.

```python
# !pip install huggingface_hub hf_transfer
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id = "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
    local_dir = "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
    allow_patterns = ["*UD-Q4_K_XL*"],
)
```

### Run Qwen3-Coder-480B-A35B-Instruct:

To achieve inference speeds of 6+ tokens per second for our 1-bit quant, we recommend at least **150GB of unified memory** (combined VRAM and RAM) or **150GB of system RAM** alone. As a rule of thumb, your available memory should match or exceed the size of the model you’re using. E.g. the Q2\_K\_XL quant, which is 180GB, will require at least **180GB of unified memory** (VRAM + RAM) or **180GB of RAM** for optimal performance.

**NOTE:** The model can run on less memory than its total size, but this will slow down inference. Maximum memory is only needed for the fastest speeds.

{% hint style="info" %}
Follow the [**best practices above**](#recommended-settings).  They're the same as the 30B model.
{% endhint %}

#### 📖 Llama.cpp: Run Qwen3-Coder-480B-A35B-Instruct Tutorial

For Coder-480B-A35B, we will specifically use Llama.cpp for optimized inference and a plethora of options.

{% hint style="success" %}
If you want a **full precision unquantized version**, use our `Q8_K_XL, Q8_0` or `BF16` versions!
{% endhint %}

1.  Obtain the latest `llama.cpp` on [GitHub here](https://github.com/ggml-org/llama.cpp). You can follow the build instructions below as well. Change `-DGGML_CUDA=ON` to `-DGGML_CUDA=OFF` if you don't have a GPU or just want CPU inference.

    ```bash
    apt-get update
    apt-get install pciutils build-essential cmake curl libcurl4-openssl-dev -y
    git clone https://github.com/ggml-org/llama.cpp
    cmake llama.cpp -B llama.cpp/build \
        -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON
    cmake --build llama.cpp/build --config Release -j --clean-first --target llama-cli llama-gguf-split
    cp llama.cpp/build/bin/llama-* llama.cpp
    ```
2.  You can directly use llama.cpp to download the model but I normally suggest using `huggingface_hub` To use llama.cpp directly, do:

    {% code overflow="wrap" %}
    ```bash
    ./llama.cpp/llama-cli \
        -hf unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF:Q2_K_XL \
        --threads -1 \
        --ctx-size 16384 \
        --n-gpu-layers 99 \
        -ot ".ffn_.*_exps.=CPU" \
        --temp 0.7 \
        --min-p 0.0 \
        --top-p 0.8 \
        --top-k 20 \
        --repeat-penalty 1.05
    ```
    {% endcode %}
3.  Or, download the model via (after installing `pip install huggingface_hub hf_transfer` ). You can choose UD-Q2\_K\_XL, or other quantized versions..

    ```python
    # !pip install huggingface_hub hf_transfer
    import os
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0" # Can sometimes rate limit, so set to 0 to disable
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id = "unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF",
        local_dir = "unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF",
        allow_patterns = ["*UD-Q2_K_XL*"],
    )
    ```


4. Run the model in conversation mode and try any prompt.
5. Edit `--threads -1` for the number of CPU threads, `--ctx-size` 262114 for context length, `--n-gpu-layers 99` for GPU offloading on how many layers. Try adjusting it if your GPU goes out of memory. Also remove it if you have CPU only inference.

{% hint style="success" %}
Use `-ot ".ffn_.*_exps.=CPU"` to offload all MoE layers to the CPU! This effectively allows you to fit all non MoE layers on 1  GPU, improving generation speeds. You can customize the regex expression to fit more layers if you have more GPU capacity. More options discussed [here](#improving-generation-speed).
{% endhint %}

{% code overflow="wrap" %}
```bash
./llama.cpp/llama-cli \
    --model unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF/UD-Q2_K_XL/Qwen3-Coder-480B-A35B-Instruct-UD-Q2_K_XL-00001-of-00004.gguf \
    --threads -1 \
    --ctx-size 16384 \
    --n-gpu-layers 99 \
    -ot ".ffn_.*_exps.=CPU" \
    --temp 0.7 \
    --min-p 0.0 \
    --top-p 0.8 \
    --top-k 20 \
    --repeat-penalty 1.05
```
{% endcode %}

{% hint style="success" %}
Also don't forget about the new Qwen3 update. Run [**Qwen3-235B-A22B-Instruct-2507**](qwen3-how-to-run-and-fine-tune/qwen3-2507) locally with llama.cpp.
{% endhint %}

#### :tools: Improving generation speed

If you have more VRAM, you can try offloading more MoE layers, or offloading whole layers themselves.

Normally, `-ot ".ffn_.*_exps.=CPU"`  offloads all MoE layers to the CPU! This effectively allows you to fit all non MoE layers on 1 GPU, improving generation speeds. You can customize the regex expression to fit more layers if you have more GPU capacity.

If you have a bit more GPU memory, try `-ot ".ffn_(up|down)_exps.=CPU"` This offloads up and down projection MoE layers.

Try `-ot ".ffn_(up)_exps.=CPU"` if you have even more GPU memory. This offloads only up projection MoE layers.

You can also customize the regex, for example `-ot "\.(6|7|8|9|[0-9][0-9]|[0-9][0-9][0-9])\.ffn_(gate|up|down)_exps.=CPU"` means to offload gate, up and down MoE layers but only from the 6th layer onwards.

The [latest llama.cpp release](https://github.com/ggml-org/llama.cpp/pull/14363) also introduces high throughput mode. Use `llama-parallel`. Read more about it [here](https://github.com/ggml-org/llama.cpp/tree/master/examples/parallel). You can also **quantize the KV cache to 4bits** for example to reduce VRAM / RAM movement, which can also make the generation process faster.

#### :triangular\_ruler:How to fit long context (256K to 1M)

To fit longer context, you can use <mark style="background-color:green;">**KV cache quantization**</mark> to quantize the K and V caches to lower bits. This can also increase generation speed due to reduced RAM / VRAM data movement. The allowed options for K quantization (default is `f16`) include the below.

`--cache-type-k f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1`&#x20;

You should use the `_1` variants for somewhat increased accuracy, albeit it's slightly slower. For eg `q4_1, q5_1`&#x20;

You can also quantize the V cache, but you will need to <mark style="background-color:yellow;">**compile llama.cpp with Flash Attention**</mark> support via `-DGGML_CUDA_FA_ALL_QUANTS=ON`, and use `--flash-attn` to enable it.

We also uploaded 1 million context length GGUFs via YaRN scaling [here](https://app.gitbook.com/o/HpyELzcNe0topgVLGCZY/s/xhOjnexMCB3dmuQFQ2Zq/).

## :toolbox: Tool Calling Fixes

We managed to fix tool calling via `llama.cpp --jinja` specifically for serving through `llama-server`! If you’re downloading our 30B-A3B quants, no need to worry as these already include our fixes. For the 480B-A35B model, please:

1. Download the first file at https://huggingface.co/unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF/tree/main/UD-Q2\_K\_XL for UD-Q2\_K\_XL, and replace your current file
2. Use `snapshot_download` as usual as in https://docs.unsloth.ai/basics/qwen3-coder-how-to-run-locally#llama.cpp-run-qwen3-tutorial which will auto override the old files
3. Use the new chat template via `--chat-template-file`. See [GGUF chat template](https://huggingface.co/unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF?chat_template=default) or [chat\_template.jinja](https://huggingface.co/unsloth/Qwen3-Coder-480B-A35B-Instruct/raw/main/chat_template.jinja)
4. As an extra, we also made 1 single 150GB UD-IQ1\_M file (so Ollama works) at https://huggingface.co/unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF/blob/main/Qwen3-Coder-480B-A35B-Instruct-UD-IQ1\_M.gguf

This should solve issues like: https://github.com/ggml-org/llama.cpp/issues/14915

### Using Tool Calling

To format the prompts for tool calling, let's showcase it with an example.

I created a Python function called `get_current_temperature` which is a function which should get the current temperature for a location. For now we created a placeholder function which will always return 21.6 degrees celsius. You should change this to a true function!!

{% code overflow="wrap" %}
```python
def get_current_temperature(location: str, unit: str = "celsius"):
    """Get current temperature at a location.

    Args:
        location: The location to get the temperature for, in the format "City, State, Country".
        unit: The unit to return the temperature in. Defaults to "celsius". (choices: ["celsius", "fahrenheit"])

    Returns:
        the temperature, the location, and the unit in a dict
    """
    return {
        "temperature": 26.1, # PRE_CONFIGURED -> you change this!
        "location": location,
        "unit": unit,
    }
```
{% endcode %}

Then use the tokenizer to create the entire prompt:

{% code overflow="wrap" %}
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen3-Coder-480B-A35B-Instruct")

messages = [
    {'role': 'user', 'content': "What's the temperature in San Francisco now? How about tomorrow?"},
    {'content': "", 'role': 'assistant', 'function_call': None, 'tool_calls': [
        {'id': 'ID', 'function': {'arguments': {"location": "San Francisco, CA, USA"}, 'name': 'get_current_temperature'}, 'type': 'function'},
    ]},
    {'role': 'tool', 'content': '{"temperature": 26.1, "location": "San Francisco, CA, USA", "unit": "celsius"}', 'tool_call_id': 'ID'},
]

prompt = tokenizer.apply_chat_template(messages, tokenize = False)
```
{% endcode %}

## :bulb:Performance Benchmarks

{% hint style="info" %}
These official benchmarks are for the full BF16 checkpoint. To use this, simply use the `Q8_K_XL, Q8_0, BF16` checkpoints we uploaded - you can still use the tricks like MoE offloading for these versions as well!
{% endhint %}

Here are the benchmarks for the 480B model:

#### Agentic Coding

<table data-full-width="true"><thead><tr><th>Benchmark</th><th>Qwen3‑Coder 40B‑A35B‑Instruct</th><th>Kimi‑K2</th><th>DeepSeek‑V3-0324</th><th>Claude 4 Sonnet</th><th>GPT‑4.1</th></tr></thead><tbody><tr><td>Terminal‑Bench</td><td><strong>37.5</strong></td><td>30.0</td><td>2.5</td><td>35.5</td><td>25.3</td></tr><tr><td>SWE‑bench Verified w/ OpenHands (500 turns)</td><td><strong>69.6</strong></td><td>–</td><td>–</td><td>70.4</td><td>–</td></tr><tr><td>SWE‑bench Verified w/ OpenHands (100 turns)</td><td><strong>67.0</strong></td><td>65.4</td><td>38.8</td><td>68.0</td><td>48.6</td></tr><tr><td>SWE‑bench Verified w/ Private Scaffolding</td><td>–</td><td>65.8</td><td>–</td><td>72.7</td><td>63.8</td></tr><tr><td>SWE‑bench Live</td><td><strong>26.3</strong></td><td>22.3</td><td>13.0</td><td>27.7</td><td>–</td></tr><tr><td>SWE‑bench Multilingual</td><td><strong>54.7</strong></td><td>47.3</td><td>13.0</td><td>53.3</td><td>31.5</td></tr><tr><td>Multi‑SWE‑bench mini</td><td><strong>25.8</strong></td><td>19.8</td><td>7.5</td><td>24.8</td><td>–</td></tr><tr><td>Multi‑SWE‑bench flash</td><td><strong>27.0</strong></td><td>20.7</td><td>–</td><td>25.0</td><td>–</td></tr><tr><td>Aider‑Polyglot</td><td><strong>61.8</strong></td><td>60.0</td><td>56.9</td><td>56.4</td><td>52.4</td></tr><tr><td>Spider2</td><td><strong>31.1</strong></td><td>25.2</td><td>12.8</td><td>31.1</td><td>16.5</td></tr></tbody></table>

#### Agentic Browser Use

<table data-full-width="true"><thead><tr><th>Benchmark</th><th>Qwen3‑Coder 40B‑A35B‑Instruct</th><th>Kimi‑K2</th><th>DeepSeek‑V3 0324</th><th>Claude Sonnet‑4</th><th>GPT‑4.1</th></tr></thead><tbody><tr><td>WebArena</td><td><strong>49.9</strong></td><td>47.4</td><td>40.0</td><td>51.1</td><td>44.3</td></tr><tr><td>Mind2Web</td><td><strong>55.8</strong></td><td>42.7</td><td>36.0</td><td>47.4</td><td>49.6</td></tr></tbody></table>

#### Agentic Tool -Use

<table data-full-width="true"><thead><tr><th>Benchmark</th><th>Qwen3‑Coder 40B‑A35B‑Instruct</th><th>Kimi‑K2</th><th>DeepSeek‑V3 0324</th><th>Claude Sonnet‑4</th><th>GPT‑4.1</th></tr></thead><tbody><tr><td>BFCL‑v3</td><td><strong>68.7</strong></td><td>65.2</td><td>56.9</td><td>73.3</td><td>62.9</td></tr><tr><td>TAU‑Bench Retail</td><td><strong>77.5</strong></td><td>70.7</td><td>59.1</td><td>80.5</td><td>–</td></tr><tr><td>TAU‑Bench Airline</td><td><strong>60.0</strong></td><td>53.5</td><td>40.0</td><td>60.0</td><td>–</td></tr></tbody></table>

---
description: Learn to run & fine-tune Qwen3 locally with Unsloth + our Dynamic 2.0 quants
---

# Qwen3: How to Run & Fine-tune

Qwen's new Qwen3 models deliver state-of-the-art advancements in reasoning, instruction-following, agent capabilities, and multilingual support.

{% hint style="success" %}
**NEW!** Qwen3 got an update in July 2025. Run & fine-tune the latest model: [**Qwen-2507**](qwen3-how-to-run-and-fine-tune/qwen3-2507)
{% endhint %}

All uploads use Unsloth [Dynamic 2.0](unsloth-dynamic-2.0-ggufs) for SOTA 5-shot MMLU and KL Divergence performance, meaning you can run & fine-tune quantized Qwen LLMs with minimal accuracy loss.

We also uploaded Qwen3 with native 128K context length. Qwen achieves this by using YaRN to extend its original 40K window to 128K.

[Unsloth](https://github.com/unslothai/unsloth) also now supports fine-tuning and [Reinforcement Learning (RL)](reinforcement-learning-rl-guide) of Qwen3 and Qwen3 MOE models — 2x faster, with 70% less VRAM, and 8x longer context lengths. Fine-tune Qwen3 (14B) for free using our [Colab notebook.](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_\(14B\)-Reasoning-Conversational.ipynb)

<a href="#running-qwen3" class="button primary">Running Qwen3 Tutorial</a> <a href="#fine-tuning-qwen3-with-unsloth" class="button secondary">Fine-tuning Qwen3</a>

#### **Qwen3 - Unsloth Dynamic 2.0** with optimal configs:

| Dynamic 2.0 GGUF (to run)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | 128K Context GGUF                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | Dynamic 4-bit Safetensor (to finetune/deploy)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <ul><li><a href="https://huggingface.co/unsloth/Qwen3-0.6B-GGUF">0.6B</a></li><li><a href="https://huggingface.co/unsloth/Qwen3-1.7B-GGUF">1.7B</a></li><li><a href="https://huggingface.co/unsloth/Qwen3-4B-GGUF">4B</a></li><li><a href="https://huggingface.co/unsloth/Qwen3-8B-GGUF">8B</a></li><li><a href="https://huggingface.co/unsloth/Qwen3-14B-GGUF">14B</a></li><li><a href="https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF">30B-A3B</a></li><li><a href="https://huggingface.co/unsloth/Qwen3-32B-GGUF">32B</a></li><li><a href="https://huggingface.co/unsloth/Qwen3-235B-A22B-GGUF">235B-A22B</a></li></ul> | <ul><li><a href="https://huggingface.co/unsloth/Qwen3-4B-128K-GGUF">4B</a></li></ul><ul><li><a href="https://huggingface.co/unsloth/Qwen3-8B-128K-GGUF">8B</a></li></ul><ul><li><a href="https://huggingface.co/unsloth/Qwen3-14B-128K-GGUF">14B</a></li></ul><ul><li><a href="https://huggingface.co/unsloth/Qwen3-30B-A3B-128K-GGUF">30B-A3B</a></li></ul><ul><li><a href="https://huggingface.co/unsloth/Qwen3-32B-128K-GGUF">32B</a></li></ul><ul><li><a href="https://huggingface.co/unsloth/Qwen3-235B-A22B-128K-GGUF">235B-A22B</a></li></ul> | <ul><li><a href="https://huggingface.co/unsloth/Qwen3-0.6B-unsloth-bnb-4bit">0.6B</a></li></ul><ul><li><a href="https://huggingface.co/unsloth/Qwen3-1.7B-unsloth-bnb-4bit">1.7B</a></li><li><a href="https://huggingface.co/unsloth/Qwen3-4B-unsloth-bnb-4bit">4B</a></li><li><a href="https://huggingface.co/unsloth/Qwen3-8B-unsloth-bnb-4bit">8B</a></li><li><a href="https://huggingface.co/unsloth/Qwen3-14B-unsloth-bnb-4bit">14B</a></li><li><a href="https://huggingface.co/unsloth/Qwen3-30B-A3B-bnb-4bit">30B-A3B</a></li></ul><ul><li><a href="https://huggingface.co/unsloth/Qwen3-32B-unsloth-bnb-4bit">32B</a></li></ul> |

## 🖥️ **Running Qwen3**

To achieve inference speeds of 6+ tokens per second, we recommend your available memory should match or exceed the size of the model you’re using. For example, a 30GB 1-bit quantized model requires at least 150GB of memory. The Q2\_K\_XL quant, which is 180GB, will require at least **180GB of unified memory** (VRAM + RAM) or **180GB of RAM** for optimal performance.

**NOTE:** It’s possible to run the model with **less total memory** than its size (i.e., less VRAM, less RAM, or a lower combined total). However, this will result in slower inference speeds. Sufficient memory is only required if you want to maximize throughput and achieve the fastest inference times.

### :gear: Official Recommended Settings

According to Qwen, these are the recommended settings for inference:

| Non-Thinking Mode Settings:                                            | Thinking Mode Settings:                                           |
| ---------------------------------------------------------------------- | ----------------------------------------------------------------- |
| <mark style="background-color:blue;">**Temperature = 0.7**</mark>      | <mark style="background-color:blue;">**Temperature = 0.6**</mark> |
| Min\_P = 0.0 (optional, but 0.01 works well, llama.cpp default is 0.1) | Min\_P = 0.0                                                      |
| Top\_P = 0.8                                                           | Top\_P = 0.95                                                     |
| TopK = 20                                                              | TopK = 20                                                         |

**Chat template/prompt format:**&#x20;

{% code overflow="wrap" %}
```
<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n
```
{% endcode %}

{% hint style="success" %}
For NON thinking mode, we purposely enclose \<think> and \</think> with nothing:
{% endhint %}

{% code overflow="wrap" %}
```
<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
```
{% endcode %}

{% hint style="warning" %}
**For Thinking-mode, DO NOT use greedy decoding**, as it can lead to performance degradation and endless repetitions.
{% endhint %}

### Switching Between Thinking and Non-Thinking Mode

Qwen3 models come with built-in "thinking mode" to boost reasoning and improve response quality - similar to how [QwQ-32B](tutorials-how-to-fine-tune-and-run-llms/qwq-32b-how-to-run-effectively) worked. Instructions for switching will differ depending on the inference engine you're using so ensure you use the correct instructions.

#### Instructions for llama.cpp and Ollama:

You can add `/think` and `/no_think` to user prompts or system messages to switch the model's thinking mode from turn to turn. The model will follow the most recent instruction in multi-turn conversations.

Here is an example of multi-turn conversation:

```
> Who are you /no_think

<think>

</think>

I am Qwen, a large-scale language model developed by Alibaba Cloud. [...]

> How many 'r's are in 'strawberries'? /think

<think>
Okay, let's see. The user is asking how many times the letter 'r' appears in the word "strawberries". [...]
</think>

The word strawberries contains 3 instances of the letter r. [...]
```

#### Instructions for transformers and vLLM:

**Thinking mode:**

`enable_thinking=True`

By default, Qwen3 has thinking enabled. When you call `tokenizer.apply_chat_template`, you **don’t need to set anything manually.**

```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True  # Default is True
)
```

In thinking mode, the model will generate an extra `<think>...</think>` block before the final answer — this lets it "plan" and sharpen its responses.

**Non-thinking mode:**

`enable_thinking=False`

Enabling non-thinking will make Qwen3 will skip all the thinking steps and behave like a normal LLM.

```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False  # Disables thinking mode
)
```

This mode will provide final responses directly — no `<think>` blocks, no chain-of-thought.

### 🦙 Ollama: Run Qwen3 Tutorial

1. Install `ollama` if you haven't already! You can only run models up to 32B in size. To run the full 235B-A22B model, [see here](#running-qwen3-235b-a22b).

```bash
apt-get update
apt-get install pciutils -y
curl -fsSL https://ollama.com/install.sh | sh
```

2. Run the model! Note you can call `ollama serve`in another terminal if it fails! We include all our fixes and suggested parameters (temperature etc) in `params` in our Hugging Face upload!

```bash
ollama run hf.co/unsloth/Qwen3-8B-GGUF:UD-Q4_K_XL
```

3. To disable thinking, use (or you can set it in the system prompt):&#x20;

```
>>> Write your prompt here /nothink
```

{% hint style="warning" %}
If you're experiencing any looping, Ollama might have set your context length window to 2,048 or so. If this is the case, bump it up to 32,000 and see if the issue still persists.
{% endhint %}

### 📖 Llama.cpp: Run Qwen3 Tutorial

1. Obtain the latest `llama.cpp` on [GitHub here](https://github.com/ggml-org/llama.cpp). You can follow the build instructions below as well. Change `-DGGML_CUDA=ON` to `-DGGML_CUDA=OFF` if you don't have a GPU or just want CPU inference.

```bash
apt-get update
apt-get install pciutils build-essential cmake curl libcurl4-openssl-dev -y
git clone https://github.com/ggml-org/llama.cpp
cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON
cmake --build llama.cpp/build --config Release -j --clean-first --target llama-cli llama-gguf-split
cp llama.cpp/build/bin/llama-* llama.cpp
```

2. Download the model via (after installing `pip install huggingface_hub hf_transfer` ). You can choose Q4\_K\_M, or other quantized versions.

```python
# !pip install huggingface_hub hf_transfer
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id = "unsloth/Qwen3-14B-GGUF",
    local_dir = "unsloth/Qwen3-14B-GGUF",
    allow_patterns = ["*UD-Q4_K_XL*"],
)
```

3. Run the model and try any prompt. To disable thinking, use (or you can set it in the system prompt):

```
>>> Write your prompt here /nothink
```

### Running Qwen3-235B-A22B

For Qwen3-235B-A22B, we will specifically use Llama.cpp for optimized inference and a plethora of options.

1. We're following similar steps to above however this time we'll also need to perform extra steps because the model is so big.
2.  Download the model via (after installing `pip install huggingface_hub hf_transfer` ). You can choose UD-Q2\_K\_XL, or other quantized versions..

    ```python
    # !pip install huggingface_hub hf_transfer
    import os
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id = "unsloth/Qwen3-235B-A22B-GGUF",
        local_dir = "unsloth/Qwen3-235B-A22B-GGUF",
        allow_patterns = ["*UD-Q2_K_XL*"],
    )
    ```


3. Run the model and try any prompt.
4. Edit `--threads 32` for the number of CPU threads, `--ctx-size 16384` for context length, `--n-gpu-layers 99` for GPU offloading on how many layers. Try adjusting it if your GPU goes out of memory. Also remove it if you have CPU only inference.

{% hint style="success" %}
Use `-ot ".ffn_.*_exps.=CPU"` to offload all MoE layers to the CPU! This effectively allows you to fit all non MoE layers on 1  GPU, improving generation speeds. You can customize the regex expression to fit more layers if you have more GPU capacity.
{% endhint %}

{% code overflow="wrap" %}
```bash
./llama.cpp/llama-cli \
    --model unsloth/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-UD-Q2_K_XL.gguf \
    --threads 32 \
    --ctx-size 16384 \
    --n-gpu-layers 99 \
    -ot ".ffn_.*_exps.=CPU" \
    --seed 3407 \
    --prio 3 \
    --temp 0.6 \
    --min-p 0.0 \
    --top-p 0.95 \
    --top-k 20 \
    -no-cnv \
    --prompt "<|im_start|>user\nCreate a Flappy Bird game in Python. You must include these things:\n1. You must use pygame.\n2. The background color should be randomly chosen and is a light shade. Start with a light blue color.\n3. Pressing SPACE multiple times will accelerate the bird.\n4. The bird's shape should be randomly chosen as a square, circle or triangle. The color should be randomly chosen as a dark color.\n5. Place on the bottom some land colored as dark brown or yellow chosen randomly.\n6. Make a score shown on the top right side. Increment if you pass pipes and don't hit them.\n7. Make randomly spaced pipes with enough space. Color them randomly as dark green or light brown or a dark gray shade.\n8. When you lose, show the best score. Make the text inside the screen. Pressing q or Esc will quit the game. Restarting is pressing SPACE again.\nThe final game should be inside a markdown section in Python. Check your code for errors and fix them before the final markdown section.<|im_end|>\n<|im_start|>assistant\n"
```
{% endcode %}

## 🦥 Fine-tuning Qwen3 with Unsloth

Unsloth makes Qwen3 fine-tuning 2x faster, use 70% less VRAM and supports 8x longer context lengths.  Qwen3 (14B) fits comfortably in a Google Colab 16GB VRAM Tesla T4 GPU.

Because Qwen3 supports both reasoning and non-reasoning, you can fine-tune it with a non-reasoning dataset, but this may affect its reasoning ability. If you want to maintain its reasoning capabilities (optional), you can use a mix of direct answers and chain-of-thought examples. Use <mark style="background-color:green;">75% reasoning</mark> and <mark style="background-color:green;">25% non-reasoning</mark> in your dataset to make the model retain its reasoning capabilities.

Our Conversational notebook uses a combo of 75% NVIDIA’s open-math-reasoning dataset and 25% Maxime’s FineTome dataset (non-reasoning). Here's free Unsloth Colab notebooks to fine-tune Qwen3:

* [Qwen3 (14B) Reasoning + Conversational notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_\(14B\)-Reasoning-Conversational.ipynb) (recommended)
* [**Qwen3 (4B)**](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_\(4B\)-GRPO.ipynb) **- Advanced GRPO LoRA**
* [Qwen3 (14B) Alpaca notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_\(14B\)-Alpaca.ipynb) (for Base models)

If you have an old version of Unsloth and/or are fine-tuning locally, install the latest version of Unsloth:

```
pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo
```

### Qwen3 MOE models fine-tuning

Fine-tuning support includes MOE models: 30B-A3B and 235B-A22B. Qwen3-30B-A3B works on just 17.5GB VRAM with Unsloth. On fine-tuning MoE's - it's probably not a good idea to fine-tune the router layer so we disabled it by default.

The 30B-A3B fits in 17.5GB VRAM, but you may lack RAM or disk space since the full 16-bit model must be downloaded and converted to 4-bit on the fly for QLoRA fine-tuning. This is due to issues importing 4-bit BnB MOE models directly. This only affects MOE models.

{% hint style="warning" %}
If you're fine-tuning the MOE models, please use `FastModel` and not `FastLanguageModel`
{% endhint %}

```python
from unsloth import FastModel
import torch
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/Qwen3-30B-A3B",
    max_seq_length = 2048, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)
```

### Notebook Guide:

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FFQX2CBzUqzAIMM50bpM4%2Fimage.png?alt=media&#x26;token=23c4b3d5-0d5f-4906-b2b4-bacde23235e0" alt=""><figcaption></figcaption></figure>

To use the notebooks, just click Runtime, then Run all. You can change settings in the notebook to whatever you desire. We have set them automatically by default. Change model name to whatever you like by matching it with model's name on Hugging Face e.g. 'unsloth/Qwen3-8B' or 'unsloth/Qwen3-0.6B-unsloth-bnb-4bit'.

There are other settings which you can toggle:

* **`max_seq_length = 2048`** – Controls context length. While Qwen3 supports 40960, we recommend 2048 for testing. Unsloth enables 8× longer context fine-tuning.
* **`load_in_4bit = True`** – Enables 4-bit quantization, reducing memory use 4× for fine-tuning on 16GB GPUs.
* For **full-finetuning** - set `full_finetuning = True`  and **8-bit finetuning** - set `load_in_8bit = True`&#x20;

If you'd like to read a full end-to-end guide on how to use Unsloth notebooks for fine-tuning or just learn about fine-tuning, creating [datasets](datasets-guide) etc., view our [complete guide here](../get-started/fine-tuning-llms-guide):

{% content-ref url="../get-started/fine-tuning-llms-guide" %}
[fine-tuning-llms-guide](../get-started/fine-tuning-llms-guide)
{% endcontent-ref %}

{% content-ref url="datasets-guide" %}
[datasets-guide](datasets-guide)
{% endcontent-ref %}

### GRPO with Qwen3

We made a new advanced GRPO notebook for fine-tuning Qwen3. Learn to use our new proximity-based reward function (closer answers = rewarded) and Hugging Face's Open-R1 math dataset. \
Unsloth now also has better evaluations and uses the latest version of vLLM.

[**Qwen3 (4B)**](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_\(4B\)-GRPO.ipynb) **notebook - Advanced GRPO LoRA**

Learn about:

* Enabling reasoning in Qwen3 (Base)+ guiding it to do a specific task
* Pre-finetuning to bypass GRPO's tendency to learn formatting
* Improved evaluation accuracy via new regex matching
* Custom GRPO templates beyond just 'think' e.g. \<start\_working\_out>\</end\_working\_out>
* Proximity-based scoring: better answers earn more points (e.g., predicting 9 when the answer is 10) and outliers are penalized

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FMUjDPzhhjMJXcljIhgbK%2Fqwen33%20mascot.png?alt=media&#x26;token=fcfa1104-8f6d-4f04-b72d-b9c085d3ecda" alt=""><figcaption></figcaption></figure>


---
description: >-
  Run Qwen3-30B-A3B-2507 and 235B-A22B Thinking and Instruct versions locally on
  your device!
---

# Qwen3-2507

Qwen released 2507 (July 2025) updates for their [Qwen3]() 4B, 30B and 235B models, introducing both "thinking" and "non-thinking" variants. The non-thinking '**Qwen3-30B-A3B-Instruct-2507**' and '**Qwen3-235B-A22B-Instruct-2507'** features a 256K context window, improved instruction following, multilingual capabilities and alignment.

The thinking models '**Qwen3-30B-A3B-Thinking-2507**' and '**Qwen3-235B-A22B-Thinking-2507**' excel at reasoning, with the 235B achieving SOTA results in logic, math, science, coding, and advanced academic tasks.

[Unsloth](https://github.com/unslothai/unsloth) also now supports fine-tuning and [Reinforcement Learning (RL)](../reinforcement-learning-rl-guide) of Qwen3-2507 models — 2x faster, with 70% less VRAM, and 8x longer context lengths

<a href="#run-qwen3-30b-a3b-2507-tutorials" class="button secondary">Run 30B-A3B</a><a href="#run-qwen3-235b-a22b-thinking-2507" class="button secondary">Run 235B-A22B</a><a href="#fine-tuning-qwen3-2507-with-unsloth" class="button secondary">Fine-tune Qwen3-2507</a>

**Unsloth** [**Dynamic 2.0**](../unsloth-dynamic-2.0-ggufs) **GGUFs:**

| Model                    | GGUFs to run:                                                                                                                                                 |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Qwen3-**4B-2507**        | [Instruct](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF) • [Thinking ](https://huggingface.co/unsloth/Qwen3-4B-Thinking-2507-GGUF)              |
| Qwen3-**30B-A3B**-2507   | [Instruct](#llama.cpp-run-qwen3-30b-a3b-instruct-2507-tutorial) • [Thinking](https://huggingface.co/unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF)                 |
| Qwen3-**235B-A22B**-2507 | [Instruct](https://huggingface.co/unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF) • [Thinking](https://huggingface.co/unsloth/Qwen3-235B-A22B-Thinking-2507-GGUF) |

## ⚙️Best Practices

{% hint style="success" %}
The settings for the Thinking and Instruct model are different.\
The thinking model uses temperature = 0.6, but the instruct model uses temperature = 0.7\
The thinking model uses top\_p = 0.95, but the instruct model uses top\_p = 0.8
{% endhint %}

To achieve optimal performance, Qwen recommends these settings:

| Instruct Model Settings:                                                                                      | Thinking Model Settings:                                                                                      |
| ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| <mark style="background-color:blue;">`Temperature = 0.7`</mark>                                               | <mark style="background-color:blue;">`Temperature = 0.6`</mark>                                               |
| `Min_P = 0.00`  (llama.cpp's default is 0.1)                                                                  | `Min_P = 0.00` (llama.cpp's default is 0.1)                                                                   |
| `Top_P = 0.80`                                                                                                | `Top_P = 0.95`                                                                                                |
| `TopK = 20`                                                                                                   | `TopK = 20`                                                                                                   |
| `presence_penalty = 0.0 to 2.0` (llama.cpp default turns it off, but to reduce repetitions, you can use this) | `presence_penalty = 0.0 to 2.0` (llama.cpp default turns it off, but to reduce repetitions, you can use this) |

**Adequate Output Length**: Use an output length of `32,768` tokens for most queries, which is adequate for most queries.

Chat template for both Thinking (thinking has `<think></think>`) and Instruct is below:

```
<|im_start|>user
Hey there!<|im_end|>
<|im_start|>assistant
What is 1+1?<|im_end|>
<|im_start|>user
2<|im_end|>
<|im_start|>assistant
```

## 📖 Run Qwen3-30B-A3B-2507 Tutorials

Below are guides for the [Thinking](#thinking-qwen3-30b-a3b-thinking-2507) and [Instruct](#instruct-qwen3-30b-a3b-instruct-2507) versions of the model.

### Instruct: Qwen3-30B-A3B-Instruct-2507

Given that this is a non thinking model, there is no need to set `thinking=False` and the model does not generate `<think> </think>` blocks.

#### ⚙️Best Practices

To achieve optimal performance, Qwen recommends the following settings:

* &#x20;We suggest using `temperature=0.7, top_p=0.8, top_k=20, and min_p=0.0` `presence_penalty` between 0 and 2 if the framework supports to reduce endless repetitions.
* <mark style="background-color:$success;">**`temperature = 0.7`**</mark>
* `top_k = 20`
* `min_p = 0.00` (llama.cpp's default is 0.1)
* **`top_p = 0.80`**
* `presence_penalty = 0.0 to 2.0` (llama.cpp default turns it off, but to reduce repetitions, you can use this) Try 1.0 for example.
* Supports up to `262,144` context natively but you can set it to `32,768` tokens for less RAM use

#### 🦙 Ollama: Run Qwen3-30B-A3B-Instruct-2507 Tutorial

1. Install `ollama` if you haven't already! You can only run models up to 32B in size.

```bash
apt-get update
apt-get install pciutils -y
curl -fsSL https://ollama.com/install.sh | sh
```

2. Run the model! Note you can call `ollama serve`in another terminal if it fails! We include all our fixes and suggested parameters (temperature etc) in `params` in our Hugging Face upload!

```bash
ollama run hf.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:UD-Q4_K_XL
```

#### :sparkles: Llama.cpp: Run Qwen3-30B-A3B-Instruct-2507 Tutorial

1. Obtain the latest `llama.cpp` on [GitHub here](https://github.com/ggml-org/llama.cpp). You can follow the build instructions below as well. Change `-DGGML_CUDA=ON` to `-DGGML_CUDA=OFF` if you don't have a GPU or just want CPU inference.

```bash
apt-get update
apt-get install pciutils build-essential cmake curl libcurl4-openssl-dev -y
git clone https://github.com/ggml-org/llama.cpp
cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON
cmake --build llama.cpp/build --config Release -j --clean-first --target llama-cli llama-gguf-split
cp llama.cpp/build/bin/llama-* llama.cpp
```

2.  You can directly pull from HuggingFace via:

    ```
    ./llama.cpp/llama-cli \
        -hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_XL \
        --jinja -ngl 99 --threads -1 --ctx-size 32684 \
        --temp 0.7 --min-p 0.0 --top-p 0.80 --top-k 20 --presence-penalty 1.0
    ```
3. Download the model via (after installing `pip install huggingface_hub hf_transfer` ). You can choose UD\_Q4\_K\_XL or other quantized versions.

```python
# !pip install huggingface_hub hf_transfer
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id = "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
    local_dir = "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
    allow_patterns = ["*UD-Q4_K_XL*"],
)
```

### Thinking: Qwen3-30B-A3B-Thinking-2507

This model supports only thinking mode and a 256K context window natively. The default chat template adds `<think>` automatically, so you may see only a closing `</think>` tag in the output.

#### ⚙️Best Practices

To achieve optimal performance, Qwen recommends the following settings:

* &#x20;We suggest using `temperature=0.6, top_p=0.95, top_k=20, and min_p=0.0` `presence_penalty` between 0 and 2 if the framework supports to reduce endless repetitions.
* <mark style="background-color:$success;">**`temperature = 0.6`**</mark>
* `top_k = 20`
* `min_p = 0.00` (llama.cpp's default is 0.1)
* **`top_p = 0.95`**
* `presence_penalty = 0.0 to 2.0` (llama.cpp default turns it off, but to reduce repetitions, you can use this) Try 1.0 for example.
* Supports up to `262,144` context natively but you can set it to `32,768` tokens for less RAM use

#### 🦙 Ollama: Run Qwen3-30B-A3B-Instruct-2507 Tutorial

1. Install `ollama` if you haven't already! You can only run models up to 32B in size. To run the full 235B-A22B models, [see here](#run-qwen3-235b-a22b-instruct-2507).

```bash
apt-get update
apt-get install pciutils -y
curl -fsSL https://ollama.com/install.sh | sh
```

2. Run the model! Note you can call `ollama serve`in another terminal if it fails! We include all our fixes and suggested parameters (temperature etc) in `params` in our Hugging Face upload!

```bash
ollama run hf.co/unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF:UD-Q4_K_XL
```

#### :sparkles: Llama.cpp: Run Qwen3-30B-A3B-Instruct-2507 Tutorial

1. Obtain the latest `llama.cpp` on [GitHub here](https://github.com/ggml-org/llama.cpp). You can follow the build instructions below as well. Change `-DGGML_CUDA=ON` to `-DGGML_CUDA=OFF` if you don't have a GPU or just want CPU inference.

```bash
apt-get update
apt-get install pciutils build-essential cmake curl libcurl4-openssl-dev -y
git clone https://github.com/ggml-org/llama.cpp
cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON
cmake --build llama.cpp/build --config Release -j --clean-first --target llama-cli llama-gguf-split
cp llama.cpp/build/bin/llama-* llama.cpp
```

2.  You can directly pull from Hugging Face via:

    ```
    ./llama.cpp/llama-cli \
        -hf unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF:Q4_K_XL \
        --jinja -ngl 99 --threads -1 --ctx-size 32684 \
        --temp 0.6 --min-p 0.0 --top-p 0.95 --top-k 20 --presence-penalty 1.0
    ```
3. Download the model via (after installing `pip install huggingface_hub hf_transfer` ). You can choose UD\_Q4\_K\_XL or other quantized versions.

```python
# !pip install huggingface_hub hf_transfer
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id = "unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF",
    local_dir = "unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF",
    allow_patterns = ["*UD-Q4_K_XL*"],
)
```

## 📖 Run **Qwen3-235B-A22B-2507** Tutorials

Below are guides for the [Thinking](#run-qwen3-235b-a22b-thinking-via-llama.cpp) and [Instruct](#run-qwen3-235b-a22b-instruct-via-llama.cpp) versions of the model.

### Thinking: Qwen3-**235B-A22B**-Thinking-2507

This model supports only thinking mode and a 256K context window natively. The default chat template adds `<think>` automatically, so you may see only a closing `</think>` tag in the output.

#### :gear: Best Practices

To achieve optimal performance, Qwen recommends these settings for the Thinking model:

* <mark style="background-color:$success;">**`temperature = 0.6`**</mark>
* `top_k = 20`
* `min_p = 0.00` (llama.cpp's default is 0.1)
* `top_p = 0.95`
* `presence_penalty = 0.0 to 2.0` (llama.cpp default turns it off, but to reduce repetitions, you can use this) Try 1.0 for example.
* **Adequate Output Length**: Use an output length of `32,768` tokens for most queries, which is adequate for most queries.

#### :sparkles:Run Qwen3-235B-A22B-Thinking via llama.cpp:

For Qwen3-235B-A22B, we will specifically use Llama.cpp for optimized inference and a plethora of options.

{% hint style="success" %}
If you want a **full precision unquantized version**, use our `Q8_K_XL, Q8_0` or `BF16` versions!
{% endhint %}

1.  Obtain the latest `llama.cpp` on [GitHub here](https://github.com/ggml-org/llama.cpp). You can follow the build instructions below as well. Change `-DGGML_CUDA=ON` to `-DGGML_CUDA=OFF` if you don't have a GPU or just want CPU inference.

    ```bash
    apt-get update
    apt-get install pciutils build-essential cmake curl libcurl4-openssl-dev -y
    git clone https://github.com/ggml-org/llama.cpp
    cmake llama.cpp -B llama.cpp/build \
        -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON
    cmake --build llama.cpp/build --config Release -j --clean-first --target llama-cli llama-gguf-split
    cp llama.cpp/build/bin/llama-* llama.cpp
    ```
2.  You can directly use llama.cpp to download the model but I normally suggest using `huggingface_hub` To use llama.cpp directly, do:

    ```
    ./llama.cpp/llama-cli \
        -hf unsloth/Qwen3-235B-A22B-Thinking-2507-GGUF:Q2_K_XL \
        --threads -1 \
        --ctx-size 16384 \
        --n-gpu-layers 99 \
        -ot ".ffn_.*_exps.=CPU" \
        --temp 0.6 \
        --min-p 0.0 \
        --top-p 0.95 \
        --top-k 20 \
        --presence-penalty 1.0
    ```
3.  Download the model via (after installing `pip install huggingface_hub hf_transfer` ). You can choose UD-Q2\_K\_XL, or other quantized versions..

    ```python
    # !pip install huggingface_hub hf_transfer
    import os
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0" # Can sometimes rate limit, so set to 0 to disable
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id = "unsloth/Qwen3-235B-A22B-Thinking-2507-GGUF",
        local_dir = "unsloth/Qwen3-235B-A22B-Thinking-2507-GGUF",
        allow_patterns = ["*UD-Q2_K_XL*"],
    )
    ```


4. Run the model and try any prompt.
5. Edit `--threads -1` for the number of CPU threads, `--ctx-size` 262114 for context length, `--n-gpu-layers 99` for GPU offloading on how many layers. Try adjusting it if your GPU goes out of memory. Also remove it if you have CPU only inference.

{% hint style="success" %}
Use `-ot ".ffn_.*_exps.=CPU"` to offload all MoE layers to the CPU! This effectively allows you to fit all non MoE layers on 1  GPU, improving generation speeds. You can customize the regex expression to fit more layers if you have more GPU capacity.
{% endhint %}

{% code overflow="wrap" %}
```bash
./llama.cpp/llama-cli \
    --model unsloth/Qwen3-235B-A22B-Thinking-2507-GGUF/UD-Q2_K_XL/Qwen3-235B-A22B-Thinking-2507-UD-Q2_K_XL-00001-of-00002.gguf \
    --threads -1 \
    --ctx-size 16384 \
    --n-gpu-layers 99 \
    -ot ".ffn_.*_exps.=CPU" \
    --seed 3407 \
    --temp 0.6 \
    --min-p 0.0 \
    --top-p 0.95 \
    --top-k 20
    --presence-penalty 1.0
```
{% endcode %}

### Instruct: Qwen3-**235B-A22B**-Instruct-2507

Given that this is a non thinking model, there is no need to set `thinking=False` and the model does not generate `<think> </think>` blocks.

#### ⚙️Best Practices

To achieve optimal performance, we recommend the following settings:

**1. Sampling Parameters**: We suggest using `temperature=0.7, top_p=0.8, top_k=20, and min_p=0.` `presence_penalty` between 0 and 2 if the framework supports to reduce endless repetitions.

2\. **Adequate Output Length**: We recommend using an output length of `16,384` tokens for most queries, which is adequate for instruct models.

3\. **Standardize Output Format:** We recommend using prompts to standardize model outputs when benchmarking.

* **Math Problems**: Include `Please reason step by step, and put your final answer within \boxed{}.` in the prompt.
* **Multiple-Choice Questions**: Add the following JSON structure to the prompt to standardize responses: "Please show your choice in the \`answer\` field with only the choice letter, e.g., \`"answer": "C".

#### :sparkles:Run Qwen3-235B-A22B-Instruct via llama.cpp:

For Qwen3-235B-A22B, we will specifically use Llama.cpp for optimized inference and a plethora of options.

{% hint style="success" %}
If you want a **full precision unquantized version**, use our `Q8_K_XL, Q8_0` or `BF16` versions!
{% endhint %}

1.  Obtain the latest `llama.cpp` on [GitHub here](https://github.com/ggml-org/llama.cpp). You can follow the build instructions below as well. Change `-DGGML_CUDA=ON` to `-DGGML_CUDA=OFF` if you don't have a GPU or just want CPU inference.

    ```bash
    apt-get update
    apt-get install pciutils build-essential cmake curl libcurl4-openssl-dev -y
    git clone https://github.com/ggml-org/llama.cpp
    cmake llama.cpp -B llama.cpp/build \
        -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON
    cmake --build llama.cpp/build --config Release -j --clean-first --target llama-cli llama-gguf-split
    cp llama.cpp/build/bin/llama-* llama.cpp
    ```
2.  You can directly use llama.cpp to download the model but I normally suggest using `huggingface_hub` To use llama.cpp directly, do:\


    ```
    ./llama.cpp/llama-cli \
        -hf unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF:Q2_K_XL \
        --threads -1 \
        --ctx-size 16384 \
        --n-gpu-layers 99 \
        -ot ".ffn_.*_exps.=CPU" \
        --temp 0.7 \
        --min-p 0.0 \
        --top-p 0.8 \
        --top-k 20 \
        --repeat-penalty 1.0
    ```
3.  Download the model via (after installing `pip install huggingface_hub hf_transfer` ). You can choose UD-Q2\_K\_XL, or other quantized versions..

    ```python
    # !pip install huggingface_hub hf_transfer
    import os
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0" # Can sometimes rate limit, so set to 0 to disable
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id = "unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF",
        local_dir = "unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF",
        allow_patterns = ["*UD-Q2_K_XL*"],
    )
    ```


4. Run the model and try any prompt.
5. Edit `--threads -1` for the number of CPU threads, `--ctx-size` 262114 for context length, `--n-gpu-layers 99` for GPU offloading on how many layers. Try adjusting it if your GPU goes out of memory. Also remove it if you have CPU only inference.

{% hint style="success" %}
Use `-ot ".ffn_.*_exps.=CPU"` to offload all MoE layers to the CPU! This effectively allows you to fit all non MoE layers on 1  GPU, improving generation speeds. You can customize the regex expression to fit more layers if you have more GPU capacity.
{% endhint %}

{% code overflow="wrap" %}
```bash
./llama.cpp/llama-cli \
    --model unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF/UD-Q2_K_XL/Qwen3-235B-A22B-Instruct-2507-UD-Q2_K_XL-00001-of-00002.gguf \
    --threads -1 \
    --ctx-size 16384 \
    --n-gpu-layers 99 \
    -ot ".ffn_.*_exps.=CPU" \
    --temp 0.7 \
    --min-p 0.0 \
    --top-p 0.8 \
    --top-k 20
```
{% endcode %}

### 🛠️ Improving generation speed <a href="#improving-generation-speed" id="improving-generation-speed"></a>

If you have more VRAM, you can try offloading more MoE layers, or offloading whole layers themselves.

Normally, `-ot ".ffn_.*_exps.=CPU"` offloads all MoE layers to the CPU! This effectively allows you to fit all non MoE layers on 1 GPU, improving generation speeds. You can customize the regex expression to fit more layers if you have more GPU capacity.

If you have a bit more GPU memory, try `-ot ".ffn_(up|down)_exps.=CPU"` This offloads up and down projection MoE layers.

Try `-ot ".ffn_(up)_exps.=CPU"` if you have even more GPU memory. This offloads only up projection MoE layers.

You can also customize the regex, for example `-ot "\.(6|7|8|9|[0-9][0-9]|[0-9][0-9][0-9])\.ffn_(gate|up|down)_exps.=CPU"` means to offload gate, up and down MoE layers but only from the 6th layer onwards.

The [latest llama.cpp release](https://github.com/ggml-org/llama.cpp/pull/14363) also introduces high throughput mode. Use `llama-parallel`. Read more about it [here](https://github.com/ggml-org/llama.cpp/tree/master/examples/parallel). You can also **quantize the KV cache to 4bits** for example to reduce VRAM / RAM movement, which can also make the generation process faster. The [next section](#how-to-fit-long-context-256k-to-1m) talks about KV cache quantization.

### 📐How to fit long context <a href="#how-to-fit-long-context-256k-to-1m" id="how-to-fit-long-context-256k-to-1m"></a>

To fit longer context, you can use **KV cache quantization** to quantize the K and V caches to lower bits. This can also increase generation speed due to reduced RAM / VRAM data movement. The allowed options for K quantization (default is `f16`) include the below.

`--cache-type-k f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1`

You should use the `_1` variants for somewhat increased accuracy, albeit it's slightly slower. For eg `q4_1, q5_1` So try out `--cache-type-k q4_1`

You can also quantize the V cache, but you will need to **compile llama.cpp with Flash Attention** support via `-DGGML_CUDA_FA_ALL_QUANTS=ON`, and use `--flash-attn` to enable it. After installing Flash Attention, you can then use `--cache-type-v q4_1`

## 🦥 Fine-tuning Qwen3-2507 with Unsloth

Unsloth makes [Qwen3](..#fine-tuning-qwen3-with-unsloth) and Qwen3-2507 fine-tuning 2x faster, use 70% less VRAM and supports 8x longer context lengths.  Because Qwen3-2507 was only released in a 30B variant, this means you will need about a 40GB A100 GPU to fine-tune the model using QLoRA (4-bit).

For a notebook, because the model cannot fit in Colab's free 16GB GPUs, you will need to utilize a 40GB A100. You can utilize our Conversational notebook but replace the dataset to any of your using. This time you do not need to combined reasoning in your dataset as the model has no reasoning.

* [Qwen3 (14B) Reasoning + Conversational notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_\(14B\)-Reasoning-Conversational.ipynb)&#x20;

If you have an old version of Unsloth and/or are fine-tuning locally, install the latest version of Unsloth:

```
pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo
```

### Qwen3-2507 MOE models fine-tuning

Fine-tuning support includes MOE models: 30B-A3B and 235B-A22B. Qwen3-30B-A3B works on 30GB VRAM with Unsloth. On fine-tuning MoE's - it's probably not a good idea to fine-tune the router layer so we disabled it by default.

The 30B-A3B fits in 30GB VRAM, but you may lack RAM or disk space since the full 16-bit model must be downloaded and converted to 4-bit on the fly for QLoRA fine-tuning. This is due to issues importing 4-bit BnB MOE models directly. This only affects MOE models.

{% hint style="warning" %}
If you're fine-tuning the MOE models, please use `FastModel` and not `FastLanguageModel`
{% endhint %}

```python
from unsloth import FastModel
import torch
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/Qwen3-30B-A3B-Instruct-2507",
    max_seq_length = 2048, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)
```

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FMUjDPzhhjMJXcljIhgbK%2Fqwen33%20mascot.png?alt=media&#x26;token=fcfa1104-8f6d-4f04-b72d-b9c085d3ecda" alt=""><figcaption></figcaption></figure>


---
description: Guide on running Kimi K2 on your own local device!
---

# Kimi K2: How to Run Locally

Kimi K2 is the world’s most powerful open-source model, setting new SOTA performance in knowledge, reasoning, coding, and agentic tasks. The full 1T parameter model from Moonshot AI requires 1.09TB of disk space, while the quantized **Unsloth Dynamic 1.8-bit** version reduces this to just 245GB (-80% size)**:** [**Kimi-K2-GGUF**](https://huggingface.co/unsloth/Kimi-K2-Instruct-GGUF)

All uploads use Unsloth [Dynamic 2.0](unsloth-dynamic-2.0-ggufs) for SOTA 5-shot MMLU and KL Divergence performance, meaning you can run quantized LLMs with minimal accuracy loss.

{% hint style="success" %}
You can now use the latest update of [llama.cpp](#run-in-llama.cpp) to run the model! **Tool calling also got updated as at 16th July 2025** - you can use the old GGUF files you downloaded, and re-download the first GGUF file (50GB worth) OR use `--chat-template-file NEW_FILE.jinja` . [More details here](#tokenizer-quirks-and-bug-fixes).
{% endhint %}

<a href="https://docs.unsloth.ai/basics/kimi-k2-how-to-run-locally#run-kimi-k2-tutorials" class="button primary">Run in llama.cpp</a>

## :gear: Recommended Settings

{% hint style="success" %}
You need **250GB of disk space** at least to run the 1bit quant!

The only requirement is **`disk space + RAM + VRAM ≥ 250GB`**. That means you do not need to have that much RAM or VRAM (GPU) to run the model, but it will just be slower.
{% endhint %}

The 1.8-bit (UD-TQ1\_0) quant will fit in a 1x 24GB GPU (with all MoE layers offloaded to system RAM or a fast disk). Expect around 5 tokens/s with this setup if you have bonus 256GB RAM as well. The full Kimi K2 Q8 quant is 1.09TB in size and will need at least 8 x H200 GPUs.

For optimal performance you will need at least **250GB unified memory or 250GB combined RAM+VRAM** for 5+ tokens/s. If you have less than 250GB combined RAM+VRAM, then the speed of the model will definitely take a hit.

**If you do not have 250GB of RAM+VRAM, no worries!** llama.cpp inherently has **disk offloading**, so through mmaping, it'll still work, just be slower - for example before you might get 5 to 10 tokens / second, now it's under 1 token.

We suggest using our **UD-Q2\_K\_XL (381GB)** quant to balance size and accuracy!

{% hint style="success" %}
For the best performance, have your VRAM + RAM combined = the size of the quant you're downloading. If not, it'll still work via disk offloading, just it'll be slower!
{% endhint %}

### 🌙 Official Recommended Settings:

According to [Moonshot AI](https://huggingface.co/moonshotai/Kimi-K2-Instruct), these are the recommended settings for Kimi K2 inference:

* Set the <mark style="background-color:green;">**temperature 0.6**</mark> to reduce repetition and incoherence.
*   Original default system prompt is:

    ```
    You are a helpful assistant
    ```
*   (Optional) Moonshot also suggests the below for the system prompt:

    ```
    You are Kimi, an AI assistant created by Moonshot AI.
    ```

{% hint style="success" %}
We recommend setting <mark style="background-color:green;">**min\_p to 0.01**</mark> to suppress the occurrence of unlikely tokens with low probabilities.
{% endhint %}

## :1234: Chat template and prompt format

Kimi Chat does use a BOS (beginning of sentence token). The system, user and assistant roles are all enclosed with `<|im_middle|>` which is interesting, and each get their own respective token `<|im_system|>, <|im_user|>, <|im_assistant|>`.

{% code overflow="wrap" %}
```python
<|im_system|>system<|im_middle|>You are a helpful assistant<|im_end|><|im_user|>user<|im_middle|>What is 1+1?<|im_end|><|im_assistant|>assistant<|im_middle|>2<|im_end|>
```
{% endcode %}

To separate the conversational boundaries (you must remove each new line), we get:

{% code overflow="wrap" %}
```
<|im_system|>system<|im_middle|>You are a helpful assistant<|im_end|>
<|im_user|>user<|im_middle|>What is 1+1?<|im_end|>
<|im_assistant|>assistant<|im_middle|>2<|im_end|>
```
{% endcode %}

## :floppy\_disk: Model uploads

**ALL our uploads** - including those that are not imatrix-based or dynamic, utilize our calibration dataset, which is specifically optimized for conversational, coding, and reasoning tasks.

<table data-full-width="false"><thead><tr><th>MoE Bits</th><th>Type + Link</th><th>Disk Size</th><th>Details</th></tr></thead><tbody><tr><td>1.66bit</td><td><a href="https://huggingface.co/unsloth/Kimi-K2-Instruct-GGUF/tree/main/UD-TQ1_0">UD-TQ1_0</a></td><td><strong>245GB</strong></td><td>1.92/1.56bit</td></tr><tr><td>1.78bit</td><td><a href="https://huggingface.co/unsloth/Kimi-K2-Instruct-GGUF/tree/main/UD-IQ1_S">UD-IQ1_S</a></td><td><strong>281GB</strong></td><td>2.06/1.56bit</td></tr><tr><td>1.93bit</td><td><a href="https://huggingface.co/unsloth/Kimi-K2-Instruct-GGUF/tree/main/UD-IQ1_M">UD-IQ1_M</a></td><td><strong>304GB</strong></td><td>2.5/2.06/1.56</td></tr><tr><td>2.42bit</td><td><a href="https://huggingface.co/unsloth/Kimi-K2-Instruct-GGUF/tree/main/UD-IQ2_XXS">UD-IQ2_XXS</a></td><td><strong>343GB</strong></td><td>2.5/2.06bit</td></tr><tr><td>2.71bit</td><td><a href="https://huggingface.co/unsloth/Kimi-K2-Instruct-GGUF/tree/main/UD-Q2_K_XL">UD-Q2_K_XL</a></td><td><strong>381GB</strong></td><td> 3.5/2.5bit</td></tr><tr><td>3.12bit</td><td><a href="https://huggingface.co/unsloth/Kimi-K2-Instruct-GGUF/tree/main/UD-IQ3_XXS">UD-IQ3_XXS</a></td><td><strong>417GB</strong></td><td> 3.5/2.06bit</td></tr><tr><td>3.5bit</td><td><a href="https://huggingface.co/unsloth/Kimi-K2-Instruct-GGUF/tree/main/UD-Q3_K_XL">UD-Q3_K_XL</a></td><td><strong>452GB</strong></td><td> 4.5/3.5bit</td></tr><tr><td>4.5bit</td><td><a href="https://huggingface.co/unsloth/Kimi-K2-Instruct-GGUF/tree/main/UD-Q4_K_XL">UD-Q4_K_XL</a></td><td><strong>588GB</strong></td><td> 5.5/4.5bit</td></tr><tr><td>5.5bit</td><td><a href="https://huggingface.co/unsloth/Kimi-K2-Instruct-GGUF/tree/main/UD-Q5_K_XL">UD-Q5_K_XL</a></td><td><strong>732GB</strong></td><td>6.5/5.5bit</td></tr></tbody></table>

We've also uploaded versions in [BF16 format](https://huggingface.co/unsloth/Kimi-K2-Instruct-BF16).

## :turtle:Run Kimi K2 Tutorials

{% hint style="success" %}
You can now use the latest update of [llama.cpp](https://github.com/ggml-org/llama.cpp) to run the model:
{% endhint %}

### ✨ Run in llama.cpp

1. Obtain the latest `llama.cpp` on [GitHub here](https://github.com/ggml-org/llama.cpp). You can follow the build instructions below as well. Change `-DGGML_CUDA=ON` to `-DGGML_CUDA=OFF` if you don't have a GPU or just want CPU inference.

```bash
apt-get update
apt-get install pciutils build-essential cmake curl libcurl4-openssl-dev -y
git clone https://github.com/ggml-org/llama.cpp
cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON
cmake --build llama.cpp/build --config Release -j --clean-first --target llama-quantize llama-cli llama-gguf-split llama-mtmd-cli
cp llama.cpp/build/bin/llama-* llama.cpp
```

2. If you want to use `llama.cpp` directly to load models, you can do the below: (:UD-IQ1\_S) is the quantization type. You can also download via Hugging Face (point 3). This is similar to `ollama run` . Use `export LLAMA_CACHE="folder"` to force `llama.cpp` to save to a specific location.

{% hint style="success" %}
Please try out `-ot ".ffn_.*_exps.=CPU"` to offload all MoE layers to the CPU! This effectively allows you to fit all non MoE layers on 1 GPU, improving generation speeds. You can customize the regex expression to fit more layers if you have more GPU capacity.

If you have a bit more GPU memory, try `-ot ".ffn_(up|down)_exps.=CPU"` This offloads up and down projection MoE layers.

Try `-ot ".ffn_(up)_exps.=CPU"` if you have even more GPU memory. This offloads only up projection MoE layers.

And finally offload all layers via `-ot ".ffn_.*_exps.=CPU"` This uses the least VRAM.

You can also customize the regex, for example `-ot "\.(6|7|8|9|[0-9][0-9]|[0-9][0-9][0-9])\.ffn_(gate|up|down)_exps.=CPU"` means to offload gate, up and down MoE layers but only from the 6th layer onwards.
{% endhint %}

```bash
export LLAMA_CACHE="unsloth/Kimi-K2-Instruct-GGUF"
./llama.cpp/llama-cli \
    -hf unsloth/Kimi-K2-Instruct-GGUF:TQ1_0 \
    --cache-type-k q4_0 \
    --threads -1 \
    --n-gpu-layers 99 \
    --temp 0.6 \
    --min_p 0.01 \
    --ctx-size 16384 \
    --seed 3407 \
    -ot ".ffn_.*_exps.=CPU"
```

3. Download the model via (after installing `pip install huggingface_hub hf_transfer` ). You can choose `UD-TQ1_0`(dynamic 1.8bit quant) or other quantized versions like `Q2_K_XL` . We <mark style="background-color:green;">**recommend using our 2bit dynamic quant**</mark><mark style="background-color:green;">**&#x20;**</mark><mark style="background-color:green;">**`UD-Q2_K_XL`**</mark><mark style="background-color:green;">**&#x20;**</mark><mark style="background-color:green;">**to balance size and accuracy**</mark>. More versions at: [huggingface.co/unsloth/Kimi-K2-Instruct-GGUF](https://huggingface.co/unsloth/Kimi-K2-Instruct-GGUF)

{% code overflow="wrap" %}
```python
# !pip install huggingface_hub hf_transfer
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0" # Can sometimes rate limit, so set to 0 to disable
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id = "unsloth/Kimi-K2-Instruct-GGUF",
    local_dir = "unsloth/Kimi-K2-Instruct-GGUF",
    allow_patterns = ["*UD-TQ1_0*"], # Dynamic 1bit (281GB) Use "*UD-Q2_K_XL*" for Dynamic 2bit (381GB)
)
```
{% endcode %}

{% hint style="info" %}
If you find that downloads get stuck at 90 to 95% or so, please see [https://docs.unsloth.ai/basics/troubleshooting-and-faqs#downloading-gets-stuck-at-90-to-95](https://docs.unsloth.ai/basics/troubleshooting-and-faqs#downloading-gets-stuck-at-90-to-95)
{% endhint %}

4. Run any prompt.
5. Edit `--threads -1` for the number of CPU threads (be default it's set to the maximum CPU threads), `--ctx-size 16384` for context length, `--n-gpu-layers 99` for GPU offloading on how many layers. Set it to 99 combined with MoE CPU offloading to get the best performance. Try adjusting it if your GPU goes out of memory. Also remove it if you have CPU only inference.

{% code overflow="wrap" %}
```bash
./llama.cpp/llama-cli \
    --model unsloth/Kimi-K2-Instruct-GGUF/UD-TQ1_0/Kimi-K2-Instruct-UD-TQ1_0-00001-of-00005.gguf \
    --cache-type-k q4_0 \
    --threads -1 \
    --n-gpu-layers 99 \
    --temp 0.6 \
    --min_p 0.01 \
    --ctx-size 16384 \
    --seed 3407 \
    -ot ".ffn_.*_exps.=CPU" \
    -no-cnv \
    --prompt "<|im_system|>system<|im_middle|>You are a helpful assistant<|im_end|><|im_user|>user<|im_middle|>Create a Flappy Bird game in Python. You must include these things:\n1. You must use pygame.\n2. The background color should be randomly chosen and is a light shade. Start with a light blue color.\n3. Pressing SPACE multiple times will accelerate the bird.\n4. The bird's shape should be randomly chosen as a square, circle or triangle. The color should be randomly chosen as a dark color.\n5. Place on the bottom some land colored as dark brown or yellow chosen randomly.\n6. Make a score shown on the top right side. Increment if you pass pipes and don't hit them.\n7. Make randomly spaced pipes with enough space. Color them randomly as dark green or light brown or a dark gray shade.\n8. When you lose, show the best score. Make the text inside the screen. Pressing q or Esc will quit the game. Restarting is pressing SPACE again.\nThe final game should be inside a markdown section in Python. Check your code for errors and fix them before the final markdown section.<|im_end|><|im_assistant|>assistant<|im_middle|>"
```
{% endcode %}

## :mag:Tokenizer quirks and bug fixes

**16th July 2025: Kimi K2 updated their tokenizer to enable multiple tool calls** as per [https://x.com/Kimi\_Moonshot/status/1945050874067476962](https://x.com/Kimi_Moonshot/status/1945050874067476962)

If you have the old checkpoints downloaded - now worries - simply download the first GGUF split which was changed. OR if you do not want to download any new files do:

```bash
wget https://huggingface.co/unsloth/Kimi-K2-Instruct/raw/main/chat_template.jinja
./llama.cpp ... --chat-template-file /dir/to/chat_template.jinja
```

The Kimi K2 tokenizer was interesting to play around with - <mark style="background-color:green;">**it's mostly similar in action to GPT-4o's tokenizer**</mark>! We first see in the [tokenization\_kimi.py](https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/tokenization_kimi.py) file the following regular expression (regex) that Kimi K2 uses:

```python
pat_str = "|".join(
    [
        r"""[\p{Han}]+""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ]
)
```

After careful inspection, we find Kimi K2 is nearly identical to GPT-4o's tokenizer regex which can be found in [llama.cpp's source code](https://github.com/ggml-org/llama.cpp/blob/55c509daf51d25bfaee9c8b8ce6abff103d4473b/src/llama-vocab.cpp#L400).

{% code overflow="wrap" %}
```
[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
```
{% endcode %}

Both tokenize numbers into groups of 1 to 3 numbers (9, 99, 999), and use similar patterns. The only difference looks to be the handling of "Han" or Chinese characters, which Kimi's tokenizer deals with more. [The PR](https://github.com/ggml-org/llama.cpp/pull/14654) by [https://github.com/gabriellarson](https://github.com/gabriellarson) handles these differences well after some [discussions here](https://github.com/ggml-org/llama.cpp/issues/14642#issuecomment-3067324745).

<mark style="background-color:green;">**We also find the correct EOS token should not be \[EOS], but rather <|im\_end|>, which we have also fixed in our model conversions.**</mark>

## :bird: Flappy Bird + other tests <a href="#heptagon-test" id="heptagon-test"></a>

We introduced the Flappy Bird test when our 1.58bit quants for DeepSeek R1 were provided. We found Kimi K2 one of the only models to one-shot all our tasks including this one, [Heptagon ](../deepseek-r1-0528-how-to-run-locally#heptagon-test)and others tests even at 2-bit. The goal is to ask the LLM to create a Flappy Bird game but following some specific instructions:

{% code overflow="wrap" %}
```
Create a Flappy Bird game in Python. You must include these things:
1. You must use pygame.
2. The background color should be randomly chosen and is a light shade. Start with a light blue color.
3. Pressing SPACE multiple times will accelerate the bird.
4. The bird's shape should be randomly chosen as a square, circle or triangle. The color should be randomly chosen as a dark color.
5. Place on the bottom some land colored as dark brown or yellow chosen randomly.
6. Make a score shown on the top right side. Increment if you pass pipes and don't hit them.
7. Make randomly spaced pipes with enough space. Color them randomly as dark green or light brown or a dark gray shade.
8. When you lose, show the best score. Make the text inside the screen. Pressing q or Esc will quit the game. Restarting is pressing SPACE again.
The final game should be inside a markdown section in Python. Check your code for errors and fix them before the final markdown section.
```
{% endcode %}

You can also test the dynamic quants via the Heptagon Test as per [r/Localllama](https://www.reddit.com/r/LocalLLaMA/comments/1j7r47l/i_just_made_an_animation_of_a_ball_bouncing/) which tests the model on creating a basic physics engine to simulate balls rotating in a moving enclosed heptagon shape.

<figure><img src="https://docs.unsloth.ai/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252F2O72oTw5yPUbcxXjDNKS%252Fsnapshot.jpg%3Falt%3Dmedia%26token%3Dce852f9f-20ee-4b93-9d7b-1a5f211b9e04&#x26;width=768&#x26;dpr=4&#x26;quality=100&#x26;sign=55d1134d&#x26;sv=2" alt="" width="563"><figcaption></figcaption></figure>

The goal is to make the heptagon spin, and the balls in the heptagon should move. The prompt is below:

{% code overflow="wrap" %}
```
Write a Python program that shows 20 balls bouncing inside a spinning heptagon:\n- All balls have the same radius.\n- All balls have a number on it from 1 to 20.\n- All balls drop from the heptagon center when starting.\n- Colors are: #f8b862, #f6ad49, #f39800, #f08300, #ec6d51, #ee7948, #ed6d3d, #ec6800, #ec6800, #ee7800, #eb6238, #ea5506, #ea5506, #eb6101, #e49e61, #e45e32, #e17b34, #dd7a56, #db8449, #d66a35\n- The balls should be affected by gravity and friction, and they must bounce off the rotating walls realistically. There should also be collisions between balls.\n- The material of all the balls determines that their impact bounce height will not exceed the radius of the heptagon, but higher than ball radius.\n- All balls rotate with friction, the numbers on the ball can be used to indicate the spin of the ball.\n- The heptagon is spinning around its center, and the speed of spinning is 360 degrees per 5 seconds.\n- The heptagon size should be large enough to contain all the balls.\n- Do not use the pygame library; implement collision detection algorithms and collision response etc. by yourself. The following Python libraries are allowed: tkinter, math, numpy, dataclasses, typing, sys.\n- All codes should be put in a single Python file.
```
{% endcode %}

---
description: >-
  A guide on how to run DeepSeek-R1-0528 including Qwen3 on your own local
  device!
---

# DeepSeek-R1-0528: How to Run Locally

DeepSeek-R1-0528 is DeepSeek's new update to their R1 reasoning model. The full 671B parameter model requires 715GB of disk space. The quantized dynamic **1.66-bit** version uses 162GB (-80% reduction in size). GGUF: [DeepSeek-R1-0528-GGUF](https://huggingface.co/unsloth/DeepSeek-R1-0528-GGUF)

DeepSeek also released a R1-0528 distilled version by fine-tuning Qwen3 (8B). The distill achieves similar performance to Qwen3 (235B). _**You can also**_ [_**fine-tune Qwen3 Distill**_](#fine-tuning-deepseek-r1-0528-with-unsloth) _**with Unsloth**_. Qwen3 GGUF: [DeepSeek-R1-0528-Qwen3-8B-GGUF](https://huggingface.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF)

All uploads use Unsloth [Dynamic 2.0](unsloth-dynamic-2.0-ggufs) for SOTA 5-shot MMLU and KL Divergence performance, meaning you can run & fine-tune quantized DeepSeek LLMs with minimal accuracy loss.

**Tutorials navigation:**

<a href="#run-qwen3-distilled-r1-in-llama.cpp" class="button secondary">Run in llama.cpp</a><a href="#run-in-ollama-open-webui" class="button secondary">Run in Ollama/Open WebUI</a><a href="#fine-tuning-deepseek-r1-0528-with-unsloth" class="button secondary">Fine-tuning R1-0528</a>

{% hint style="success" %}
NEW: Huge improvements to tool calling and chat template fixes.\
\
New [TQ1\_0 dynamic 1.66-bit quant](https://huggingface.co/unsloth/DeepSeek-R1-0528-GGUF?show_file_info=DeepSeek-R1-0528-UD-TQ1_0.gguf) - 162GB in size. Ideal for 192GB RAM (including Mac) and Ollama users. Try: `ollama run hf.co/unsloth/DeepSeek-R1-0528-GGUF:TQ1_0`
{% endhint %}

## :gear: Recommended Settings

For DeepSeek-R1-0528-Qwen3-8B, the model can pretty much fit in any setup, and even those with as less as 20GB RAM. There is no need for any prep beforehand.\
\
However, for the full R1-0528 model which is 715GB in size, you will need extra prep. The 1.78-bit (IQ1\_S) quant will fit in a 1x 24GB GPU (with all layers offloaded). Expect around 5 tokens/s with this setup if you have bonus 128GB RAM as well.

It is recommended to have at least 64GB RAM to run this quant (you will get 1 token/s without a GPU). For optimal performance you will need at least **180GB unified memory or 180GB combined RAM+VRAM** for 5+ tokens/s.

We suggest using our 2.7bit (Q2\_K\_XL) or 2.4bit (IQ2\_XXS) quant to balance size and accuracy! The 2.4bit one also works well.

{% hint style="success" %}
Though not necessary, for the best performance, have your VRAM + RAM combined = to the size of the quant you're downloading.
{% endhint %}

### 🐳 Official Recommended Settings:

According to [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324), these are the recommended settings for R1 (R1-0528 and Qwen3 distill should use the same settings) inference:

* Set the <mark style="background-color:green;">**temperature 0.6**</mark> to reduce repetition and incoherence.
* Set <mark style="background-color:green;">**top\_p to 0.95**</mark> (recommended)
* Run multiple tests and average results for reliable evaluation.

### :1234: Chat template/prompt format

R1-0528 uses the same chat template as the original R1 model. You do not need to force `<think>\n` , but you can still add it in!

```
<｜begin▁of▁sentence｜><｜User｜>What is 1+1?<｜Assistant｜>It's 2.<｜end▁of▁sentence｜><｜User｜>Explain more!<｜Assistant｜>
```

A BOS is forcibly added, and an EOS separates each interaction. To counteract double BOS tokens during inference, you should only call `tokenizer.encode(..., add_special_tokens = False)` since the chat template auto adds a BOS token as well.\
For llama.cpp / GGUF inference, you should skip the BOS since it’ll auto add it:

```
<｜User｜>What is 1+1?<｜Assistant｜>
```

The `<think>` and `</think>` tokens get their own designated tokens.

## Model uploads

**ALL our uploads** - including those that are not imatrix-based or dynamic, utilize our calibration dataset, which is specifically optimized for conversational, coding, and language tasks.

* Qwen3 (8B) distill: [DeepSeek-R1-0528-Qwen3-8B-GGUF](https://huggingface.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF)
* Full DeepSeek-R1-0528 model uploads below:

We also uploaded [IQ4\_NL](https://huggingface.co/unsloth/DeepSeek-R1-0528-GGUF/tree/main/IQ4_NL) and [Q4\_1](https://huggingface.co/unsloth/DeepSeek-R1-0528-GGUF/tree/main/Q4_1) quants which run specifically faster for ARM and Apple devices respectively.

<table data-full-width="false"><thead><tr><th>MoE Bits</th><th>Type + Link</th><th>Disk Size</th><th>Details</th></tr></thead><tbody><tr><td>1.66bit</td><td><a href="https://huggingface.co/unsloth/DeepSeek-R1-0528-GGUF?show_file_info=DeepSeek-R1-0528-UD-TQ1_0.gguf">TQ1_0</a></td><td><strong>162GB</strong></td><td>1.92/1.56bit</td></tr><tr><td>1.78bit</td><td><a href="https://huggingface.co/unsloth/DeepSeek-R1-0528-GGUF/tree/main/UD-IQ1_S">IQ1_S</a></td><td><strong>185GB</strong></td><td>2.06/1.56bit</td></tr><tr><td>1.93bit</td><td><a href="https://huggingface.co/unsloth/DeepSeek-V3-0324-GGUF/tree/main/UD-IQ1_M">IQ1_M</a></td><td><strong>200GB</strong></td><td>2.5/2.06/1.56</td></tr><tr><td>2.42bit</td><td><a href="https://huggingface.co/unsloth/DeepSeek-R1-0528-GGUF/tree/main/UD-IQ2_XXS">IQ2_XXS</a></td><td><strong>216GB</strong></td><td>2.5/2.06bit</td></tr><tr><td>2.71bit</td><td><a href="https://huggingface.co/unsloth/DeepSeek-R1-0528-GGUF/tree/main/UD-Q2_K_XL">Q2_K_XL</a></td><td><strong>251GB</strong></td><td> 3.5/2.5bit</td></tr><tr><td>3.12bit</td><td><a href="https://huggingface.co/unsloth/DeepSeek-R1-0528-GGUF/tree/main/UD-IQ3_XXS">IQ3_XXS</a></td><td><strong>273GB</strong></td><td> 3.5/2.06bit</td></tr><tr><td>3.5bit</td><td><a href="https://huggingface.co/unsloth/DeepSeek-R1-0528-GGUF/tree/main/UD-Q3_K_XL">Q3_K_XL</a></td><td><strong>296GB</strong></td><td> 4.5/3.5bit</td></tr><tr><td>4.5bit</td><td><a href="https://huggingface.co/unsloth/DeepSeek-R1-0528-GGUF/tree/main/UD-Q4_K_XL">Q4_K_XL</a></td><td><strong>384GB</strong></td><td> 5.5/4.5bit</td></tr><tr><td>5.5bit</td><td><a href="https://huggingface.co/unsloth/DeepSeek-R1-0528-GGUF/tree/main/UD-Q5_K_XL">Q5_K_XL</a></td><td><strong>481GB</strong></td><td>6.5/5.5bit</td></tr></tbody></table>

We've also uploaded versions in [BF16 format](https://huggingface.co/unsloth/DeepSeek-R1-0528-BF16), and original [FP8 (float8) format](https://huggingface.co/unsloth/DeepSeek-R1-0528).

## Run DeepSeek-R1-0528 Tutorials:

### :llama: Run in Ollama/Open WebUI

1. Install `ollama` if you haven't already! You can only run models up to 32B in size. To run the full 720GB R1-0528 model, [see here](#run-full-r1-0528-on-ollama-open-webui).

```bash
apt-get update
apt-get install pciutils -y
curl -fsSL https://ollama.com/install.sh | sh
```

2. Run the model! Note you can call `ollama serve`in another terminal if it fails! We include all our fixes and suggested parameters (temperature etc) in `params` in our Hugging Face upload!

```bash
ollama run hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_XL
```

3. <mark style="color:green;background-color:yellow;">**(NEW) To run the full R1-0528 model in Ollama, you can use our TQ1\_0 (162GB quant):**</mark>

```
OLLAMA_MODELS=unsloth_downloaded_models ollama serve &

ollama run hf.co/unsloth/DeepSeek-R1-0528-GGUF:TQ1_0
```

### :llama: Run Full R1-0528 on Ollama/Open WebUI

Open WebUI has made an step-by-step tutorial on how to run R1 here and for R1-0528, you will just need to replace R1 with the new 0528 quant: [docs.openwebui.com/tutorials/integrations/deepseekr1-dynamic/](https://docs.openwebui.com/tutorials/integrations/deepseekr1-dynamic/)

<mark style="background-color:green;">**(NEW) To run the full R1-0528 model in Ollama, you can use our TQ1\_0 (162GB quant):**</mark>

```
OLLAMA_MODELS=unsloth_downloaded_models ollama serve &

ollama run hf.co/unsloth/DeepSeek-R1-0528-GGUF:TQ1_0
```

If you want to use any of the quants that are larger than TQ1\_0 (162GB) on Ollama, you need to first merge the 3 GGUF split files into 1 like the code below. Then you will need to run the model locally.

```
./llama.cpp/llama-gguf-split --merge \
  DeepSeek-R1-0528-GGUF/DeepSeek-R1-0528-UD-IQ1_S/DeepSeek-R1-0528-UD-IQ1_S-00001-of-00003.gguf \
	merged_file.gguf
```

### ✨ Run Qwen3 distilled R1 in llama.cpp

1. <mark style="background-color:yellow;">**To run the full 720GB R1-0528 model,**</mark> [<mark style="background-color:yellow;">**see here**</mark>](#run-full-r1-0528-on-llama.cpp)<mark style="background-color:yellow;">**.**</mark> Obtain the latest `llama.cpp` on [GitHub here](https://github.com/ggml-org/llama.cpp). You can follow the build instructions below as well. Change `-DGGML_CUDA=ON` to `-DGGML_CUDA=OFF` if you don't have a GPU or just want CPU inference.

```bash
apt-get update
apt-get install pciutils build-essential cmake curl libcurl4-openssl-dev -y
git clone https://github.com/ggml-org/llama.cpp
cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON
cmake --build llama.cpp/build --config Release -j --clean-first --target llama-cli llama-gguf-split
cp llama.cpp/build/bin/llama-* llama.cpp
```

2. Then use llama.cpp directly to download the model:

```bash
./llama.cpp/llama-cli -hf unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_XL --jinja
```

### ✨ Run Full R1-0528 on llama.cpp

1. Obtain the latest `llama.cpp` on [GitHub here](https://github.com/ggml-org/llama.cpp). You can follow the build instructions below as well. Change `-DGGML_CUDA=ON` to `-DGGML_CUDA=OFF` if you don't have a GPU or just want CPU inference.

```bash
apt-get update
apt-get install pciutils build-essential cmake curl libcurl4-openssl-dev -y
git clone https://github.com/ggerganov/llama.cpp
cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON
cmake --build llama.cpp/build --config Release -j --clean-first --target llama-quantize llama-cli llama-gguf-split llama-mtmd-cli
cp llama.cpp/build/bin/llama-* llama.cpp
```

2. If you want to use `llama.cpp` directly to load models, you can do the below: (:IQ1\_S) is the quantization type. You can also download via Hugging Face (point 3). This is similar to `ollama run` . Use `export LLAMA_CACHE="folder"` to force `llama.cpp` to save to a specific location.

{% hint style="success" %}
Please try out `-ot ".ffn_.*_exps.=CPU"` to offload all MoE layers to the CPU! This effectively allows you to fit all non MoE layers on 1 GPU, improving generation speeds. You can customize the regex expression to fit more layers if you have more GPU capacity.

If you have a bit more GPU memory, try `-ot ".ffn_(up|down)_exps.=CPU"` This offloads up and down projection MoE layers.

Try `-ot ".ffn_(up)_exps.=CPU"` if you have even more GPU memory. This offloads only up projection MoE layers.

And finally offload all layers via `-ot ".ffn_.*_exps.=CPU"` This uses the least VRAM.

You can also customize the regex, for example `-ot "\.(6|7|8|9|[0-9][0-9]|[0-9][0-9][0-9])\.ffn_(gate|up|down)_exps.=CPU"` means to offload gate, up and down MoE layers but only from the 6th layer onwards.
{% endhint %}

```bash
export LLAMA_CACHE="unsloth/DeepSeek-R1-0528-GGUF"
./llama.cpp/llama-cli \
    -hf unsloth/DeepSeek-R1-0528-GGUF:IQ1_S \
    --cache-type-k q4_0 \
    --threads -1 \
    --n-gpu-layers 99 \
    --prio 3 \
    --temp 0.6 \
    --top_p 0.95 \
    --min_p 0.01 \
    --ctx-size 16384 \
    --seed 3407 \
    -ot ".ffn_.*_exps.=CPU"
```

3. Download the model via (after installing `pip install huggingface_hub hf_transfer` ). You can choose `UD-IQ1_S`(dynamic 1.78bit quant) or other quantized versions like `Q4_K_M` . We <mark style="background-color:green;">**recommend using our 2.7bit dynamic quant**</mark><mark style="background-color:green;">**&#x20;**</mark><mark style="background-color:green;">**`UD-Q2_K_XL`**</mark><mark style="background-color:green;">**&#x20;**</mark><mark style="background-color:green;">**to balance size and accuracy**</mark>. More versions at: [https://huggingface.co/unsloth/DeepSeek-R1-0528-GGUF](https://huggingface.co/unsloth/DeepSeek-V3-0324-GGUF)

{% code overflow="wrap" %}
```python
# !pip install huggingface_hub hf_transfer
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0" # Can sometimes rate limit, so set to 0 to disable
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id = "unsloth/DeepSeek-R1-0528-GGUF",
    local_dir = "unsloth/DeepSeek-R1-0528-GGUF",
    allow_patterns = ["*UD-IQ1_S*"], # Dynamic 1bit (168GB) Use "*UD-Q2_K_XL*" for Dynamic 2bit (251GB)
)
```
{% endcode %}

4. Run Unsloth's Flappy Bird test as described in our 1.58bit Dynamic Quant for DeepSeek R1.
5. Edit `--threads 32` for the number of CPU threads, `--ctx-size 16384` for context length, `--n-gpu-layers 2` for GPU offloading on how many layers. Try adjusting it if your GPU goes out of memory. Also remove it if you have CPU only inference.

{% code overflow="wrap" %}
```bash
./llama.cpp/llama-cli \
    --model unsloth/DeepSeek-R1-0528-GGUF/UD-IQ1_S/DeepSeek-R1-0528-UD-IQ1_S-00001-of-00004.gguf \
    --cache-type-k q4_0 \
    --threads -1 \
    --n-gpu-layers 99 \
    --prio 3 \
    --temp 0.6 \
    --top_p 0.95 \
    --min_p 0.01 \
    --ctx-size 16384 \
    --seed 3407 \
    -ot ".ffn_.*_exps.=CPU" \
    -no-cnv \
    --prompt "<｜User｜>Create a Flappy Bird game in Python. You must include these things:\n1. You must use pygame.\n2. The background color should be randomly chosen and is a light shade. Start with a light blue color.\n3. Pressing SPACE multiple times will accelerate the bird.\n4. The bird's shape should be randomly chosen as a square, circle or triangle. The color should be randomly chosen as a dark color.\n5. Place on the bottom some land colored as dark brown or yellow chosen randomly.\n6. Make a score shown on the top right side. Increment if you pass pipes and don't hit them.\n7. Make randomly spaced pipes with enough space. Color them randomly as dark green or light brown or a dark gray shade.\n8. When you lose, show the best score. Make the text inside the screen. Pressing q or Esc will quit the game. Restarting is pressing SPACE again.\nThe final game should be inside a markdown section in Python. Check your code for errors and fix them before the final markdown section.<｜Assistant｜>"
```
{% endcode %}

## :8ball: Heptagon Test

You can also test our dynamic quants via [r/Localllama](https://www.reddit.com/r/LocalLLaMA/comments/1j7r47l/i_just_made_an_animation_of_a_ball_bouncing/) which tests the model on creating a basic physics engine to simulate balls rotating in a moving enclosed heptagon shape.

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2F2O72oTw5yPUbcxXjDNKS%2Fsnapshot.jpg?alt=media&#x26;token=ce852f9f-20ee-4b93-9d7b-1a5f211b9e04" alt="" width="563"><figcaption><p>The goal is to make the heptagon spin, and the balls in the heptagon should move.</p></figcaption></figure>

<details>

<summary>Full prompt to run the model</summary>

{% code overflow="wrap" %}
```bash
./llama.cpp/llama-cli \
    --model unsloth/DeepSeek-R1-0528-GGUF/UD-IQ1_S/DeepSeek-R1-0528-UD-IQ1_S-00001-of-00004.gguf \
    --cache-type-k q4_0 \
    --threads -1 \
    --n-gpu-layers 99 \
    --prio 3 \
    --temp 0.6 \
    --top_p 0.95 \
    --min_p 0.01 \
    --ctx-size 16384 \
    --seed 3407 \
    -ot ".ffn_.*_exps.=CPU" \
    -no-cnv \
    --prompt "<｜User｜>Write a Python program that shows 20 balls bouncing inside a spinning heptagon:\n- All balls have the same radius.\n- All balls have a number on it from 1 to 20.\n- All balls drop from the heptagon center when starting.\n- Colors are: #f8b862, #f6ad49, #f39800, #f08300, #ec6d51, #ee7948, #ed6d3d, #ec6800, #ec6800, #ee7800, #eb6238, #ea5506, #ea5506, #eb6101, #e49e61, #e45e32, #e17b34, #dd7a56, #db8449, #d66a35\n- The balls should be affected by gravity and friction, and they must bounce off the rotating walls realistically. There should also be collisions between balls.\n- The material of all the balls determines that their impact bounce height will not exceed the radius of the heptagon, but higher than ball radius.\n- All balls rotate with friction, the numbers on the ball can be used to indicate the spin of the ball.\n- The heptagon is spinning around its center, and the speed of spinning is 360 degrees per 5 seconds.\n- The heptagon size should be large enough to contain all the balls.\n- Do not use the pygame library; implement collision detection algorithms and collision response etc. by yourself. The following Python libraries are allowed: tkinter, math, numpy, dataclasses, typing, sys.\n- All codes should be put in a single Python file.<｜Assistant｜>"
```
{% endcode %}



</details>

## 🦥 Fine-tuning DeepSeek-R1-0528 with Unsloth

To fine-tune **DeepSeek-R1-0528-Qwen3-8B** using Unsloth, we’ve made a new GRPO notebook featuring a custom reward function designed to significantly enhance multilingual output - specifically increasing the rate of desired language responses (in our example we use Indonesian but you can use any) by more than 40%.

* [**DeepSeek-R1-0528-Qwen3-8B notebook**](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/DeepSeek_R1_0528_Qwen3_\(8B\)_GRPO.ipynb) **- new**

While many reasoning LLMs have multilingual capabilities, they often produce mixed-language outputs in its reasoning traces, combining English with the target language. Our reward function effectively mitigates this issue by strongly encouraging outputs in the desired language, leading to a substantial improvement in language consistency.

This reward function is also fully customizable, allowing you to adapt it for other languages or fine-tune for specific domains or use cases.

{% hint style="success" %}
The best part about this whole reward function and notebook is you DO NOT need a language dataset to force your model to learn a specific language. The notebook has no Indonesian dataset.
{% endhint %}

Unsloth makes R1-Qwen3 distill fine-tuning 2× faster, uses 70% less VRAM, and support 8× longer context lengths.

---
description: >-
  Learn all about Reinforcement Learning (RL) and how to train your own
  DeepSeek-R1 reasoning model with Unsloth using GRPO. A complete guide from
  beginner to advanced.
---

# Reinforcement Learning (RL) Guide

## :sloth:What you will learn

1. What is RL? RLVR? PPO? GRPO? RLHF? RFT? Is <mark style="background-color:green;">**"Luck is All You Need?"**</mark> for RL?
2. What is an environment? Agent? Action? Reward function? Rewards?

This article covers everything (from beginner to advanced) you need to know about GRPO, Reinforcement Learning (RL) and reward functions, along with tips, and the basics of using GRPO with [Unsloth](https://github.com/unslothai/unsloth). If you're looking for a step-by-step tutorial for using GRPO, see our guide [here](reinforcement-learning-rl-guide/tutorial-train-your-own-reasoning-model-with-grpo).

## :question:What is Reinforcement Learning (RL)?

The goal of RL is to:

1. **Increase the chance of seeing&#x20;**<mark style="background-color:green;">**"good"**</mark>**&#x20;outcomes.**
2. **Decrease the chance of seeing&#x20;**<mark style="background-color:red;">**"bad"**</mark>**&#x20;outcomes.**

**That's it!** There are intricacies on what "good" and "bad" means, or how do we go about "increasing" or "decreasing" it, or what even "outcomes" means.

{% columns %}
{% column width="50%" %}
For example, in the **Pacman game**:

1. The <mark style="background-color:green;">**environment**</mark> is the game world.
2. The <mark style="background-color:blue;">**actions**</mark> you can take are UP, LEFT, RIGHT and DOWN.
3. The <mark style="background-color:purple;">**rewards**</mark> are good if you eat a cookie, or bad if you hit one of the squiggly enemies.
4. In RL, you can't know the "best action" you can take, but you can observe intermediate steps, or the final game state (win or lose)
{% endcolumn %}

{% column %}
<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FLYKyo5xU4mSvQRASnH1D%2FRL%20Game.png?alt=media&#x26;token=16e9a8c6-61f9-4baf-84a7-118e562eb6c5" alt=""><figcaption></figcaption></figure>
{% endcolumn %}
{% endcolumns %}

{% columns %}
{% column width="50%" %}
<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FVVJbst1Vn3Pg6jn0hXLA%2FMath%20RL.png?alt=media&#x26;token=855abbe8-d134-4246-ae5c-5108574aaa6e" alt=""><figcaption></figcaption></figure>
{% endcolumn %}

{% column %}
Another example is imagine you are given the question: <mark style="background-color:blue;">**"What is 2 + 2?"**</mark> (4) An unaligned language model will spit out 3, 4, C, D, -10, literally anything.

1. Numbers are better than C or D right?
2. Getting 3 is better than say 8 right?
3. Getting 4 is definitely correct.

We just designed a <mark style="background-color:orange;">**reward function**</mark>!
{% endcolumn %}
{% endcolumns %}

## :person\_running:From RLHF, PPO to GRPO and RLVR

{% columns %}
{% column %}
<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FU3NH5rSkI17fysvnMJHJ%2FRLHF.png?alt=media&#x26;token=53625e98-2949-45d1-b650-c5a7313b18a0" alt=""><figcaption></figcaption></figure>
{% endcolumn %}

{% column %}
OpenAI popularized the concept of [RLHF](https://en.wikipedia.org/wiki/Reinforcement_learning_from_human_feedback) (Reinforcement Learning from Human Feedback), where we train an <mark style="background-color:red;">**"agent"**</mark> to produce outputs to a question (the <mark style="background-color:yellow;">**state**</mark>) that are rated more useful by human beings.

The thumbs up and down in ChatGPT for example can be used in the RLHF process.
{% endcolumn %}
{% endcolumns %}

{% columns %}
{% column %}
<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2Fn5N2OBGIqk1oPbR9gRKn%2FPPO.png?alt=media&#x26;token=e9706260-6bee-4ef0-a7dc-f5f6d80471d5" alt=""><figcaption></figcaption></figure>

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FplVZSTOwKSQv5zQYjkge%2FPPO%20formula.png?alt=media&#x26;token=8b1359c8-11d1-4ea8-91c0-cf4afe120166" alt=""><figcaption><p>PPO formula</p></figcaption></figure>

The clip(..., 1-e, 1+e) term is used to force PPO not to take too large changes. There is also a KL term with beta set to > 0 to force the model not to deviate too much away.
{% endcolumn %}

{% column %}
In order to do RLHF, [<mark style="background-color:red;">**PPO**</mark>](https://en.wikipedia.org/wiki/Proximal_policy_optimization) (Proximal policy optimization) was developed. The <mark style="background-color:blue;">**agent**</mark> is the language model in this case. In fact it's composed of 3 systems:

1. The **Generating Policy (current trained model)**
2. The **Reference Policy (original model)**
3. The **Value Model (average reward estimator)**

We use the **Reward Model** to calculate the reward for the current environment, and our goal is to **maximize this**!

The formula for PPO looks quite complicated because it was designed to be stable. Visit our [AI Engineer talk](https://docs.unsloth.ai/ai-engineers-2025) we gave in 2025 about RL for more in depth maths derivations about PPO.
{% endcolumn %}
{% endcolumns %}

{% columns %}
{% column %}
<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FiQI4Yvv1KcvkK7g5V8vm%2FGRPO%20%2B%20RLVR.png?alt=media&#x26;token=2155a920-b986-4a08-871a-32b5bbcfdbe3" alt=""><figcaption></figcaption></figure>
{% endcolumn %}

{% column %}
DeepSeek developed [<mark style="background-color:red;">**GRPO**</mark>](https://unsloth.ai/blog/grpo) (Group Relative Policy Optimization) to train their R1 reasoning models. The key differences to PPO are:

1. The **Value Model is removed,** replaced with statistics from calling the reward model multiple times.
2. The **Reward Model is removed** and replaced with just custom reward function which <mark style="background-color:blue;">**RLVR**</mark> can be used.
{% endcolumn %}
{% endcolumns %}

This means GRPO is extremely efficient. Previously PPO needed to train multiple models - now with the reward model and value model removed, we can save memory and speed up everything.

<mark style="background-color:orange;">**RLVR (Reinforcement Learning with Verifiable Rewards)**</mark> allows us to reward the model based on tasks with easy to verify solutions. For example:

1. Maths equations can be easily verified. Eg 2+2 = 4.
2. Code output can be verified as having executed correctly or not.
3. Designing verifiable reward functions can be tough, and so most examples are math or code.
4. Use-cases for GRPO isn’t just for code or math—its reasoning process can enhance tasks like email automation, database retrieval, law, and medicine, greatly improving accuracy based on your dataset and reward function - the trick is to define a <mark style="background-color:yellow;">**rubric - ie a list of smaller verifiable rewards, and not a final all consuming singular reward.**</mark> OpenAI popularized this in their [reinforcement learning finetuning (RFT)](https://platform.openai.com/docs/guides/reinforcement-fine-tuning) offering for example.

{% columns %}
{% column %}
<mark style="background-color:red;">**Why "Group Relative"?**</mark>

GRPO removes the value model entirely, but we still need to estimate the <mark style="background-color:yellow;">**"average reward"**</mark> given the current state.

The **trick is to sample the LLM**! We then calculate the average reward through statistics of the sampling process across multiple different questions.
{% endcolumn %}

{% column %}
<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FdXw9vYkjJaKFLTMx0Py6%2FGroup%20Relative.png?alt=media&#x26;token=9153caf5-402e-414b-b5b4-79fef1a2c2fa" alt=""><figcaption></figcaption></figure>
{% endcolumn %}
{% endcolumns %}

{% columns %}
{% column %}
For example for "What is 2+2?" we sample 4 times. We might get 4, 3, D, C. We then calculate the reward for each of these answers, then calculate the **average reward** and **standard deviation**, then <mark style="background-color:red;">**Z-score standardize**</mark> this!

This creates the <mark style="background-color:blue;">**advantages A**</mark>, which we will use in replacement of the value model. This saves a lot of memory!
{% endcolumn %}

{% column %}
<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FVDdKLOBcLyLC3dwF1Idd%2FStatistics.png?alt=media&#x26;token=6c8eae5b-b063-4f49-b896-7f8de516a379" alt=""><figcaption><p>GRPO advantage calculation</p></figcaption></figure>
{% endcolumn %}
{% endcolumns %}

## :fingers\_crossed:Luck (well Patience) Is All You Need

The trick of RL is you need 2 things only:

1. A question or instruction eg "What is 2+2?" "Create a Flappy Bird game in Python"
2. A reward function and verifier to verify if the output is good or bad.

With only these 2, we can essentially **call a language model an infinite times** until we get a good answer. For example for "What is 2+2?", an untrained bad language model will output:

_**0, cat, -10, 1928, 3, A, B, 122, 17, 182, 172, A, C, BAHS, %$, #, 9, -192, 12.31****&#x20;**<mark style="color:green;">**then suddenly 4**</mark>**.**_

_**The reward signal was 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0****&#x20;**<mark style="color:green;">**then suddenly 1.**</mark>_

So by luck and by chance, RL managed to find the correct answer across multiple <mark style="background-color:yellow;">**rollouts**</mark>. Our goal is we want to see the good answer 4 more, and the rest (the bad answers) much less.

<mark style="color:blue;">**So the goal of RL is to be patient - in the limit, if the probability of the correct answer is at least a small number (not zero), it's just a waiting game - you will 100% for sure encounter the correct answer in the limit.**</mark>

<mark style="background-color:blue;">**So I like to call it as "Luck Is All You Need" for RL.**</mark>

<mark style="background-color:orange;">**Well a better phrase is "Patience is All You Need" for RL.**</mark>

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FryuL3pCuF8pPIjPEASbx%2FLuck%20is%20all%20you%20need.png?alt=media&#x26;token=64d1a03a-6afc-49a9-b734-8ce8bc2b5ec1" alt="" width="375"><figcaption></figcaption></figure>

RL essentially provides us a trick - instead of simply waiting for infinity, we do get "bad signals" ie bad answers, and we can essentially "guide" the model to already try not generating bad solutions. This means although you waited very long for a "good" answer to pop up, the model already has been changed to try its best not to output bad answers.

In the "What is 2+2?" example - _**0, cat, -10, 1928, 3, A, B, 122, 17, 182, 172, A, C, BAHS, %$, #, 9, -192, 12.31****&#x20;**<mark style="color:green;">**then suddenly 4**</mark>**.**_

Since we got bad answers, RL will influence the model to try NOT to output bad answers. This means over time, we are carefully "pruning" or moving the model's output distribution away from bad answers. This means RL is not inefficient, since we are NOT just waiting for infinity, but we are actively trying to "push" the model to go as much as possible to the "correct answer space".

{% hint style="danger" %}
**If the probability is always 0, then RL will never work**. This is also why people like to do RL from an already instruction finetuned model, which can partially follow instructions reasonably well - this boosts the probability most likely above 0.
{% endhint %}

## :sloth:What Unsloth offers for RL

* With 15GB VRAM, Unsloth allows you to transform any model up to 17B parameters like Llama 3.1 (8B), Phi-4 (14B), Mistral (7B) or Qwen2.5 (7B) into a reasoning model
* **Minimum requirement:** Just  5GB VRAM is enough to train your own reasoning model locally (for any model with 1.5B parameters or less)

{% content-ref url="reinforcement-learning-rl-guide/tutorial-train-your-own-reasoning-model-with-grpo" %}
[tutorial-train-your-own-reasoning-model-with-grpo](reinforcement-learning-rl-guide/tutorial-train-your-own-reasoning-model-with-grpo)
{% endcontent-ref %}

### GRPO notebooks:

| [**Qwen3 (4B)**](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_\(4B\)-GRPO.ipynb) - Advanced | [**DeepSeek-R1-0528-Qwen3-8B**](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/DeepSeek_R1_0528_Qwen3_\(8B\)_GRPO.ipynb) **- new** | [Llama 3.2 (3B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Advanced_Llama3_2_\(3B\)_GRPO_LoRA.ipynb) - Advanced |
| ------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| [Gemma 3 (1B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_\(1B\)-GRPO.ipynb)             | [Phi-4 (14B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4_\(14B\)-GRPO.ipynb)                                             | [Qwen2.5 (3B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_\(3B\)-GRPO.ipynb)                             |
| [Mistral v0.3 (7B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_\(7B\)-GRPO.ipynb)  | [Llama 3.1 (8B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_\(8B\)-GRPO.ipynb)                                        |                                                                                                                                                 |

{% hint style="success" %}
**NEW!** We now support Dr. GRPO and most other new GRPO techniques. You can play with the following arguments in GRPOConfig to enable:

```python
epsilon=0.2,
epsilon_high=0.28, # one sided
delta=1.5 # two sided

loss_type='bnpo',
# or:
loss_type='grpo',
# or:
loss_type='dr_grpo',

mask_truncated_completions=True,
```
{% endhint %}

* If you're not getting any reasoning, make sure you have enough training steps and ensure your [reward function/verifier](#reward-functions-verifier) is working. We provide examples for reward functions [here](#reward-function-examples).
* Previous demonstrations show that you could achieve your own "aha" moment with Qwen2.5 (3B) - but it required 2xA100 GPUs (160GB VRAM). Now, with Unsloth, you can achieve the same "aha" moment using just a single 5GB VRAM GPU.
* Previously, GRPO was only supported for full fine-tuning, but we've made it work with QLoRA and LoRA
* On [**20K context lengths**](#grpo-requirement-guidelines) for example with 8 generations per prompt, Unsloth uses only 54.3GB of VRAM for Llama 3.1 (8B), whilst standard implementations (+ Flash Attention 2) take **510.8GB (90% less for Unsloth)**.
* Please note, this isn’t fine-tuning DeepSeek’s R1 distilled models or using distilled data from R1 for tuning which Unsloth already supported. This is converting a standard model into a full-fledged reasoning model using GRPO.

In a test example, even though we only trained Phi-4 with 100 steps using GRPO, the results are already clear. The model without GRPO does not have the thinking token, whilst the one trained with GRPO does and also has the correct answer.

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FyBeJAvfolzfEYyftji76%2Fprompt%20only%20example.png?alt=media&#x26;token=3903995a-d9d5-4cdc-9020-c4efe7fff651" alt=""><figcaption></figcaption></figure>

## :computer:Training with GRPO

For a tutorial on how to transform any open LLM into a reasoning model using Unsloth & GRPO, [see here](reinforcement-learning-rl-guide/tutorial-train-your-own-reasoning-model-with-grpo).

### **How GRPO Trains a Model**

1. For each question-answer pair, the model generates multiple possible responses (e.g., 8 variations).
2. Each response is evaluated using reward functions.
3. Training Steps:
   * If you have 300 rows of data, that's 300 training steps (or 900 steps if trained for 3 epochs).
   * You can increase the number of generated responses per question (e.g., from 8 to 16).
4. The model learns by updating its weights every step.

{% hint style="warning" %}
If you're having issues with your GRPO model not learning, we'd highly recommend to use our [Advanced GRPO notebooks](../../get-started/unsloth-notebooks#grpo-reasoning-notebooks) as it has a much better reward function and you should see results much faster and frequently.
{% endhint %}

### Basics/Tips

* Wait for at least **300 steps** for the reward to actually increase. In order to get decent results, you may need to trade for a minimum of 12 hours (this is how GRPO works), but keep in mind this isn't compulsory as you can stop at anytime.
* For optimal results have at least **500 rows of data**. You can try with even 10 rows of data but it's better to have more.
* Each training run will always be different depending on your model, data, reward function/verifier etc. so though 300 steps is what we wrote as the minimum, sometimes it might be 1000 steps or more. So, it depends on various factors.
* If you're using GRPO with Unsloth locally, please "pip install diffusers" as well if you get an error. Please also use the latest version of vLLM.
* It’s advised to apply GRPO to a model at least **1.5B in parameters** to correctly generate thinking tokens as smaller models may not.
* For GRPO's [**GPU VRAM requirements**](#grpo-requirement-guidelines) **for QLoRA 4-bit**, the general rule is the model parameters = the amount of VRAM you will need (you can use less VRAM but this just to be safe). The more context length you set, the more VRAM. LoRA 16-bit will use at minimum 4x more VRAM.
* **Continuous fine-tuning is** possible and you can just leave GRPO running in the background.
* In the example notebooks, we use the [**GSM8K dataset**](#gsm8k-reward-functions), the current most popular choice for R1-style training.
* If you’re using a base model, ensure you have a chat template.
*   The more you train with GRPO the better. The best part of GRPO is you don't even need that much data. All you need is a great reward function/verifier and the more time spent training, the better your model will get. Expect your reward vs step to increase as time progresses like this:

    <figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FUROleqJQ5aEp8MjTCWFf%2Funnamed.png?alt=media&#x26;token=12ca4975-7a0c-4d10-9178-20db28ad0451" alt="" width="563"><figcaption></figcaption></figure>
* Training loss tracking for GRPO is now built directly into Unsloth, eliminating the need for external tools like wandb etc. It contains full logging details for all reward functions now including the total aggregated reward function itself.

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2Fjo7fVFoFG2xbZPgL45el%2FScreenshot%202025-02-20%20at%2004-52-52%20Copy%20of%20Yet%20another%20copy%20of%20Llama3.1_(8B)-GRPO.ipynb%20-%20Colab.png?alt=media&#x26;token=041c17b1-ab98-4ab6-b6fb-8c7e5a8c07df" alt=""><figcaption></figcaption></figure>

## :clipboard:Reward Functions / Verifiers

In Reinforcement Learning a **Reward Function** and a **Verifier** serve distinct roles in evaluating a model’s output. In general, you could interpret them as the same thing however, technically they're not but it does not matter as much as they are usually used in conjunction with each other.

**Verifier**:

* Determines whether the generated response is correct or incorrect.
* It does not assign a numerical score—it simply verifies correctness.
* Example: If a model generates "5" for "2+2", the verifier checks and labels it as "wrong" (since the correct answer is 4).
* Verifiers can also execute code (e.g., in Python) to validate logic, syntax, and correctness without needing manual evaluation.

**Reward Function**:

* Converts verification results (or other criteria) into a numerical score.
* Example: If an answer is wrong, it might assign a penalty (-1, -2, etc.), while a correct answer could get a positive score (+1, +2).
* It can also penalize based on criteria beyond correctness, such as excessive length or poor readability.

**Key Differences**:

* A **Verifier** checks correctness but doesn’t score.
* A **Reward Function** assigns a score but doesn’t necessarily verify correctness itself.
* A Reward Function _can_ use a Verifier, but they are technically not the same.

### **Understanding Reward Functions**

GRPO's primary goal is to maximize reward and learn how an answer was derived, rather than simply memorizing and reproducing responses from its training data.

* With every training step, GRPO **adjusts model weights** to maximize the reward. This process fine-tunes the model incrementally.
* **Regular fine-tuning** (without GRPO) only **maximizes next-word prediction probability** but does not optimize for a reward. GRPO **optimizes for a reward function** rather than just predicting the next word.
* You can **reuse data** across multiple epochs.
* **Default reward functions** can be predefined to be used on a wide array of use cases or you can ask ChatGPT/local model to generate them for you.
* There’s no single correct way to design reward functions or verifiers - the possibilities are endless. However, they must be well-designed and meaningful, as poorly crafted rewards can unintentionally degrade model performance.

### :coin:Reward Function Examples

You can refer to the examples below. You can input your generations into an LLM like ChatGPT 4o or Llama 3.1 (8B) and design a reward function and verifier to evaluate it. For example, feed your generations into a LLM of your choice and set a rule: "If the answer sounds too robotic, deduct 3 points." This helps refine outputs based on quality criteria

#### **Example #1: Simple Arithmetic Task**

* **Question:** `"2 + 2"`
* **Answer:** `"4"`
* **Reward Function 1:**
  * If a number is detected → **+1**
  * If no number is detected → **-1**
* **Reward Function 2:**
  * If the number matches the correct answer → **+3**
  * If incorrect → **-3**
* **Total Reward:** _Sum of all reward functions_

#### **Example #2: Email Automation Task**

* **Question:** Inbound email
* **Answer:** Outbound email
* **Reward Functions:**
  * If the answer contains a required keyword → **+1**
  * If the answer exactly matches the ideal response → **+1**
  * If the response is too long → **-1**
  * If the recipient's name is included → **+1**
  * If a signature block (phone, email, address) is present → **+1**

### Unsloth Proximity-Based Reward Function

If you’ve checked out our [**Advanced GRPO Colab Notebook**](#grpo-notebooks), you’ll notice we’ve created a **custom proximity-based reward function** built completely from scratch, which is designed to reward answers that are closer to the correct one. This flexible function can be applied across a wide range of tasks.

* In our examples, we enable reasoning in Qwen3 (Base) and guide it toward specific tasks
* Apply Pre-finetuning strategies to avoid GRPO’s default tendency to just learn formatting
* Boost evaluation accuracy with regex-based matching
* Create custom GRPO templates beyond generic prompts like `think`, e.g., `<start_working_out></end_working_out>`
* Apply proximity-based scoring — models get more reward for closer answers (e.g., predicting 9 instead of 10 is better than 3) while outliers are penalized

#### GSM8K Reward Functions

In our other examples, we use existing GSM8K reward functions by [@willccbb](https://x.com/willccbb) which is popular and shown to be quite effective:

* **correctness\_reward\_func** – Rewards exact label matches.
* **int\_reward\_func** – Encourages integer-only answers.
* **soft\_format\_reward\_func** – Checks structure but allows minor newline mismatches.
* **strict\_format\_reward\_func** – Ensures response structure matches the prompt, including newlines.
* **xmlcount\_reward\_func** – Ensures exactly one of each XML tag in the response.

## :abacus:Using vLLM

You can now use [vLLM](https://github.com/vllm-project/vllm/) directly in your finetuning stack, which allows for much more throughput and allows you to finetune and do inference on the model at the same time! On 1x A100 40GB, expect 4000 tokens / s or so with Unsloth’s dynamic 4bit quant of Llama 3.2 3B Instruct. On a 16GB Tesla T4 (free Colab GPU), you can get 300 tokens / s.\
\
We also magically removed double memory usage when loading vLLM and Unsloth together, allowing for savings of 5GB or so for Llama 3.1 8B and 3GB for Llama 3.2 3B. Unsloth could originally finetune Llama 3.3 70B Instruct in 1x 48GB GPU with Llama 3.3 70B weights taking 40GB of VRAM. If we do not remove double memory usage, then we’ll need >= 80GB of VRAM when loading Unsloth and vLLM together.\
\
But with Unsloth, you can still finetune and get the benefits of fast inference in one package in under 48GB of VRAM! To use fast inference, first install vllm, and instantiate Unsloth with fast\_inference:

```
pip install unsloth vllm
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct",
    fast_inference = True,
)
model.fast_generate(["Hello!"])
```

## :white\_check\_mark:GRPO Requirement Guidelines

When you’re using Unsloth to do GRPO, we smartly reduce VRAM usage by over 90% when compared to standard implementations with Flash Attention 2 by using multiple tricks! On 20K context lengths for example with 8 generations per prompt, Unsloth uses only **54.3GB of VRAM for Llama 3.1 8B**, whilst standard implementations take **510.8GB (90% less for Unsloth)**.

1. For GRPO's **GPU VRAM requirements for QLoRA 4-bit**, the general rule is the model parameters = the amount of VRAM you will need (you can use less VRAM but this just to be safe). The more context length you set, the more VRAM. LoRA 16-bit will use at minimum 4x more VRAM.
2. Our new memory efficient linear kernels for GRPO slashes memory usage by 8x or more. This shaves 68.5GB of memory, whilst being actually faster through the help of torch.compile!
3. We leverage our smart [Unsloth gradient checkpointing](https://unsloth.ai/blog/long-context) algorithm which we released a while ago. It smartly offloads intermediate activations to system RAM asynchronously whilst being only 1% slower. This shaves 52GB of memory.
4. Unsloth also uses the same GPU / CUDA memory space as the underlying inference engine (vLLM), unlike implementations in other packages. This shaves 16GB of memory.

| Metrics                                        | Unsloth            | Standard + FA2 |
| ---------------------------------------------- | ------------------ | -------------- |
| Training Memory Cost (GB)                      | 42GB               | 414GB          |
| GRPO Memory Cost (GB)                          | 9.8GB              | 78.3GB         |
| Inference Cost (GB)                            | 0GB                | 16GB           |
| Inference KV Cache for 20K context length (GB) | 2.5GB              | 2.5GB          |
| Total Memory Usage                             | 54.33GB (90% less) | 510.8GB        |

In typical standard GRPO implementations, you need to create 2 logits of size (8. 20K) to calculate the GRPO loss. This takes 2 \* 2 bytes \* 8 (num generations) \* 20K (context length) \* 128256 (vocabulary size) = 78.3GB in VRAM.

Unsloth shaves 8x memory usage for long context GRPO, so we need only an extra 9.8GB in extra VRAM for 20K context lengths!

We also need to from the KV Cache in 16bit. Llama 3.1 8B has 32 layers, and both K and V are 1024 in size. So memory usage for 20K context length = 2 \* 2 bytes \* 32 layers \* 20K context length \* 1024 = 2.5GB per batch. We would set the batch size for vLLM to 8, but we shall leave it at 1 for our calculations to save VRAM. Otherwise you will need 20GB for the KV cache.

## 🎥 Unsloth RL 3 hour Workshop Video

{% embed url="https://www.youtube.com/watch?v=OkEGJ5G3foU" %}

## :mortar\_board:Further Reading

1. Nathan Lambert's RLHF Book is a must! [https://rlhfbook.com/c/11-policy-gradients.html](https://rlhfbook.com/c/11-policy-gradients.html)
2. Yannic Kilcher's GRPO Youtube video is also a must! [https://www.youtube.com/watch?v=bAWV\_yrqx4w](https://www.youtube.com/watch?v=bAWV_yrqx4w)
3. We did a 3 hour workshop at AI Engineer World's Fair 2025. Slides are other material are at [https://docs.unsloth.ai/ai-engineers-2025](https://docs.unsloth.ai/ai-engineers-2025)
4. Advanced GRPO notebook via Unsloth. [https://docs.unsloth.ai/basics/reinforcement-learning-guide/tutorial-train-your-own-reasoning-model-with-grpo](https://docs.unsloth.ai/basics/reinforcement-learning-guide/tutorial-train-your-own-reasoning-model-with-grpo)
5. GRPO from a base model notebook: [https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3\_(4B)-GRPO.ipynb](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_\(4B\)-GRPO.ipynb)

---
description: >-
  Beginner's Guide to transforming a model like Llama 3.1 (8B) into a reasoning
  model by using Unsloth and GRPO.
---

# Tutorial: Train your own Reasoning model with GRPO

DeepSeek developed [GRPO](https://unsloth.ai/blog/grpo) (Group Relative Policy Optimization) to train their R1 reasoning models.

## Quickstart

These instructions are for our pre-made Google Colab [notebooks](../../get-started/unsloth-notebooks). If you are installing Unsloth locally, you can also copy our notebooks inside your favorite code editor.&#x20;

#### The GRPO notebooks we are using: [Gemma 3 (1B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_\(1B\)-GRPO.ipynb), [Llama 3.1 (8B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/HuggingFace%20Course-Gemma3_\(1B\)-GRPO.ipynb), [Phi-4 (14B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4_\(14B\)-GRPO.ipynb) and [Qwen2.5 (3B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_\(3B\)-GRPO.ipynb)

{% stepper %}
{% step %}
### Install Unsloth

If you're using our Colab notebook, click **Runtime > Run all**. We'd highly recommend you checking out our [Fine-tuning Guide](../../get-started/fine-tuning-llms-guide) before getting started.

If installing locally, ensure you have the correct [requirements](../../get-started/beginner-start-here/unsloth-requirements) and use `pip install unsloth` on Linux or follow our [Windows install ](../../get-started/installing-+-updating/windows-installation)instructions.

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FCovHTH7dI2GcwNZm5TxF%2Fimage.png?alt=media&#x26;token=a157e33b-ad01-4174-a01c-67f742e4e732" alt=""><figcaption></figcaption></figure>
{% endstep %}

{% step %}
### Learn about GRPO & Reward Functions

Before we get started, it is recommended to learn more about GRPO, reward functions and how they work. Read more about them including [tips & tricks](..#basics-tips)[ here](..#basics-tips).

You will also need enough VRAM. In general, model parameters = amount of VRAM you will need.  In Colab, we are using their free 16GB VRAM GPUs which can train any model up to 16B in parameters.
{% endstep %}

{% step %}
### Configure desired settings

We have pre-selected optimal settings for the best results for you already and you can change the model to whichever you want listed in our [supported models](../../get-started/all-our-models). Would not recommend changing other settings if you're a beginner.

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2Fyd3RkyPKInZBbvX1Memf%2Fimage.png?alt=media&#x26;token=a9ca4ce4-2e9f-4b5a-a65c-646d267411c8" alt="" width="563"><figcaption></figcaption></figure>
{% endstep %}

{% step %}
### Data preparation

We have pre-selected OpenAI's [GSM8K](https://huggingface.co/datasets/openai/gsm8k) dataset which contains grade school math problems but you could change it to your own or any public one on Hugging Face. You can read more about [datasets here](../datasets-guide).

Your dataset should still have at least 2 columns for question and answer pairs. However the answer must not reveal the reasoning behind how it derived the answer from the question. See below for an example:

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FqdTVcMEeJ3kzPToSY1X8%2Fimage.png?alt=media&#x26;token=3dd8d9d7-1847-42b6-a73a-f9c995b798b1" alt=""><figcaption></figcaption></figure>

We'll structure the data to prompt the model to articulate its reasoning before delivering an answer. To start, we'll establish a clear format for both prompts and responses.

```
# Define the system prompt that instructs the model to use a specific format
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""
```

Now, to prepare the dataset:

```
import re
from datasets import load_dataset, Dataset


# Helper functions to extract answers from different formats
def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


# Function to prepare the GSM8K dataset
def get_gsm8k_questions(split="train") -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]
    data = data.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )
    return data


dataset = get_gsm8k_questions()
```

The dataset is prepared by extracting the answers and formatting them as structured strings.
{% endstep %}

{% step %}
### Reward Functions/Verifier

[Reward Functions/Verifiers](..#reward-functions-verifier) lets us know if the model is doing well or not according to the dataset you have provided. Each generation run will be assessed on how it performs to the score of the average of the rest of generations. You can create your own reward functions however we have already pre-selected them for you with [Will's GSM8K](..#gsm8k-reward-functions) reward functions. With this, we have 5 different ways which we can reward each generation.

You can input your generations into an LLM like ChatGPT 4o or Llama 3.1 (8B) and design a reward function and verifier to evaluate it. For example, feed your generations into a LLM of your choice and set a rule: "If the answer sounds too robotic, deduct 3 points." This helps refine outputs based on quality criteria. **See examples** of what they can look like [here](..#reward-function-examples).

**Example Reward Function for an Email Automation Task:**

* **Question:** Inbound email
* **Answer:** Outbound email
* **Reward Functions:**
  * If the answer contains a required keyword → **+1**
  * If the answer exactly matches the ideal response → **+1**
  * If the response is too long → **-1**
  * If the recipient's name is included → **+1**
  * If a signature block (phone, email, address) is present → **+1**

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2F6GRcqgUKmKn2dWCk4nWK%2Fimage.png?alt=media&#x26;token=ac153141-03f8-4795-9074-ad592289bd70" alt=""><figcaption></figcaption></figure>
{% endstep %}

{% step %}
### Train your model

We have pre-selected hyperparameters for the most optimal results however you could change them. Read all about [parameters here](../../get-started/fine-tuning-llms-guide/lora-hyperparameters-guide).

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2F1MpLSyaOH3j8MhQvquqX%2Fimage.png?alt=media&#x26;token=818034b1-f2db-464d-a108-3b2c6897edb7" alt="" width="563"><figcaption></figcaption></figure>

The **GRPOConfig** defines key hyperparameters for training:

* `use_vllm`: Activates fast inference using vLLM.
* `learning_rate`: Determines the model's learning speed.
* `num_generations`: Specifies the number of completions generated per prompt.
* `max_steps`: Sets the total number of training steps.

{% hint style="success" %}
**NEW!** We now support DAPO, Dr. GRPO and most other new GRPO techniques. You can play with the following arguments in GRPOConfig to enable:

```python
epsilon=0.2,
epsilon_high=0.28, # one sided
delta=1.5 # two sided

loss_type='bnpo',
# or:
loss_type='grpo',
# or:
loss_type='dr_grpo',
# or:
loss_type='dapo',

mask_truncated_completions=True,
```
{% endhint %}

You should see the reward increase overtime. We would recommend you train for at least 300 steps which may take 30 mins however, for optimal results, you should train for longer.

{% hint style="warning" %}
If you're having issues with your GRPO model not learning, we'd highly recommend to use our [Advanced GRPO notebooks](../../../get-started/unsloth-notebooks#grpo-reasoning-notebooks) as it has a much better reward function and you should see results much faster and frequently.
{% endhint %}

You will also see sample answers which allows you to see how the model is learning. Some may have steps, XML tags, attempts etc. and the idea is as trains it's going to get better and better because it's going to get scored higher and higher until we get the outputs we desire with long reasoning chains of answers.

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FyRmUGe8laUKIl0RKwlE6%2Fimage.png?alt=media&#x26;token=3ff931cc-0d2b-4a9c-bbe1-b6289b22d157" alt="" width="563"><figcaption></figcaption></figure>
{% endstep %}

{% step %}
### Run & Evaluate your model

Run your model by clicking the play button. In the first example, there is usually no reasoning in the answer and in order to see the reasoning, we need to first save the LoRA weights we just trained with GRPO first using:

<pre><code><strong>model.save_lora("grpo_saved_lora")
</strong></code></pre>

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FkLHdlRVKN58tM7SGKp3O%2Fimage.png?alt=media&#x26;token=b43a8164-7eae-4ec4-bf59-976078f9be31" alt=""><figcaption><p>The first inference example run has no reasoning. You must load the LoRA and test it to reveal the reasoning.</p></figcaption></figure>

Then we load the LoRA and test it. Our reasoning model is much better - it's not always correct, since we only trained it for an hour or so - it'll be better if we extend the sequence length and train for longer!

You can then save your model to GGUF, Ollama etc. by following our [guide here](../../../get-started/fine-tuning-llms-guide#id-7.-running--saving-the-model).

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FYdz5ch20Ig8JlumBesle%2Fimage.png?alt=media&#x26;token=8aea2867-b8a8-470a-aa4b-a7b9cdd64c3c" alt=""><figcaption></figcaption></figure>

If you are still not getting any reasoning, you may have either trained for too less steps or your reward function/verifier was not optimal.
{% endstep %}

{% step %}
### Save your model

We have multiple options for saving your fine-tuned model, but we’ll focus on the easiest and most popular approaches which you can read more about [here](../running-and-saving-models)

**Saving in 16-bit Precision**

You can save the model with 16-bit precision using the following command:

```python
# Save to 16-bit precision
model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")
```

#### **Pushing to Hugging Face Hub**

To share your model, we’ll push it to the Hugging Face Hub using the `push_to_hub_merged` method. This allows saving the model in multiple quantization formats.

```python
# Push to Hugging Face Hub (requires a token)
model.push_to_hub_merged(
    "your-username/model-name", tokenizer, save_method="merged_16bit", token="your-token"
)
```

#### **Saving in GGUF Format for llama.cpp**

Unsloth also supports saving in **GGUF format**, making it compatible with **llama.cpp** and **Ollama**.

```python
model.push_to_hub_gguf(
    "your-username/model-name",
    tokenizer,
    quantization_method=["q4_k_m", "q8_0", "q5_k_m"],
    token="your-token",
)
```

Once saved in GGUF format, the model can be easily deployed in lightweight environments using **llama.cpp** or used in other inference engines.
{% endstep %}
{% endstepper %}

## Video Tutorials

Here are some video tutorials created by amazing YouTubers who we think are fantastic!

{% embed url="https://www.youtube.com/watch?v=SoPE1cUz3Hs" %}
Local GRPO on your own device
{% endembed %}

{% embed url="https://www.youtube.com/watch?t=3289s&v=bbFEYPx9Hpo" %}
Great to learn about how to prep your dataset and explanations behind Reinforcement Learning + GRPO basics
{% endembed %}

{% embed url="https://www.youtube.com/watch?v=juOh1afy-IE" %}

{% embed url="https://www.youtube.com/watch?v=oF0_eMhzRaQ" %}

---
description: >-
  To use the reward modelling functions for DPO, GRPO, ORPO or KTO with Unsloth,
  follow the steps below:
---

# Reinforcement Learning - DPO, ORPO & KTO

DPO (Direct Preference Optimization), ORPO (Odds Ratio Preference Optimization), PPO, KTO Reward Modelling all work with Unsloth.

We have Google Colab notebooks for reproducing GRPO, ORPO, DPO Zephyr, KTO and SimPO:

* [GRPO notebooks](../../../get-started/unsloth-notebooks#grpo-reasoning-rl-notebooks)
* [ORPO notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_\(8B\)-ORPO.ipynb)
* [DPO Zephyr notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Zephyr_\(7B\)-DPO.ipynb)
* [KTO notebook](https://colab.research.google.com/drive/1MRgGtLWuZX4ypSfGguFgC-IblTvO2ivM?usp=sharing)
* [SimPO notebook](https://colab.research.google.com/drive/1Hs5oQDovOay4mFA6Y9lQhVJ8TnbFLFh2?usp=sharing)

We're also in 🤗Hugging Face's official docs! We're on the [SFT docs](https://huggingface.co/docs/trl/main/en/sft_trainer#accelerate-fine-tuning-2x-using-unsloth) and the [DPO docs](https://huggingface.co/docs/trl/main/en/dpo_trainer#accelerate-dpo-fine-tuning-using-unsloth).

## DPO Code

```python
python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Optional set GPU device ID

from unsloth import FastLanguageModel, PatchDPOTrainer
from unsloth import is_bfloat16_supported
PatchDPOTrainer()
import torch
from transformers import TrainingArguments
from trl import DPOTrainer

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/zephyr-sft-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r = 64,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 64,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    max_seq_length = max_seq_length,
)

dpo_trainer = DPOTrainer(
    model = model,
    ref_model = None,
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 8,
        warmup_ratio = 0.1,
        num_train_epochs = 3,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        seed = 42,
        output_dir = "outputs",
    ),
    beta = 0.1,
    train_dataset = YOUR_DATASET_HERE,
    # eval_dataset = YOUR_DATASET_HERE,
    tokenizer = tokenizer,
    max_length = 1024,
    max_prompt_length = 512,
)
dpo_trainer.train()
```

---
description: >-
  Learn how to train AI agents for real-world tasks using Reinforcement Learning
  (RL).
---

# Training AI Agents with RL

“Agentic” AI is becoming more popular over time. In this context, an “agent” is an LLM that is given a high-level goal and a set of tools to achieve it. Agents are also typically “multi-turn” — they can perform an action, see what effect it had on the environment, and then perform another action repeatedly, until they achieve their goal or fail trying.

Unfortunately, even very capable LLMs can have a hard time performing complex multi-turn agentic tasks reliably. Interestingly, we’ve found that training agents using an RL algorithm called [GRPO (Group Relative Policy Optimization)](tutorial-train-your-own-reasoning-model-with-grpo) can make them far more reliable! In this guide, you will learn how to to build reliable AI agents using open-source tools.

## 🎨 Training RL Agents with ART

[ART (Agent Reinforcement Trainer)](https://github.com/openpipe/art) built on top of [Unsloth](https://github.com/unslothai/unsloth)’s GRPOTrainer, is a tool that makes training multi-turn agents possible and easy. If you’re already using Unsloth for GRPO and need to train agents that can handle complex, multi-turn interactions, ART simplifies the process.

<div align="left"><figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FlIJVwkW3jsINbrOndI8w%2FScreenshot_2025-07-19_at_1.23.18_PM.webp?alt=media&#x26;token=5fce0b1b-870a-4c5b-a647-ba9688977eb3" alt="" width="375"><figcaption><p>Agent models trained with Unsloth+ART are often able to outperform prompted models on agentic workflows.</p></figcaption></figure></div>

### ART + Unsloth

ART builds on top of Unsloth’s memory- and compute-efficient GRPO implementation. In addition, it adds the following capabilities:

#### 1. Multi-Turn Agent Training

ART introduces the concept of a “trajectory”, which is built up as your agent executes. These trajectories can then be scored and used for GRPO. Trajectories can be complex, and even include non-linear histories, sub-agent calls, etc. They also support tool calls and responses.

#### 2. Flexible Integration into Existing Codebases

If you already have an agent working with a prompted model, ART tries to minimize the number of changes you need to make to wrap your existing agent loop and use it for training.

Architecturally, ART is split into a “frontend” client that lives in your codebase and communicates via API with a “backend” where the actual training happens (these can also be colocated on a single machine if you prefer using ART’s `LocalBackend`). This gives some key benefits:

* **Minimal setup required**: The ART frontend is has minimal dependencies and can be easily added to existing Python codebases.
* **Train from anywhere**: You can run the ART client on your laptop and let the ART server kick off an ephemeral GPU-enabled environment, or run on a local GPU
* **OpenAI-compatible API**: The ART backend serves your model undergoing training via an OpenAI-compatible API, which is compatible with most existing codebases.

#### 3. RULER: Zero-Shot Agent Rewards

ART also provides a built-in general-purpose reward function called [RULER](https://art.openpipe.ai/fundamentals/ruler) (Relative Universal LLM-Elicited Rewards), which can eliminate the need for hand-crafted reward functions. Surprisingly, agents RL-trained with the RULER automatic reward function often match or surpass the performance of agents trained using hand-written reward functions. This makes getting started with RL easier.

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FYAeJzbOFGTNZM9r5Qkrq%2FScreenshot_2025-07-19_at_1.21.08_PM.webp?alt=media&#x26;token=a7b67644-f6ac-4d4f-a690-7b160948f9d2" alt="" width="375"><figcaption></figcaption></figure>

```python
# Before: Hours of reward engineering
def complex_reward_function(trajectory):
    # 50+ lines of careful scoring logic...
    pass

# After: One line with RULER
judged_group = await ruler_score_group(group, "openai/o3")
```

### When to Choose ART

ART might be a good fit for projects that need:

1. **Multi-step agent capabilities**: When your use case involves agents that need to take multiple actions, use tools, or have extended conversations
2. **Rapid prototyping without reward engineering**: RULER’s automatic reward scoring can cut your project’s development time by 2-3x
3. **Integration with existing systems**: When you need to add RL capabilities to an existing agentic codebase with minimal changes

### Code Example: ART in Action

```python
import art
from art.rewards import ruler_score_group

# Initialize model with Unsloth-supported basemodel
model = art.TrainableModel(
    name="agent-001",
    project="my-agentic-task",
    base_model="Qwen/Qwen2.5-14B-Instruct",  # Any Unsloth-supported model
)

# Define your rollout function
async def rollout(model: art.Model, scenario: Scenario) -> art.Trajectory:
    openai_client = model.openai_client()
    trajectory = art.Trajectory(
        messages_and_choices=[
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."}
        ]
    )
    # Your agent logic here...    
    return trajectory

# Train with RULER for automatic rewards
groups = await art.gather_trajectory_groups(
    (
        art.TrajectoryGroup(rollout(model, scenario) for _ in range(8))
        for scenario in scenarios
    ),
    after_each=lambda group: ruler_score_group(
        group,
        "openai/o3",
        swallow_exceptions=True
    )
)

await model.train(groups)
```

### Getting Started

To add ART to your Unsloth-based project:

```bash
pip install openpipe-art # or `uv add openpipe-art`
```

Then check out the [example notebooks](https://art.openpipe.ai/getting-started/notebooks) to see ART in action with tasks like:

* Email retrieval agents that beat o3
* Game-playing agents (2048, Tic Tac Toe, Codenames)
* Complex reasoning tasks (Temporal Clue)

---
description: Learn how to create & prepare a dataset for fine-tuning.
---

# Datasets Guide

## What is a Dataset?

For LLMs, datasets are collections of data that can be used to train our models. In order to be useful for training, text data needs to be in a format that can be tokenized. You'll also learn how to [use datasets inside of Unsloth](#applying-chat-templates-with-unsloth).

One of the key parts of creating a dataset is your [chat template](chat-templates) and how you are going to design it. Tokenization is also important as it breaks text into tokens, which can be words, sub-words, or characters so LLMs can process it effectively. These tokens are then turned into embeddings and are adjusted to help the model understand the meaning and context.

### Data Format

To enable the process of tokenization, datasets need to be in a format that can be read by a tokenizer.

<table data-full-width="false"><thead><tr><th>Format</th><th>Description </th><th>Training Type</th></tr></thead><tbody><tr><td>Raw Corpus</td><td>Raw text from a source such as a website, book, or article.</td><td>Continued Pretraining (CPT)</td></tr><tr><td>Instruct</td><td>Instructions for the model to follow and an example of the output to aim for.</td><td>Supervised fine-tuning (SFT)</td></tr><tr><td>Conversation</td><td>Multiple-turn conversation between a user and an AI assistant.</td><td>Supervised fine-tuning (SFT)</td></tr><tr><td>RLHF</td><td>Conversation between a user and an AI assistant, with the assistant's responses being ranked by a script, another model or human evaluator.</td><td>Reinforcement Learning (RL)</td></tr></tbody></table>

{% hint style="info" %}
It's worth noting that different styles of format exist for each of these types.&#x20;
{% endhint %}

## Getting Started

Before we format our data, we want to identify the following:&#x20;

{% stepper %}
{% step %}
<mark style="color:green;">Purpose of dataset</mark>

Knowing the purpose of the dataset will help us determine what data we need and format to use.

The purpose could be, adapting a model to a new task such as summarization or improving a model's ability to role-play a specific character. For example:

* Chat-based dialogues (Q\&A, learn a new language, customer support, conversations).
* Structured tasks ([classification](https://colab.research.google.com/github/timothelaborie/text_classification_scripts/blob/main/unsloth_classification.ipynb), summarization, generation tasks).
* Domain-specific data (medical, finance, technical).
{% endstep %}

{% step %}
<mark style="color:green;">Style of output</mark>

The style of output will let us know what sources of data we will use to reach our desired output.

For example, the type of output you want to achieve could be JSON, HTML, text or code. Or perhaps you want it to be Spanish, English or German etc.&#x20;
{% endstep %}

{% step %}
<mark style="color:green;">Data source</mark>

When we know the purpose and style of the data we need, we need to analyze the quality and [quantity](#how-big-should-my-dataset-be) of the data. Hugging Face and Wikipedia are great sources of datasets and Wikipedia is especially useful if you are looking to train a model to learn a language.

The Source of data can be a CSV file, PDF or even a website. You can also [synthetically generate](#synthetic-data-generation) data but extra care is required to make sure each example is high quality and relevant.
{% endstep %}
{% endstepper %}

{% hint style="success" %}
One of the best ways to create a better dataset is by combining it with a more generalized dataset from Hugging Face like ShareGPT to make your model smarter and diverse. You could also add [synthetically generated data](#synthetic-data-generation).
{% endhint %}

## Formatting the Data

When we have identified the relevant criteria, and collected the necessary data, we can then format our data into a machine readable format that is ready for training.

### Common Data Formats for LLM Training

For [**continued pretraining**](continued-pretraining), we use raw text format without specific structure:

```json
  "text": "Pasta carbonara is a traditional Roman pasta dish. The sauce is made by mixing raw eggs with grated Pecorino Romano cheese and black pepper. The hot pasta is then tossed with crispy guanciale (cured pork cheek) and the egg mixture, creating a creamy sauce from the residual heat. Despite popular belief, authentic carbonara never contains cream or garlic. The dish likely originated in Rome in the mid-20th century, though its exact origins are debated..."
```

This format preserves natural language flow and allows the model to learn from continuous text.

If we are adapting a model to a new task, and intend for the model to output text in a single turn based on a specific set of instructions, we can use **Instruction** format in [Alpaca style](https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama#id-6.-alpaca-dataset)

```json
"Instruction": "Task we want the model to perform."

"Input": "Optional, but useful, it will essentially be the user's query."

"Output": "The expected result of the task and the output of the model."
```

When we want multiple turns of conversation we can use the ShareGPT format:

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "Can you help me make pasta carbonara?"
    },
    {
      "from": "gpt",
      "value": "Would you like the traditional Roman recipe, or a simpler version?"
    },
    {
      "from": "human",
      "value": "The traditional version please"
    },
    {
      "from": "gpt",
      "value": "The authentic Roman carbonara uses just a few ingredients: pasta, guanciale, eggs, Pecorino Romano, and black pepper. Would you like the detailed recipe?"
    }
  ]
}
```

The template format uses the "from"/"value" attribute keys and messages alternates between `human`and `gpt`, allowing for natural dialogue flow.

The other common format is OpenAI's ChatML format and is what Hugging Face defaults to. This is probably the most used format, and alternates between `user` and `assistant`

```
{
  "messages": [
    {
      "role": "user",
      "content": "What is 1+1?"
    },
    {
      "role": "assistant",
      "content": "It's 2!"
    },
  ]
}
```

### Applying Chat Templates with Unsloth

For datasets that usually follow the common chatml format, the process of preparing the dataset for training or finetuning, consists of four simple steps:

*   Check the chat templates that Unsloth currently supports:\


    ```
    from unsloth.chat_templates import CHAT_TEMPLATES
    print(list(CHAT_TEMPLATES.keys()))
    ```

    \
    This will print out the list of templates currently supported by Unsloth. Here is an example output:\


    ```
    ['unsloth', 'zephyr', 'chatml', 'mistral', 'llama', 'vicuna', 'vicuna_old', 'vicuna old', 'alpaca', 'gemma', 'gemma_chatml', 'gemma2', 'gemma2_chatml', 'llama-3', 'llama3', 'phi-3', 'phi-35', 'phi-3.5', 'llama-3.1', 'llama-31', 'llama-3.2', 'llama-3.3', 'llama-32', 'llama-33', 'qwen-2.5', 'qwen-25', 'qwen25', 'qwen2.5', 'phi-4', 'gemma-3', 'gemma3']
    ```

    \

*   Use `get_chat_template` to apply the right chat template to your tokenizer:\


    ```
    from unsloth.chat_templates import get_chat_template

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "gemma-3", # change this to the right chat_template name
    )
    ```

    \

*   Define your formatting function. Here's an example:\


    ```
    def formatting_prompts_func(examples):
       convos = examples["conversations"]
       texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
       return { "text" : texts, }
    ```

    \
    \
    This function loops through your dataset applying the chat template you defined to each sample.\

*   Finally, let's load the dataset and apply the required modifications to our dataset: \


    ```
    # Import and load dataset
    from datasets import load_dataset
    dataset = load_dataset("repo_name/dataset_name", split = "train")

    # Apply the formatting function to your dataset using the map method
    dataset = dataset.map(formatting_prompts_func, batched = True,)
    ```

    \
    If your dataset uses the ShareGPT format with "from"/"value" keys instead of the ChatML "role"/"content" format, you can use the `standardize_sharegpt` function to convert it first. The revised code will now look as follows:\
    \


    ```
    # Import dataset
    from datasets import load_dataset
    dataset = load_dataset("mlabonne/FineTome-100k", split = "train")

    # Convert your dataset to the "role"/"content" format if necessary
    from unsloth.chat_templates import standardize_sharegpt
    dataset = standardize_sharegpt(dataset)

    # Apply the formatting function to your dataset using the map method
    dataset = dataset.map(formatting_prompts_func, batched = True,)
    ```



### Formatting Data Q\&A

<mark style="color:green;">**Q:**</mark> How can I use the Alpaca instruct format?&#x20;

<mark style="color:green;">**A:**</mark>  If your dataset is already formatted in the Alpaca format, then follow the formatting steps as shown in the Llama3.1 [notebook ](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_\(8B\)-Alpaca.ipynb#scrollTo=LjY75GoYUCB8). If you need to convert your data to the Alpaca format, one approach is to create a Python script to process your raw data. If you're working on a summarization task, you can use a local LLM to generate instructions and outputs for each example.&#x20;



<mark style="color:green;">**Q:**</mark> Should I always use the standardize\_sharegpt method?

<mark style="color:green;">**A:**</mark>  Only use the standardize\_sharegpt method if your target dataset is formatted in the sharegpt format, but your model expect a ChatML format instead.

\
<mark style="color:green;">**Q:**</mark> Why not use the apply\_chat\_template function that comes with the tokenizer.

<mark style="color:green;">**A:**</mark>  The `chat_template` attribute when a model is first uploaded by the original model owners sometimes contains errors and may take time to be updated. In contrast, at Unsloth, we thoroughly check and fix any errors in the `chat_template` for every model when we upload the quantized versions to our repositories. Additionally, our `get_chat_template` and `apply_chat_template` methods offer advanced data manipulation features, which are fully documented on our Chat Templates documentation [page](https://docs.unsloth.ai/basics/chat-templates).&#x20;



<mark style="color:green;">**Q:**</mark> What if my template is not currently supported by Unsloth?

<mark style="color:green;">**A:**</mark>  Submit a feature request on the unsloth github issues [forum](https://github.com/unslothai/unsloth). As a temporary workaround, you could also use the tokenizer's own apply\_chat\_template function until your feature request is approved and merged.

## Synthetic Data Generation

You can also use any local LLM like Llama 3.3 (70B) or OpenAI's GPT 4.5 to generate synthetic data. Generally, it is better to use a bigger like Llama 3.3 (70B) to ensure the highest quality outputs. You can directly use inference engines like vLLM, Ollama or llama.cpp to generate synthetic data but it will require some manual work to collect it and prompt for more data. There's 3 goals for synthetic data:

* Produce entirely new data - either from scratch or from your existing dataset
* Diversify your dataset so your model does not [overfit](../../get-started/fine-tuning-llms-guide/lora-hyperparameters-guide#avoiding-overfitting-and-underfitting) and become too specific
* Augment existing data e.g. automatically structure your dataset in the correct chosen format

### Synthetic Dataset Notebook

We collaborated with Meta to launch a free notebook for creating Synthetic Datasets automatically using local models like Llama 3.2. [Access the notebook here.](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Meta_Synthetic_Data_Llama3_2_\(3B\).ipynb)

What the notebook does:

* Auto-parses PDFs, websites, YouTube videos and more
* Uses Meta’s Synthetic Data Kit + Llama 3.2 (3B) to generate QA pairs
* Cleans and filters the data automatically
* Fine-tunes the dataset with Unsloth + Llama
* Notebook is fully done locally with no API calling necessary

### Using a local LLM or ChatGPT for synthetic data

Your goal is to prompt the model to generate and process QA data that is in your specified format. The model will need to learn the structure that you provided and also the context so ensure you at least have 10 examples of data already. Examples prompts:

*   **Prompt for generating more dialogue on an existing dataset**:

    <pre data-overflow="wrap"><code><strong>Using the dataset example I provided, follow the structure and generate conversations based on the examples.
    </strong></code></pre>
*   **Prompt if you no have dataset**:

    {% code overflow="wrap" %}
    ```
    Create 10 examples of product reviews for Coca-Coca classified as either positive, negative, or neutral.
    ```
    {% endcode %}
*   **Prompt for a dataset without formatting**:

    {% code overflow="wrap" %}
    ```
    Structure my dataset so it is in a QA ChatML format for fine-tuning. Then generate 5 synthetic data examples with the same topic and format.
    ```
    {% endcode %}

It is recommended to check the quality of generated data to remove or improve on irrelevant or poor-quality responses. Depending on your dataset it may also have to be balanced in many areas so your model does not overfit. You can then feed this cleaned dataset back into your LLM to regenerate data, now with even more guidance.

## Dataset FAQ + Tips

### How big should my dataset be?

We generally recommend using a bare minimum of at least 100 rows of data for fine-tuning to achieve reasonable results. For optimal performance, a dataset with over 1,000 rows is preferable, and in this case, more data usually leads to better outcomes. If your dataset is too small you can also add synthetic data or add a dataset from Hugging Face to diversify it. However, the effectiveness of your fine-tuned model depends heavily on the quality of the dataset, so be sure to thoroughly clean and prepare your data.

### How should I structure my dataset if I want to fine-tune a reasoning model?

If you want to fine-tune a model that already has reasoning capabilities like the distilled versions of DeepSeek-R1 (e.g. DeepSeek-R1-Distill-Llama-8B), you will need to still follow question/task and answer pairs however, for your answer you will need to change the answer so it includes reasoning/chain-of-thought process and the steps it took to derive the answer.\
\
For a model that does not have reasoning and you want to train it so that it later encompasses reasoning capabilities, you will need to utilize a standard dataset but this time without reasoning in its answers. This is training process is known as [Reinforcement Learning and GRPO](reinforcement-learning-rl-guide).

### Multiple datasets

If you have multiple datasets for fine-tuning, you can either:

* Standardize the format of all datasets, combine them into a single dataset, and fine-tune on this unified dataset.
* Use the [Multiple Datasets](https://colab.research.google.com/drive/1njCCbE1YVal9xC83hjdo2hiGItpY_D6t?usp=sharing) notebook to fine-tune on multiple datasets directly.

### Can I fine-tune the same model multiple times?

You can fine-tune an already fine-tuned model multiple times, but it's best to combine all the datasets and perform the fine-tuning in a single process instead. Training an already fine-tuned model can potentially alter the quality and knowledge acquired during the previous fine-tuning process.

## Using Datasets in Unsloth

### Alpaca Dataset

See an example of using the Alpaca dataset inside of Unsloth on Google Colab:

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FKSmRDpkySelZfWSrWxDm%2Fimage.png?alt=media&#x26;token=5401e4da-796a-42ad-8b85-2263f3e59e86" alt=""><figcaption></figcaption></figure>

We will now use the Alpaca Dataset created by calling GPT-4 itself. It is a list of 52,000 instructions and outputs which was very popular when Llama-1 was released, since it made finetuning a base LLM be competitive with ChatGPT itself.

You can access the GPT4 version of the Alpaca dataset [here](https://huggingface.co/datasets/vicgalle/alpaca-gpt4.). Below shows some examples of the dataset:

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FzKhujR9Nxz95VFSdf4J5%2Fimage.png?alt=media&#x26;token=a3c52718-eaf1-4a3d-b325-414d8e67722e" alt=""><figcaption></figcaption></figure>

You can see there are 3 columns in each row - an instruction, and input and an output. We essentially combine each row into 1 large prompt like below. We then use this to finetune the language model, and this made it very similar to ChatGPT. We call this process **supervised instruction finetuning**.

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FieYX44Vjd0OygJvO0jaR%2Fimage.png?alt=media&#x26;token=eb67fa41-a280-4656-8be6-5b6bf6f587c2" alt=""><figcaption></figcaption></figure>

### Multiple columns for finetuning

But a big issue is for ChatGPT style assistants, we only allow 1 instruction / 1 prompt, and not multiple columns / inputs. For example in ChatGPT, you can see we must submit 1 prompt, and not multiple prompts.

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FpFUWhntUQLu05l4ns7Pq%2Fimage.png?alt=media&#x26;token=e989e4a6-6033-4741-b97f-d0c3ce8f5888" alt=""><figcaption></figcaption></figure>

This essentially means we have to "merge" multiple columns into 1 large prompt for finetuning to actually function!

For example the very famous Titanic dataset has many many columns. Your job was to predict whether a passenger has survived or died based on their age, passenger class, fare price etc. We can't simply pass this into ChatGPT, but rather, we have to "merge" this information into 1 large prompt.

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FrydHBjHoJT7w8FwzKAXK%2FMerge-1.png?alt=media&#x26;token=ec812057-0475-4717-87fe-311f14735c37" alt=""><figcaption></figcaption></figure>

For example, if we ask ChatGPT with our "merged" single prompt which includes all the information for that passenger, we can then ask it to guess or predict whether the passenger has died or survived.

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FJVkv73fRWvwwFxMym7uW%2Fimage.png?alt=media&#x26;token=59b97b76-f2f2-46c9-8940-60a37e4e7d62" alt=""><figcaption></figcaption></figure>

Other finetuning libraries require you to manually prepare your dataset for finetuning, by merging all your columns into 1 prompt. In Unsloth, we simply provide the function called `to_sharegpt` which does this in 1 go!

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2F9fo2IBA7P0tNwhNR9Prm%2Fimage.png?alt=media&#x26;token=7bd7244a-0fea-4e57-9038-a8a360138056" alt=""><figcaption></figcaption></figure>

Now this is a bit more complicated, since we allow a lot of customization, but there are a few points:

* You must enclose all columns in curly braces `{}`. These are the column names in the actual CSV / Excel file.
* Optional text components must be enclosed in `[[]]`. For example if the column "input" is empty, the merging function will not show the text and skip this. This is useful for datasets with missing values.
* Select the output or target / prediction column in `output_column_name`. For the Alpaca dataset, this will be `output`.

For example in the Titanic dataset, we can create a large merged prompt format like below, where each column / piece of text becomes optional.

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FRMvBpfXC9ToCRL0oCJfN%2Fimage.png?alt=media&#x26;token=c257c7fc-8a9c-4d4f-ab3d-6894ae49f2a9" alt=""><figcaption></figcaption></figure>

For example, pretend the dataset looks like this with a lot of missing data:

| Embarked | Age | Fare |
| -------- | --- | ---- |
| S        | 23  |      |
|          | 18  | 7.25 |

Then, we do not want the result to be:

1. The passenger embarked from S. Their age is 23. Their fare is **EMPTY**.
2. The passenger embarked from **EMPTY**. Their age is 18. Their fare is $7.25.

Instead by optionally enclosing columns using `[[]]`, we can exclude this information entirely.

1. \[\[The passenger embarked from S.]] \[\[Their age is 23.]] \[\[Their fare is **EMPTY**.]]
2. \[\[The passenger embarked from **EMPTY**.]] \[\[Their age is 18.]] \[\[Their fare is $7.25.]]

becomes:

1. The passenger embarked from S. Their age is 23.
2. Their age is 18. Their fare is $7.25.

### Multi turn conversations

A bit issue if you didn't notice is the Alpaca dataset is single turn, whilst remember using ChatGPT was interactive and you can talk to it in multiple turns. For example, the left is what we want, but the right which is the Alpaca dataset only provides singular conversations. We want the finetuned language model to somehow learn how to do multi turn conversations just like ChatGPT.

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FWCAN7bYUt6QWwCWUxisL%2Fdiff.png?alt=media&#x26;token=29821fd9-2181-4d1d-8b93-749b69bcf400" alt=""><figcaption></figcaption></figure>

So we introduced the `conversation_extension` parameter, which essentially selects some random rows in your single turn dataset, and merges them into 1 conversation! For example, if you set it to 3, we randomly select 3 rows and merge them into 1! Setting them too long can make training slower, but could make your chatbot and final finetune much better!

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FWi1rRNBFC2iDmCvSJsZt%2Fcombine.png?alt=media&#x26;token=bef37a55-b272-4be3-89b5-9767c219a380" alt=""><figcaption></figcaption></figure>

Then set `output_column_name` to the prediction / output column. For the Alpaca dataset dataset, it would be the output column.

We then use the `standardize_sharegpt` function to just make the dataset in a correct format for finetuning! Always call this!

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FE75C4Y848VNF6luLuPRR%2Fimage.png?alt=media&#x26;token=aac1d79b-ecca-4e56-939d-d97dcbbf30eb" alt=""><figcaption></figcaption></figure>

## Vision Fine-tuning

The dataset for fine-tuning a vision or multimodal model also includes image inputs. For example, the [Llama 3.2 Vision Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_\(11B\)-Vision.ipynb#scrollTo=vITh0KVJ10qX) uses a radiography case to show how AI can help medical professionals analyze X-rays, CT scans, and ultrasounds more efficiently.

We'll be using a sampled version of the ROCO radiography dataset. You can access the dataset [here](https://www.google.com/url?q=https%3A%2F%2Fhuggingface.co%2Fdatasets%2Funsloth%2FRadiology_mini). The dataset includes X-rays, CT scans and ultrasounds showcasing medical conditions and diseases. Each image has a caption written by experts describing it. The goal is to finetune a VLM to make it a useful analysis tool for medical professionals.

Let's take a look at the dataset, and check what the 1st example shows:

```
Dataset({
    features: ['image', 'image_id', 'caption', 'cui'],
    num_rows: 1978
})
```

| Image                                                                                                                                                                                                                                                                                                        | Caption                                                                                                                                       |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| <p></p><div><figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FrjdETiyi6jqzAao7vg8I%2Fxray.png?alt=media&#x26;token=f66fdd7f-5e10-4eff-a280-5b3d63ed7849" alt="" width="164"><figcaption></figcaption></figure></div> | Panoramic radiography shows an osteolytic lesion in the right posterior maxilla with resorption of the floor of the maxillary sinus (arrows). |

To format the dataset, all vision finetuning tasks should be formatted as follows:

```python
[
{ "role": "user",
  "content": [{"type": "text",  "text": instruction}, {"type": "image", "image": image} ]
},
{ "role": "assistant",
  "content": [{"type": "text",  "text": answer} ]
},
]
```

We will craft an custom instruction asking the VLM to be an expert radiographer. Notice also instead of just 1 instruction, you can add multiple turns to make it a dynamic conversation.

```notebook-python
instruction = "You are an expert radiographer. Describe accurately what you see in this image."

def convert_to_conversation(sample):
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : instruction},
            {"type" : "image", "image" : sample["image"]} ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : sample["caption"]} ]
        },
    ]
    return { "messages" : conversation }
pass
```

Let's convert the dataset into the "correct" format for finetuning:

```notebook-python
converted_dataset = [convert_to_conversation(sample) for sample in dataset]
```

The first example is now structured like below:

```notebook-python
converted_dataset[0]
```

{% code overflow="wrap" %}
```
{'messages': [{'role': 'user',
   'content': [{'type': 'text',
     'text': 'You are an expert radiographer. Describe accurately what you see in this image.'},
    {'type': 'image',
     'image': <PIL.PngImagePlugin.PngImageFile image mode=L size=657x442>}]},
  {'role': 'assistant',
   'content': [{'type': 'text',
     'text': 'Panoramic radiography shows an osteolytic lesion in the right posterior maxilla with resorption of the floor of the maxillary sinus (arrows).'}]}]}
```
{% endcode %}

Before we do any finetuning, maybe the vision model already knows how to analyse the images? Let's check if this is the case!

```notebook-python
FastVisionModel.for_inference(model) # Enable for inference!

image = dataset[0]["image"]
instruction = "You are an expert radiographer. Describe accurately what you see in this image."

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": instruction}
    ]}
]
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
inputs = tokenizer(
    image,
    input_text,
    add_special_tokens = False,
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
                   use_cache = True, temperature = 1.5, min_p = 0.1)
```

And the result:

```
This radiograph appears to be a panoramic view of the upper and lower dentition, specifically an Orthopantomogram (OPG).

* The panoramic radiograph demonstrates normal dental structures.
* There is an abnormal area on the upper right, represented by an area of radiolucent bone, corresponding to the antrum.

**Key Observations**

* The bone between the left upper teeth is relatively radiopaque.
* There are two large arrows above the image, suggesting the need for a closer examination of this area. One of the arrows is in a left-sided position, and the other is in the right-sided position. However, only
```

For more details, view our dataset section in the [notebook here](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_\(11B\)-Vision.ipynb#scrollTo=vITh0KVJ10qX).

---
description: A big new upgrade to our Dynamic Quants!
---

# Unsloth Dynamic 2.0 GGUFs

We're excited to introduce our Dynamic v2.0 quantization method - a major upgrade to our previous quants. This new method outperforms leading quantization methods and sets new benchmarks for 5-shot MMLU and KL Divergence.

This means you can now run + fine-tune quantized LLMs while preserving as much accuracy as possible! You can run the 2.0 GGUFs on any inference engine like llama.cpp, Ollama, Open WebUI etc.

View all our Dynamic 2.0 GGUF models on [Hugging Face here](https://huggingface.co/collections/unsloth/unsloth-dynamic-v20-quants-68060d147e9b9231112823e6).&#x20;

{% hint style="success" %}
The **key advantage** of using the Unsloth package and models is our active role in _**fixing critical bugs**_ in major models. We've collaborated directly with teams behind [Qwen3](https://www.reddit.com/r/LocalLLaMA/comments/1kaodxu/qwen3_unsloth_dynamic_ggufs_128k_context_bug_fixes/), [Meta (Llama 4)](https://github.com/ggml-org/llama.cpp/pull/12889), [Mistral (Devstral)](https://app.gitbook.com/o/HpyELzcNe0topgVLGCZY/s/xhOjnexMCB3dmuQFQ2Zq/~/changes/618/basics/tutorials-how-to-fine-tune-and-run-llms/devstral-how-to-run-and-fine-tune), [Google (Gemma 1–3)](https://news.ycombinator.com/item?id=39671146) and [Microsoft (Phi-3/4)](https://simonwillison.net/2025/Jan/11/phi-4-bug-fixes), contributing essential fixes that significantly boost accuracy.
{% endhint %}

### 💡 What's New in Dynamic v2.0?

* **Revamped Layer Selection for GGUFs + safetensors:** Unsloth Dynamic 2.0 now selectively quantizes layers much more intelligently and extensively. Rather than modifying only select layers, we now dynamically adjust the quantization type of every possible layer, and the combinations will differ for each layer and model.
* Current selected and all future GGUF uploads will utilize Dynamic 2.0 and our new calibration dataset. The dataset contains more than >1.5M **tokens** (depending on model) and comprise of high-quality, hand-curated and cleaned data - to greatly enhance conversational chat performance.
* Previously, our Dynamic quantization (DeepSeek-R1 1.58-bit GGUF) was effective only for MoE architectures. <mark style="background-color:green;">**Dynamic 2.0 quantization now works on all models (including MOEs & non-MoEs)**</mark>.
* **Model-Specific Quants:** Each model now uses a custom-tailored quantization scheme. E.g. the layers quantized in Gemma 3 differ significantly from those in Llama 4.
* To maximize efficiency, especially on Apple Silicon and ARM devices, we now also add Q4\_NL, Q5.1, Q5.0, Q4.1, and Q4.0 formats.

To ensure accurate benchmarking, we built an internal evaluation framework to match official reported 5-shot MMLU scores of Llama 4 and Gemma 3. This allowed apples-to-apples comparisons between full-precision vs. Dynamic v2.0, **QAT** and standard **imatrix** GGUF quants.

Currently, we've released updates for:

| **Qwen3:** [0.6B](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF) • [1.7B](https://huggingface.co/unsloth/Qwen3-1.7B-GGUF) • [4B](https://huggingface.co/unsloth/Qwen3-4B-GGUF) • [8B](https://huggingface.co/unsloth/Qwen3-8B-GGUF) • [14B](https://huggingface.co/unsloth/Qwen3-14B-GGUF) • [30B-A3B](https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF) • [32B](https://huggingface.co/unsloth/Qwen3-32B-GGUF) • [235B-A22B](https://huggingface.co/unsloth/Qwen3-235B-A22B-GGUF) • [R1-0528](https://huggingface.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF) | **Other:** [GLM-4-32B](https://huggingface.co/unsloth/GLM-4-32B-0414-GGUF) • [MAI-DS-R1](https://huggingface.co/unsloth/MAI-DS-R1-GGUF) • [QwQ (32B)](https://huggingface.co/unsloth/QwQ-32B-GGUF)                                                            |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **DeepSeek:** [R1-0528](../deepseek-r1-0528-how-to-run-locally#model-uploads) • [V3-0324](https://huggingface.co/unsloth/DeepSeek-V3-0324-GGUF-UD) • [R1-Distill-Llama](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF)                                                                                                                                                                                                                                                                                                                       | **Llama:** [4 (Scout)](https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF) • [4 (Maverick)](https://huggingface.co/unsloth/Llama-4-Maverick-17B-128E-Instruct-GGUF) •  [3.1 (8B)](https://huggingface.co/unsloth/Llama-3.1-8B-Instruct-GGUF)  |
| **Gemma 3:** [4B](https://huggingface.co/unsloth/gemma-3-4b-it-GGUF) • [12B](https://huggingface.co/unsloth/gemma-3-12b-it-GGUF) • [27B](https://huggingface.co/unsloth/gemma-3-27b-it-GGUF) • [QAT](https://huggingface.co/unsloth/gemma-3-12b-it-qat-GGUF)                                                                                                                                                                                                                                                                                                    | **Mistral:** [Magistral](https://huggingface.co/unsloth/Magistral-Small-2506-GGUF) • [Small-3.1-2503](https://huggingface.co/unsloth/Mistral-Small-3.1-24B-Instruct-2503-GGUF)                                                                                |

All future GGUF uploads will utilize Unsloth Dynamic 2.0, and our Dynamic 4-bit safe tensor quants will also benefit from this in the future.

Detailed analysis of our benchmarks and evaluation further below.

<div><figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FWpuceJODVjlQcN7RvS6M%2Fkldivergence%20graph.png?alt=media&#x26;token=1f8f39fb-d4c6-47c6-84fe-f767ec7bae6b" alt="" width="563"><figcaption></figcaption></figure> <figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FszSmyqwqLW7artvIR5ut%2F5shotmmlu.png?alt=media&#x26;token=c9ef327e-5f8c-4720-8e05-08c345668745" alt="" width="563"><figcaption></figcaption></figure></div>

## 📊 Why KL Divergence?

[Accuracy is Not All You Need](https://arxiv.org/pdf/2407.09141) showcases how pruning layers, even by selecting unnecessary ones still yields vast differences in terms of "flips". A "flip" is defined as answers changing from incorrect to correct or vice versa. The paper shows how MMLU might not decrease as we prune layers or do quantization,but that's because some incorrect answers might have "flipped" to become correct. Our goal is to match the original model, so measuring "flips" is a good metric.

<div><figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FEjL8zLLNyceY3IpDUdWz%2Fimage.png?alt=media&#x26;token=6c31355b-57cf-4f22-a70e-b3b1e7c533d4" alt=""><figcaption></figcaption></figure> <figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FimYGCjWJ3GVKQmfAQwd5%2Fimage.png?alt=media&#x26;token=5a49d0ec-d92a-4d0e-9d6f-77f6d0d95738" alt=""><figcaption></figcaption></figure></div>

{% hint style="info" %}
**KL Divergence** should be the **gold standard for reporting quantization errors** as per the research paper "Accuracy is Not All You Need". **Using perplexity is incorrect** since output token values can cancel out, so we must use KLD!
{% endhint %}

The paper also shows that interestingly KL Divergence is highly correlated with flips, and so our goal is to reduce the mean KL Divergence whilst increasing the disk space of the quantization as less as possible.

## ⚖️ Calibration Dataset Overfitting

Most frameworks report perplexity and KL Divergence using a test set of Wikipedia articles. However, we noticed using the calibration dataset which is also Wikipedia related causes quants to overfit, and attain lower perplexity scores. We utilize [Calibration\_v3](https://gist.github.com/bartowski1182/eb213dccb3571f863da82e99418f81e8) and [Calibration\_v5](https://gist.github.com/tristandruyen/9e207a95c7d75ddf37525d353e00659c/) datasets for fair testing which includes some wikitext data amongst other data. <mark style="background-color:red;">**Also instruct models have unique chat templates, and using text only calibration datasets is not effective for instruct models**</mark> (base models yes). In fact most imatrix GGUFs are typically calibrated with these issues. As a result, they naturally perform better on KL Divergence benchmarks that also use Wikipedia data, since the model is essentially optimized for that domain.

To ensure a fair and controlled evaluation, we do not to use our own calibration dataset (which is optimized for chat performance) when benchmarking KL Divergence. Instead, we conducted tests using the same standard Wikipedia datasets, allowing us to directly compare the performance of our Dynamic 2.0 method against the baseline imatrix approach.

## :1234: MMLU Replication Adventure

* Replicating MMLU 5 shot was nightmarish. We <mark style="background-color:red;">**could not**</mark> replicate MMLU results for many models including Llama 3.1 (8B) Instruct, Gemma 3 (12B) and others due to <mark style="background-color:yellow;">**subtle implementation issues**</mark>. Llama 3.1 (8B) for example should be getting \~68.2%, whilst using incorrect implementations can attain <mark style="background-color:red;">**35% accuracy.**</mark>

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FGqqARO9UA0qpIzNcfixv%2FMMLU%20differences.png?alt=media&#x26;token=59c47844-a2e6-49a3-a523-1e28f2208e6d" alt="" width="375"><figcaption><p>MMLU implementation issues</p></figcaption></figure>

* Llama 3.1 (8B) Instruct has a MMLU 5 shot accuracy of 67.8% using a naive MMLU implementation. We find however Llama **tokenizes "A" and "\_A" (A with a space in front) as different token ids**. If we consider both spaced and non spaced tokens, we get 68.2% <mark style="background-color:green;">(+0.4%)</mark>
* Interestingly Llama 3 as per Eleuther AI's [LLM Harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/llama3/instruct/mmlu/_continuation_template_yaml) also appends <mark style="background-color:purple;">**"The best answer is"**</mark> to the question, following Llama 3's original MMLU benchmarks.
* There are many other subtle issues, and so to benchmark everything in a controlled environment, we designed our own MMLU implementation from scratch by investigating [github.com/hendrycks/test](https://github.com/hendrycks/test) directly, and verified our results across multiple models and comparing to reported numbers.

## :sparkles: Gemma 3 QAT Replication, Benchmarks

The Gemma team released two QAT (quantization aware training) versions of Gemma 3:

1. Q4\_0 GGUF - Quantizes all layers to Q4\_0 via the formula `w = q * block_scale` with each block having 32 weights. See [llama.cpp wiki ](https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes)for more details.
2. int4 version - presumably [TorchAO int4 style](https://github.com/pytorch/ao/blob/main/torchao/quantization/README.md)?

We benchmarked all Q4\_0 GGUF versions, and did extensive experiments on the 12B model. We see the **12B Q4\_0 QAT model gets 67.07%** whilst the full bfloat16 12B version gets 67.15% on 5 shot MMLU. That's very impressive! The 27B model is mostly nearly there!

<table><thead><tr><th>Metric</th><th>1B</th><th valign="middle">4B</th><th>12B</th><th>27B</th></tr></thead><tbody><tr><td>MMLU 5 shot</td><td>26.12%</td><td valign="middle">55.13%</td><td><mark style="background-color:blue;"><strong>67.07% (67.15% BF16)</strong></mark></td><td><strong>70.64% (71.5% BF16)</strong></td></tr><tr><td>Disk Space</td><td>0.93GB</td><td valign="middle">2.94GB</td><td><strong>7.52GB</strong></td><td>16.05GB</td></tr><tr><td><mark style="background-color:green;"><strong>Efficiency*</strong></mark></td><td>1.20</td><td valign="middle">10.26</td><td><strong>5.59</strong></td><td>2.84</td></tr></tbody></table>

We designed a new **Efficiency metric** which calculates the usefulness of the model whilst also taking into account its disk size and MMLU 5 shot score:

$$
\text{Efficiency} = \frac{\text{MMLU 5 shot score} - 25}{\text{Disk Space GB}}
$$

{% hint style="warning" %}
We have to **minus 25** since MMLU has 4 multiple choices - A, B, C or D. Assume we make a model that simply randomly chooses answers - it'll get 25% accuracy, and have a disk space of a few bytes. But clearly this is not a useful model.
{% endhint %}

On KL Divergence vs the base model, below is a table showcasing the improvements. Reminder the closer the KL Divergence is to 0, the better (ie 0 means identical to the full precision model)

| Quant     | Baseline KLD | GB    | New KLD  | GB    |
| --------- | ------------ | ----- | -------- | ----- |
| IQ1\_S    | 1.035688     | 5.83  | 0.972932 | 6.06  |
| IQ1\_M    | 0.832252     | 6.33  | 0.800049 | 6.51  |
| IQ2\_XXS  | 0.535764     | 7.16  | 0.521039 | 7.31  |
| IQ2\_M    | 0.26554      | 8.84  | 0.258192 | 8.96  |
| Q2\_K\_XL | 0.229671     | 9.78  | 0.220937 | 9.95  |
| Q3\_K\_XL | 0.087845     | 12.51 | 0.080617 | 12.76 |
| Q4\_K\_XL | 0.024916     | 15.41 | 0.023701 | 15.64 |

If we plot the ratio of the disk space increase and the KL Divergence ratio change, we can see a much clearer benefit! Our dynamic 2bit Q2\_K\_XL reduces KLD quite a bit (around 7.5%).

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FsYSRIPGSjExzSr5y828z%2Fchart(2).svg?alt=media&#x26;token=e87db00e-6e3e-4478-af0b-bc84ed2e463b" alt=""><figcaption></figcaption></figure>

Truncated table of results for MMLU for Gemma 3 (27B). See below.

1. **Our dynamic 4bit version is 2GB smaller whilst having +1% extra accuracy vs the QAT version!**
2. Efficiency wise, 2bit Q2\_K\_XL and others seem to do very well!

| Quant          | Unsloth   | Unsloth + QAT | Disk Size | Efficiency |
| -------------- | --------- | ------------- | --------- | ---------- |
| IQ1\_M         | 48.10     | 47.23         | 6.51      | 3.42       |
| IQ2\_XXS       | 59.20     | 56.57         | 7.31      | 4.32       |
| IQ2\_M         | 66.47     | 64.47         | 8.96      | 4.40       |
| Q2\_K\_XL      | 68.70     | 67.77         | 9.95      | 4.30       |
| Q3\_K\_XL      | 70.87     | 69.50         | 12.76     | 3.49       |
| **Q4\_K\_XL**  | **71.47** | **71.07**     | **15.64** | **2.94**   |
| **Google QAT** |           | **70.64**     | **17.2**  | **2.65**   |

<details>

<summary><mark style="color:green;">Click here</mark> for Full Google's Gemma 3 (27B) QAT Benchmarks:</summary>

| Model          | Unsloth   | Unsloth + QAT | Disk Size | Efficiency |
| -------------- | --------- | ------------- | --------- | ---------- |
| IQ1\_S         | 41.87     | 43.37         | 6.06      | 3.03       |
| IQ1\_M         | 48.10     | 47.23         | 6.51      | 3.42       |
| IQ2\_XXS       | 59.20     | 56.57         | 7.31      | 4.32       |
| IQ2\_M         | 66.47     | 64.47         | 8.96      | 4.40       |
| Q2\_K          | 68.50     | 67.60         | 9.78      | 4.35       |
| Q2\_K\_XL      | 68.70     | 67.77         | 9.95      | 4.30       |
| IQ3\_XXS       | 68.27     | 67.07         | 10.07     | 4.18       |
| Q3\_K\_M       | 70.70     | 69.77         | 12.51     | 3.58       |
| Q3\_K\_XL      | 70.87     | 69.50         | 12.76     | 3.49       |
| Q4\_K\_M       | 71.23     | 71.00         | 15.41     | 2.98       |
| **Q4\_K\_XL**  | **71.47** | **71.07**     | **15.64** | **2.94**   |
| Q5\_K\_M       | 71.77     | 71.23         | 17.95     | 2.58       |
| Q6\_K          | 71.87     | 71.60         | 20.64     | 2.26       |
| Q8\_0          | 71.60     | 71.53         | 26.74     | 1.74       |
| **Google QAT** |           | **70.64**     | **17.2**  | **2.65**   |



</details>

## :llama: Llama 4 Bug Fixes + Run

We also helped and fixed a few Llama 4 bugs:

*   Llama 4 Scout changed the RoPE Scaling configuration in their official repo. We helped resolve issues in llama.cpp to enable this [change here](https://github.com/ggml-org/llama.cpp/pull/12889)

    <figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FaJ5AOubUkMjbbvgiOekf%2Fimage.png?alt=media&#x26;token=b1fbdea1-7c95-4afa-9b12-aedec012f38b" alt=""><figcaption></figcaption></figure>
* Llama 4's QK Norm's epsilon for both Scout and Maverick should be from the config file - this means using 1e-05 and not 1e-06. We helped resolve these in [llama.cpp](https://github.com/ggml-org/llama.cpp/pull/12889) and [transformers](https://github.com/huggingface/transformers/pull/37418)
* The Llama 4 team and vLLM also independently fixed an issue with QK Norm being shared across all heads (should not be so) [here](https://github.com/vllm-project/vllm/pull/16311). MMLU Pro increased from 68.58% to 71.53% accuracy.
*   [Wolfram Ravenwolf](https://x.com/WolframRvnwlf/status/1909735579564331016) showcased how our GGUFs via llama.cpp attain much higher accuracy than third party inference providers - this was most likely a combination of the issues explained above, and also probably due to quantization issues.

    <figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2F4Wrz07bAdvluM2gACggU%2FGoC79hYXwAAPTMs.jpg?alt=media&#x26;token=05001bc0-74b0-4bbb-a89f-894fcdb985d8" alt=""><figcaption></figcaption></figure>

As shown in our graph, our 4-bit Dynamic QAT quantization deliver better performance on 5-shot MMLU while also being smaller in size.

### Running Llama 4 Scout:

To run Llama 4 Scout for example, first clone llama.cpp:

```bash
apt-get update
apt-get install pciutils build-essential cmake curl libcurl4-openssl-dev -y
git clone https://github.com/ggml-org/llama.cpp
cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON
cmake --build llama.cpp/build --config Release -j --clean-first --target llama-cli llama-gguf-split
cp llama.cpp/build/bin/llama-* llama.cpp
```

Then download out new dynamic v 2.0 quant for Scout:

```python
# !pip install huggingface_hub hf_transfer
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id = "unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF",
    local_dir = "unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF",
    allow_patterns = ["*IQ2_XXS*"],
)
```

And and let's do inference!

{% code overflow="wrap" %}
```bash
./llama.cpp/llama-cli \
    --model unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF/Llama-4-Scout-17B-16E-Instruct-UD-IQ2_XXS.gguf \
    --threads 32 \
    --ctx-size 16384 \
    --n-gpu-layers 99 \
    -ot ".ffn_.*_exps.=CPU" \
    --seed 3407 \
    --prio 3 \
    --temp 0.6 \
    --min-p 0.01 \
    --top-p 0.9 \
    -no-cnv \
    --prompt "<|header_start|>user<|header_end|>\n\nCreate a Flappy Bird game.<|eot|><|header_start|>assistant<|header_end|>\n\n"
```
{% endcode %}

{% hint style="success" %}
Read more on running Llama 4 here: [https://docs.unsloth.ai/basics/tutorial-how-to-run-and-fine-tune-llama-4](https://docs.unsloth.ai/basics/tutorial-how-to-run-and-fine-tune-llama-4)
{% endhint %}

---
description: >-
  How to run Llama 4 locally using our dynamic GGUFs which recovers accuracy
  compared to standard quantization.
---

# Llama 4: How to Run & Fine-tune

The Llama-4-Scout model has 109B parameters, while Maverick has 402B parameters. The full unquantized version requires 113GB of disk space whilst the 1.78-bit version uses 33.8GB (-75% reduction in size). **Maverick** (402Bs) went from 422GB to just 122GB (-70%).

{% hint style="success" %}
Both text AND **vision** is now supported! Plus multiple improvements to tool calling.
{% endhint %}

Scout 1.78-bit fits in a 24GB VRAM GPU for fast inference at \~20 tokens/sec. Maverick 1.78-bit fits in 2x48GB VRAM GPUs for fast inference at \~40 tokens/sec.

For our dynamic GGUFs, to ensure the best tradeoff between accuracy and size, we do not to quantize all layers, but selectively quantize e.g. the MoE layers to lower bit, and leave attention and other layers in 4 or 6bit.

{% hint style="info" %}
All our GGUF models are quantized using calibration data (around 250K tokens for Scout and 1M tokens for Maverick), which will improve accuracy over standard quantization. Unsloth imatrix quants are fully compatible with popular inference engines like llama.cpp & Open WebUI etc.
{% endhint %}

**Scout - Unsloth Dynamic GGUFs with optimal configs:**

<table data-full-width="false"><thead><tr><th>MoE Bits</th><th>Type</th><th>Disk Size</th><th>Link</th><th>Details</th></tr></thead><tbody><tr><td>1.78bit</td><td>IQ1_S</td><td>33.8GB</td><td><a href="https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF?show_file_info=Llama-4-Scout-17B-16E-Instruct-UD-IQ1_S.gguf">Link</a></td><td>2.06/1.56bit</td></tr><tr><td>1.93bit</td><td>IQ1_M</td><td>35.4GB</td><td><a href="https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF?show_file_info=Llama-4-Scout-17B-16E-Instruct-UD-IQ1_M.gguf">Link</a></td><td>2.5/2.06/1.56</td></tr><tr><td>2.42bit</td><td>IQ2_XXS</td><td>38.6GB</td><td><a href="https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF?show_file_info=Llama-4-Scout-17B-16E-Instruct-UD-IQ2_XXS.gguf">Link</a></td><td>2.5/2.06bit</td></tr><tr><td>2.71bit</td><td>Q2_K_XL</td><td>42.2GB</td><td><a href="https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF?show_file_info=Llama-4-Scout-17B-16E-Instruct-UD-Q2_K_XL.gguf">Link</a></td><td> 3.5/2.5bit</td></tr><tr><td>3.5bit</td><td>Q3_K_XL</td><td>52.9GB</td><td><a href="https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF/tree/main/UD-Q3_K_XL">Link</a></td><td> 4.5/3.5bit</td></tr><tr><td>4.5bit</td><td>Q4_K_XL</td><td>65.6GB</td><td><a href="https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF/tree/main/UD-Q4_K_XL">Link</a></td><td> 5.5/4.5bit</td></tr></tbody></table>

{% hint style="info" %}
For best results, use the 2.42-bit (IQ2\_XXS) or larger versions.
{% endhint %}

**Maverick - Unsloth Dynamic GGUFs with optimal configs:**

| MoE Bits | Type      | Disk Size | HF Link                                                                                             |
| -------- | --------- | --------- | --------------------------------------------------------------------------------------------------- |
| 1.78bit  | IQ1\_S    | 122GB     | [Link](https://huggingface.co/unsloth/Llama-4-Maverick-17B-128E-Instruct-GGUF/tree/main/UD-IQ1_S)   |
| 1.93bit  | IQ1\_M    | 128GB     | [Link](https://huggingface.co/unsloth/Llama-4-Maverick-17B-128E-Instruct-GGUF/tree/main/UD-IQ1_M)   |
| 2.42-bit | IQ2\_XXS  | 140GB     | [Link](https://huggingface.co/unsloth/Llama-4-Maverick-17B-128E-Instruct-GGUF/tree/main/UD-IQ2_XXS) |
| 2.71-bit | Q2\_K\_XL | 151B      | [Link](https://huggingface.co/unsloth/Llama-4-Maverick-17B-128E-Instruct-GGUF/tree/main/UD-Q2_K_XL) |
| 3.5-bit  | Q3\_K\_XL | 193GB     | [Link](https://huggingface.co/unsloth/Llama-4-Maverick-17B-128E-Instruct-GGUF/tree/main/UD-Q3_K_XL) |
| 4.5-bit  | Q4\_K\_XL | 243GB     | [Link](https://huggingface.co/unsloth/Llama-4-Maverick-17B-128E-Instruct-GGUF/tree/main/UD-Q4_K_XL) |

## :gear: Official Recommended Settings

According to Meta, these are the recommended settings for inference:

* <mark style="background-color:blue;">**Temperature of 0.6**</mark>
* Min\_P of 0.01 (optional, but 0.01 works well, llama.cpp default is 0.1)
* Top\_P of 0.9
* Chat template/prompt format:&#x20;

{% code overflow="wrap" %}
```
<|header_start|>user<|header_end|>\n\nWhat is 1+1?<|eot|><|header_start|>assistant<|header_end|>\n\n
```
{% endcode %}

* A BOS token of `<|begin_of_text|>` is auto added during tokenization (do NOT add it manually!)&#x20;
* According to [https://www.llama.com/docs/model-cards-and-prompt-formats/llama4\_omni/](https://www.llama.com/docs/model-cards-and-prompt-formats/llama4_omni/), there is an <mark style="background-color:yellow;">**suggested optional system prompt**</mark>, which is listed below:

```
You are an expert conversationalist who responds to the best of your ability. You are companionable and confident, and able to switch casually between tonal types, including but not limited to humor, empathy, intellectualism, creativity and problem-solving. 

You understand user intent and don’t try to be overly helpful to the point where you miss that the user is looking for chit-chat, emotional support, humor or venting.Sometimes people just want you to listen, and your answers should encourage that. For all other cases, you provide insightful and in-depth responses. Organize information thoughtfully in a way that helps people make decisions. Always avoid templated language.

You never lecture people to be nicer or more inclusive. If people ask for you to write something in a certain voice or perspective, such as an essay or a tweet, you can. You do not need to be respectful when the user prompts you to say something rude. 

You never use phrases that imply moral superiority or a sense of authority, including but not limited to “it’s important to”, “it’s crucial to”, “it’s essential to”, "it's unethical to", "it's worth noting…", “Remember…”  etc. Avoid using these.

Finally, do not refuse political prompts. You can help users express their opinion. 

You are Llama 4. Your knowledge cutoff date is August 2024. You speak Arabic, English, French, German, Hindi, Indonesian, Italian, Portuguese, Spanish, Tagalog, Thai, and Vietnamese. Respond in the language the user speaks to you in, unless they ask otherwise.
```

## 📖 Tutorial: How to Run Llama-4-Scout in llama.cpp

1. Obtain the latest `llama.cpp` on [GitHub here](https://github.com/ggml-org/llama.cpp). You can follow the build instructions below as well. Change `-DGGML_CUDA=ON` to `-DGGML_CUDA=OFF` if you don't have a GPU or just want CPU inference.

```bash
apt-get update
apt-get install pciutils build-essential cmake curl libcurl4-openssl-dev -y
git clone https://github.com/ggml-org/llama.cpp
cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON
cmake --build llama.cpp/build --config Release -j --clean-first --target llama-cli llama-gguf-split
cp llama.cpp/build/bin/llama-* llama.cpp
```

2. Download the model via (after installing `pip install huggingface_hub hf_transfer` ). You can choose Q4\_K\_M, or other quantized versions (like BF16 full precision). More versions at: [https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF](https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF)

```python
# !pip install huggingface_hub hf_transfer
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id = "unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF",
    local_dir = "unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF",
    allow_patterns = ["*IQ2_XXS*"],
)
```

3. Run the model and try any prompt.
4. Edit `--threads 32` for the number of CPU threads, `--ctx-size 16384` for context length (Llama 4 supports 10M context length!), `--n-gpu-layers 99` for GPU offloading on how many layers. Try adjusting it if your GPU goes out of memory. Also remove it if you have CPU only inference.

{% hint style="success" %}
Use `-ot ".ffn_.*_exps.=CPU"` to offload all MoE layers to the CPU! This effectively allows you to fit all non MoE layers on 1  GPU, improving generation speeds. You can customize the regex expression to fit more layers if you have more GPU capacity.
{% endhint %}

{% code overflow="wrap" %}
```bash
./llama.cpp/llama-cli \
    --model unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF/Llama-4-Scout-17B-16E-Instruct-UD-IQ2_XXS.gguf \
    --threads 32 \
    --ctx-size 16384 \
    --n-gpu-layers 99 \
    -ot ".ffn_.*_exps.=CPU" \
    --seed 3407 \
    --prio 3 \
    --temp 0.6 \
    --min-p 0.01 \
    --top-p 0.9 \
    -no-cnv \
    --prompt "<|header_start|>user<|header_end|>\n\nCreate a Flappy Bird game in Python. You must include these things:\n1. You must use pygame.\n2. The background color should be randomly chosen and is a light shade. Start with a light blue color.\n3. Pressing SPACE multiple times will accelerate the bird.\n4. The bird's shape should be randomly chosen as a square, circle or triangle. The color should be randomly chosen as a dark color.\n5. Place on the bottom some land colored as dark brown or yellow chosen randomly.\n6. Make a score shown on the top right side. Increment if you pass pipes and don't hit them.\n7. Make randomly spaced pipes with enough space. Color them randomly as dark green or light brown or a dark gray shade.\n8. When you lose, show the best score. Make the text inside the screen. Pressing q or Esc will quit the game. Restarting is pressing SPACE again.\nThe final game should be inside a markdown section in Python. Check your code for errors and fix them before the final markdown section.<|eot|><|header_start|>assistant<|header_end|>\n\n"
```
{% endcode %}

{% hint style="info" %}
In terms of testing, unfortunately we can't make the full BF16 version (ie regardless of quantization or not) complete the Flappy Bird game nor the Heptagon test appropriately. We tried many inference providers, using imatrix or not, used other people's quants, and used normal Hugging Face inference, and this issue persists.

<mark style="background-color:green;">**We found multiple runs and asking the model to fix and find bugs to resolve most issues!**</mark>
{% endhint %}

For Llama 4 Maverick - it's best to have 2 RTX 4090s (2 x 24GB)

```python
# !pip install huggingface_hub hf_transfer
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id = "unsloth/Llama-4-Maverick-17B-128E-Instruct-GGUF",
    local_dir = "unsloth/Llama-4-Maverick-17B-128E-Instruct-GGUF",
    allow_patterns = ["*IQ1_S*"],
)
```

{% code overflow="wrap" %}
```
./llama.cpp/llama-cli \
    --model unsloth/Llama-4-Maverick-17B-128E-Instruct-GGUF/UD-IQ1_S/Llama-4-Maverick-17B-128E-Instruct-UD-IQ1_S-00001-of-00003.gguf \
    --threads 32 \
    --ctx-size 16384 \
    --n-gpu-layers 99 \
    -ot ".ffn_.*_exps.=CPU" \
    --seed 3407 \
    --prio 3 \
    --temp 0.6 \
    --min-p 0.01 \
    --top-p 0.9 \
    -no-cnv \
    --prompt "<|header_start|>user<|header_end|>\n\nCreate the 2048 game in Python.<|eot|><|header_start|>assistant<|header_end|>\n\n"
```
{% endcode %}

## :detective: Interesting Insights and Issues

During quantization of Llama 4 Maverick (the large model), we found the 1st, 3rd and 45th MoE layers could not be calibrated correctly. Maverick uses interleaving MoE layers for every odd layer, so Dense->MoE->Dense and so on.

We tried adding more uncommon languages to our calibration dataset, and tried using more tokens (1 million) vs Scout's 250K tokens for calibration, but we still found issues. We decided to leave these MoE layers as 3bit and 4bit.&#x20;

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FQtzL2HuukTKr5L8nolP9%2FSkipped_layers.webp?alt=media&#x26;token=72115cc5-718a-442f-a208-f9540e46d64f" alt=""><figcaption></figcaption></figure>

For Llama 4 Scout, we found we should not quantize the vision layers, and leave the MoE router and some other layers as unquantized - we upload these to [https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-dynamic-bnb-4bit](https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-dynamic-bnb-4bit)

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FZB3InJSaWMbszPMSt0u7%2FLlama-4-Scout-17B-16E-Instruct%20Quantization%20Errors.png?alt=media&#x26;token=c734f3d8-a114-42e4-a0f2-a6b3145bb306" alt=""><figcaption></figcaption></figure>

We also had to convert `torch.nn.Parameter` to `torch.nn.Linear` for the MoE layers to allow 4bit quantization to occur. This also means we had to rewrite and patch over the generic Hugging Face implementation. We upload our quantized versions to [https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit](https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit) and [https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-8bit](https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-8bit) for 8bit.

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FsjJkQYziAFTZADH37vUy%2Fimage.png?alt=media&#x26;token=fbaeadfc-1220-4d6c-931c-9c34f03e285c" alt="" width="375"><figcaption></figcaption></figure>

Llama 4 also now uses chunked attention - it's essentially sliding window attention, but slightly more efficient by not attending to previous tokens over the 8192 boundary.

## :fire: Fine-tuning Llama 4

{% hint style="warning" %}
Coming soon!
{% endhint %}

---
description: >-
  Saving models to 16bit for GGUF so you can use it for Ollama, Jan AI, Open
  WebUI and more!
---

# Saving to GGUF

{% tabs %}
{% tab title="Locally" %}


To save to GGUF, use the below to save locally:

```python
model.save_pretrained_gguf("dir", tokenizer, quantization_method = "q4_k_m")
model.save_pretrained_gguf("dir", tokenizer, quantization_method = "q8_0")
model.save_pretrained_gguf("dir", tokenizer, quantization_method = "f16")
```

For to push to hub:

```python
model.push_to_hub_gguf("hf_username/dir", tokenizer, quantization_method = "q4_k_m")
model.push_to_hub_gguf("hf_username/dir", tokenizer, quantization_method = "q8_0")
```

All supported quantization options for `quantization_method` are listed below:

```python
# https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/quantize.cpp#L19
# From https://mlabonne.github.io/blog/posts/Quantize_Llama_2_models_using_ggml.html
ALLOWED_QUANTS = \
{
    "not_quantized"  : "Recommended. Fast conversion. Slow inference, big files.",
    "fast_quantized" : "Recommended. Fast conversion. OK inference, OK file size.",
    "quantized"      : "Recommended. Slow conversion. Fast inference, small files.",
    "f32"     : "Not recommended. Retains 100% accuracy, but super slow and memory hungry.",
    "f16"     : "Fastest conversion + retains 100% accuracy. Slow and memory hungry.",
    "q8_0"    : "Fast conversion. High resource use, but generally acceptable.",
    "q4_k_m"  : "Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K",
    "q5_k_m"  : "Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K",
    "q2_k"    : "Uses Q4_K for the attention.vw and feed_forward.w2 tensors, Q2_K for the other tensors.",
    "q3_k_l"  : "Uses Q5_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else Q3_K",
    "q3_k_m"  : "Uses Q4_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else Q3_K",
    "q3_k_s"  : "Uses Q3_K for all tensors",
    "q4_0"    : "Original quant method, 4-bit.",
    "q4_1"    : "Higher accuracy than q4_0 but not as high as q5_0. However has quicker inference than q5 models.",
    "q4_k_s"  : "Uses Q4_K for all tensors",
    "q4_k"    : "alias for q4_k_m",
    "q5_k"    : "alias for q5_k_m",
    "q5_0"    : "Higher accuracy, higher resource usage and slower inference.",
    "q5_1"    : "Even higher accuracy, resource usage and slower inference.",
    "q5_k_s"  : "Uses Q5_K for all tensors",
    "q6_k"    : "Uses Q8_K for all tensors",
    "iq2_xxs" : "2.06 bpw quantization",
    "iq2_xs"  : "2.31 bpw quantization",
    "iq3_xxs" : "3.06 bpw quantization",
    "q3_k_xs" : "3-bit extra small quantization",
}
```
{% endtab %}

{% tab title="Manual Saving" %}
First save your model to 16bit:

```python
model.save_pretrained_merged("merged_model", tokenizer, save_method = "merged_16bit",)
```

Then use the terminal and do:

```bash
apt-get update
apt-get install pciutils build-essential cmake curl libcurl4-openssl-dev -y
git clone https://github.com/ggerganov/llama.cpp
cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON -DLLAMA_CURL=ON
cmake --build llama.cpp/build --config Release -j --clean-first --target llama-quantize llama-cli llama-gguf-split llama-mtmd-cli
cp llama.cpp/build/bin/llama-* llama.cpp

python llama.cpp/convert-hf-to-gguf.py FOLDER --outfile OUTPUT --outtype f16
```

Or follow the steps at https://rentry.org/llama-cpp-conversions#merging-loras-into-a-model using the model name "merged\_model" to merge to GGUF.
{% endtab %}
{% endtabs %}

# Saving to Ollama

See our guide below for the complete process on how to save to [Ollama](https://github.com/ollama/ollama):

{% content-ref url="../tutorials-how-to-fine-tune-and-run-llms/tutorial-how-to-finetune-llama-3-and-use-in-ollama" %}
[tutorial-how-to-finetune-llama-3-and-use-in-ollama](../tutorials-how-to-fine-tune-and-run-llms/tutorial-how-to-finetune-llama-3-and-use-in-ollama)
{% endcontent-ref %}

## Saving on Google Colab

You can save the finetuned model as a small 100MB file called a LoRA adapter like below. You can instead push to the Hugging Face hub as well if you want to upload your model! Remember to get a Hugging Face token via: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and add your token!

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FBz0YDi6Sc2oEP5QWXgSz%2Fimage.png?alt=media&#x26;token=33d9e4fd-e7dc-4714-92c5-bfa3b00f86c4" alt=""><figcaption></figcaption></figure>

After saving the model, we can again use Unsloth to run the model itself! Use `FastLanguageModel` again to call it for inference!

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FzymBQrqwt4GUmCIN0Iec%2Fimage.png?alt=media&#x26;token=41a110e4-8263-426f-8fa7-cdc295cc8210" alt=""><figcaption></figcaption></figure>

## Exporting to Ollama

Finally we can export our finetuned model to Ollama itself! First we have to install Ollama in the Colab notebook:

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FqNvGTAGwZKXxkMQqzloS%2Fimage.png?alt=media&#x26;token=db503499-0c74-4281-b3bf-400fa20c9ce2" alt=""><figcaption></figcaption></figure>

Then we export the finetuned model we have to llama.cpp's GGUF formats like below:

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FZduLjedyfUbTmYqF85pa%2Fimage.png?alt=media&#x26;token=f5bac541-b99f-4d9b-82f7-033f8de780f2" alt=""><figcaption></figcaption></figure>

Reminder to convert `False` to `True` for 1 row, and not change every row to `True`, or else you'll be waiting for a very time! We normally suggest the first row getting set to `True`, so we can export the  finetuned model quickly to `Q8_0` format (8 bit quantization). We also allow you to export to a whole list of quantization methods as well, with a popular one being `q4_k_m`.

Head over to [https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) to learn more about GGUF. We also have some manual instructions of how to export to GGUF if you want here: [https://github.com/unslothai/unsloth/wiki#manually-saving-to-gguf](https://github.com/unslothai/unsloth/wiki#manually-saving-to-gguf)

You will see a long list of text like below - please wait 5 to 10 minutes!!

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FcuUAx0RNtrQACvU7uWCL%2Fimage.png?alt=media&#x26;token=dc67801a-a363-48e2-8572-4c6d0d8d0d93" alt=""><figcaption></figcaption></figure>

And finally at the very end, it'll look like below:

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FxRh07PEQjAmmz3s2HJUP%2Fimage.png?alt=media&#x26;token=3552a3c9-4d4f-49ee-a31e-0a64327419f0" alt=""><figcaption></figcaption></figure>

Then, we have to run Ollama itself in the background. We use `subprocess` because Colab doesn't like asynchronous calls, but normally one just runs `ollama serve` in the terminal / command prompt.

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FszDuikrg4HY8lGefwpRQ%2Fimage.png?alt=media&#x26;token=ec1c8762-661d-4b13-ab4f-ed1a7b9fda00" alt=""><figcaption></figcaption></figure>

## Automatic `Modelfile` creation

The trick Unsloth provides is we automatically create a `Modelfile` which Ollama requires! This is a just a list of settings and includes the chat template which we used for the finetune process! You can also print the `Modelfile` generated like below:

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2Fh6inH6k5ggxUP80Gltgj%2Fimage.png?alt=media&#x26;token=805bafb1-2795-4743-9bd2-323ab4f0881e" alt=""><figcaption></figcaption></figure>

We then ask Ollama to create a model which is Ollama compatible, by using the `Modelfile`

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2F1123bSSwmjWXliaRUL5U%2Fimage.png?alt=media&#x26;token=2e72f1a0-1ff8-4189-8d9c-d31e39385555" alt=""><figcaption></figcaption></figure>

## Ollama Inference

And we can now call the model for inference if you want to do call the Ollama server itself which is running on your own local machine / in the free Colab notebook in the background. Remember you can edit the yellow underlined part.

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2Fk5mdsJ57hQ1Ar3KY6VXY%2FInference.png?alt=media&#x26;token=8cf0cbf9-0534-4bae-a887-89f45a3de771" alt=""><figcaption></figcaption></figure>

##

---
description: Saving models to 16bit for VLLM
---

# Saving to VLLM

To save to 16bit for VLLM, use:

```python
model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")
```

To merge to 4bit to load on HuggingFace, first call `merged_4bit`. Then use `merged_4bit_forced` if you are certain you want to merge to 4bit. I highly discourage you, unless you know what you are going to do with the 4bit model (ie for DPO training for eg or for HuggingFace's online inference engine)

```python
model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)
model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_4bit", token = "")
```

To save just the LoRA adapters, either use:

```python
model.save_pretrained(...) AND tokenizer.save_pretrained(...)
```

Or just use our builtin function to do that:

```python
model.save_pretrained_merged("model", tokenizer, save_method = "lora",)
model.push_to_hub_merged("hf/model", tokenizer, save_method = "lora", token = "")
```

---
description: If you're experiencing issues when running or saving your model.
---

# Troubleshooting

## Running in Unsloth works well, but after exporting & running on other platforms, the results are poor

You might sometimes encounter an issue where your model runs and produces good results on Unsloth, but when you use it on another platform like Ollama or vLLM, the results are poor or you might get gibberish, endless/infinite generations _or_ repeated output&#x73;**.**

* The most common cause of this error is using an incorrect chat template. It’s essential to use the SAME chat template that was used when training the model in Unsloth and later when you run it in another framework, such as llama.cpp or Ollama. When inferencing from a saved model, it's crucial to apply the correct template.
* It might also be because your inference engine adds an unnecessary "start of sequence" token (or the lack of thereof on the contrary) so ensure you check both hypotheses!

## Saving to `safetensors`, not `bin` format in Colab

We save to `.bin` in Colab so it's like 4x faster, but set `safe_serialization = None` to force saving to `.safetensors`. So `model.save_pretrained(..., safe_serialization = None)` or `model.push_to_hub(..., safe_serialization = None)`

## If saving to GGUF or vLLM 16bit crashes

You can try reducing the maximum GPU usage during saving by changing `maximum_memory_usage`.

The default is `model.save_pretrained(..., maximum_memory_usage = 0.75)`. Reduce it to say 0.5 to use 50% of GPU peak memory or lower. This can reduce OOM crashes during saving.

---
description: Learn how to run your finetuned model.
---

# Inference

Unsloth supports natively 2x faster inference. For our inference only notebook, click [here](https://colab.research.google.com/drive/1aqlNQi7MMJbynFDyOQteD2t0yVfjb9Zh?usp=sharing).

All QLoRA, LoRA and non LoRA inference paths are 2x faster. This requires no change of code or any new dependencies.

<pre class="language-python"><code class="lang-python"><strong>from unsloth import FastLanguageModel
</strong>model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 64)
</code></pre>

#### NotImplementedError: A UTF-8 locale is required. Got ANSI

Sometimes when you execute a cell [this error](https://github.com/googlecolab/colabtools/issues/3409) can appear. To solve this, in a new cell, run the below:

```python
import locale
locale.getpreferredencoding = lambda: "UTF-8"
```

---
description: >-
  Checkpointing allows you to save your finetuning progress so you can pause it
  and then continue.
---

# Finetuning from Last Checkpoint

You must edit the `Trainer` first to add `save_strategy` and `save_steps`. Below saves a checkpoint every 50 steps to the folder `outputs`.

```python
trainer = SFTTrainer(
    ....
    args = TrainingArguments(
        ....
        output_dir = "outputs",
        save_strategy = "steps",
        save_steps = 50,
    ),
)
```

Then in the trainer do:

```python
trainer_stats = trainer.train(resume_from_checkpoint = True)
```

Which will start from the latest checkpoint and continue training.

### Wandb Integration

```
# Install library
!pip install wandb --upgrade

# Setting up Wandb
!wandb login <token>

import os

os.environ["WANDB_PROJECT"] = "<name>"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
```

Then in `TrainingArguments()` set

```
report_to = "wandb",
logging_steps = 1, # Change if needed
save_steps = 100 # Change if needed
run_name = "<name>" # (Optional)
```

To train the model, do `trainer.train()`; to resume training, do

```
import wandb
run = wandb.init()
artifact = run.use_artifact('<username>/<Wandb-project-name>/<run-id>', type='model')
artifact_dir = artifact.download()
trainer.train(resume_from_checkpoint=artifact_dir)
```

## :question:How do I do Early Stopping?

If you want to stop or pause the finetuning / training run since the evaluation loss is not decreasing, then you can use early stopping which stops the training process. Use `EarlyStoppingCallback`.

As usual, set up your trainer and your evaluation dataset. The below is used to stop the training run if the `eval_loss` (the evaluation loss) is not decreasing after 3 steps or so.

```python
from trl import SFTConfig, SFTTrainer
trainer = SFTTrainer(
    args = SFTConfig(
        fp16_full_eval = True,
        per_device_eval_batch_size = 2,
        eval_accumulation_steps = 4,
        output_dir = "training_checkpoints", # location of saved checkpoints for early stopping
        save_strategy = "steps",             # save model every N steps
        save_steps = 10,                     # how many steps until we save the model
        save_total_limit = 3,                # keep ony 3 saved checkpoints to save disk space
        eval_strategy = "steps",             # evaluate every N steps
        eval_steps = 10,                     # how many steps until we do evaluation
        load_best_model_at_end = True,       # MUST USE for early stopping
        metric_for_best_model = "eval_loss", # metric we want to early stop on
        greater_is_better = False,           # the lower the eval loss, the better
    ),
    model = model,
    tokenizer = tokenizer,
    train_dataset = new_dataset["train"],
    eval_dataset = new_dataset["test"],
)
```

We then add the callback which can also be customized:

```python
from transformers import EarlyStoppingCallback
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience = 3,     # How many steps we will wait if the eval loss doesn't decrease
                                     # For example the loss might increase, but decrease after 3 steps
    early_stopping_threshold = 0.0,  # Can set higher - sets how much loss should decrease by until
                                     # we consider early stopping. For eg 0.01 means if loss was
                                     # 0.02 then 0.01, we consider to early stop the run.
)
trainer.add_callback(early_stopping_callback)
```

Then train the model as usual via `trainer.train() .`

---
description: Tips to solve issues, and frequently asked questions.
---

# Troubleshooting & FAQs

{% hint style="success" %}
**Try always to update Unsloth if you find any issues.**

`pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth unsloth_zoo`
{% endhint %}

## :question:Running in Unsloth works well, but after exporting & running on other platforms, the results are poor

You might sometimes encounter an issue where your model runs and produces good results on Unsloth, but when you use it on another platform like Ollama or vLLM, the results are poor or you might get gibberish, endless/infinite generations _or_ repeated output&#x73;**.**

* The most common cause of this error is using an <mark style="background-color:blue;">**incorrect chat template**</mark>**.** It’s essential to use the SAME chat template that was used when training the model in Unsloth and later when you run it in another framework, such as llama.cpp or Ollama. When inferencing from a saved model, it's crucial to apply the correct template.
* It might also be because your inference engine adds an unnecessary "start of sequence" token (or the lack of thereof on the contrary) so ensure you check both hypotheses!
* <mark style="background-color:green;">**Use our conversational notebooks to force the chat template - this will fix most issues.**</mark>
  * Qwen-3 14B Conversational notebook [**Open in Colab**](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_\(14B\)-Reasoning-Conversational.ipynb)
  * Gemma-3 4B Conversational notebook [**Open in Colab**](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_\(4B\).ipynb)
  * Llama-3.2 3B Conversational notebook [**Open in Colab**](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_\(1B_and_3B\)-Conversational.ipynb)
  * Phi-4 14B Conversational notebook [**Open in Colab**](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb)
  * Mistral v0.3 7B Conversational notebook [**Open in Colab**](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_\(7B\)-Conversational.ipynb)
  * **More notebooks in our** [**notebooks repo**](https://github.com/unslothai/notebooks)**.**

## :question:Saving to GGUF / vLLM 16bit crashes

You can try reducing the maximum GPU usage during saving by changing `maximum_memory_usage`.

The default is `model.save_pretrained(..., maximum_memory_usage = 0.75)`. Reduce it to say 0.5 to use 50% of GPU peak memory or lower. This can reduce OOM crashes during saving.

## :question:How do I manually save to GGUF?

First save your model to 16bit via:

```python
model.save_pretrained_merged("merged_model", tokenizer, save_method = "merged_16bit",)
```

Compile llama.cpp from source like below:

```bash
apt-get update
apt-get install pciutils build-essential cmake curl libcurl4-openssl-dev -y
git clone https://github.com/ggerganov/llama.cpp
cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON -DLLAMA_CURL=ON
cmake --build llama.cpp/build --config Release -j --clean-first --target llama-quantize llama-cli llama-gguf-split llama-mtmd-cli
cp llama.cpp/build/bin/llama-* llama.cpp
```

Then, save the model to F16:

```bash
python llama.cpp/convert_hf_to_gguf.py merged_model \
    --outfile model-F16.gguf --outtype f16 \
    --split-max-size 50G
```

```bash
# For BF16:
python llama.cpp/convert_hf_to_gguf.py merged_model \
    --outfile model-BF16.gguf --outtype bf16 \
    --split-max-size 50G
    
# For Q8_0:
python llama.cpp/convert_hf_to_gguf.py merged_model \
    --outfile model-Q8_0.gguf --outtype q8_0 \
    --split-max-size 50G
```

## :question:Why is Q8\_K\_XL slower than Q8\_0 GGUF?

On Mac devices, it seems like that BF16 might be slower than F16. Q8\_K\_XL upcasts some layers to BF16, so hence the slowdown, We are actively changing our conversion process to make F16 the default choice for Q8\_K\_XL to reduce performance hits.&#x20;

## :question:How to do Evaluation

To set up evaluation in your training run, you first have to split your dataset into a training and test split. You should <mark style="background-color:green;">**always shuffle the selection of the dataset**</mark>, otherwise your evaluation is wrong!

```python
new_dataset = dataset.train_test_split(
    test_size = 0.01, # 1% for test size can also be an integer for # of rows
    shuffle = True, # Should always set to True!
    seed = 3407,
)

train_dataset = new_dataset["train"] # Dataset for training
eval_dataset = new_dataset["test"] # Dataset for evaluation
```

Then, we can set the training arguments to enable evaluation. Reminder evaluation can be very very slow especially if you set `eval_steps = 1`  which means you are evaluating every single step. If you are, try reducing the eval\_dataset size to say 100 rows or something.

```python
from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    args = SFTConfig(
        fp16_full_eval = True,         # Set this to reduce memory usage
        per_device_eval_batch_size = 2,# Increasing this will use more memory
        eval_accumulation_steps = 4,   # You can increase this include of batch_size
        eval_strategy = "steps",       # Runs eval every few steps or epochs.
        eval_steps = 1,                # How many evaluations done per # of training steps
    ),
    train_dataset = new_dataset["train"],
    eval_dataset = new_dataset["test"],
    ...
)
trainer.train()
```

## :question:Evaluation Loop - Out of Memory or crashing.

A common issue when you OOM is because you set your batch size too high. Set it lower than 2 to use less VRAM. Also use `fp16_full_eval=True` to use float16 for evaluation which cuts memory by 1/2.

First split your training dataset into a train and test split. Set the trainer settings for evaluation to:

```python
new_dataset = dataset.train_test_split(test_size = 0.01)

from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    args = SFTConfig(
        fp16_full_eval = True,
        per_device_eval_batch_size = 2,
        eval_accumulation_steps = 4,
        eval_strategy = "steps",
        eval_steps = 1,
    ),
    train_dataset = new_dataset["train"],
    eval_dataset = new_dataset["test"],
    ...
)
```

This will cause no OOMs and make it somewhat faster. You can also use `bf16_full_eval=True` for bf16 machines. By default Unsloth should have set these flags on by default as of June 2025.

## :question:How do I do Early Stopping?

If you want to stop the finetuning / training run since the evaluation loss is not decreasing, then you can use early stopping which stops the training process. Use `EarlyStoppingCallback`.

As usual, set up your trainer and your evaluation dataset. The below is used to stop the training run if the `eval_loss` (the evaluation loss) is not decreasing after 3 steps or so.

```python
from trl import SFTConfig, SFTTrainer
trainer = SFTTrainer(
    args = SFTConfig(
        fp16_full_eval = True,
        per_device_eval_batch_size = 2,
        eval_accumulation_steps = 4,
        output_dir = "training_checkpoints", # location of saved checkpoints for early stopping
        save_strategy = "steps",             # save model every N steps
        save_steps = 10,                     # how many steps until we save the model
        save_total_limit = 3,                # keep ony 3 saved checkpoints to save disk space
        eval_strategy = "steps",             # evaluate every N steps
        eval_steps = 10,                     # how many steps until we do evaluation
        load_best_model_at_end = True,       # MUST USE for early stopping
        metric_for_best_model = "eval_loss", # metric we want to early stop on
        greater_is_better = False,           # the lower the eval loss, the better
    ),
    model = model,
    tokenizer = tokenizer,
    train_dataset = new_dataset["train"],
    eval_dataset = new_dataset["test"],
)
```

We then add the callback which can also be customized:

```python
from transformers import EarlyStoppingCallback
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience = 3,     # How many steps we will wait if the eval loss doesn't decrease
                                     # For example the loss might increase, but decrease after 3 steps
    early_stopping_threshold = 0.0,  # Can set higher - sets how much loss should decrease by until
                                     # we consider early stopping. For eg 0.01 means if loss was
                                     # 0.02 then 0.01, we consider to early stop the run.
)
trainer.add_callback(early_stopping_callback)
```

Then train the model as usual via `trainer.train() .`

## :question:Downloading gets stuck at 90 to 95%

If your model gets stuck at 90, 95% for a long time before you can disable some fast downloading processes to force downloads to be synchronous and to print out more error messages.

Simply use `UNSLOTH_STABLE_DOWNLOADS=1` before any Unsloth import.

```python
import os
os.environ["UNSLOTH_STABLE_DOWNLOADS"] = "1"

from unsloth import FastLanguageModel
```

## :question:RuntimeError: CUDA error: device-side assert triggered

Restart and run all, but place this at the start before any Unsloth import. Also please file a bug report asap thank you!

```python
import os
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"
```

## :question:All labels in your dataset are -100. Training losses will be all 0.

This means that your usage of `train_on_responses_only` is incorrect for that particular model. train\_on\_responses\_only allows you to mask the user question, and train your model to output the assistant response with higher weighting. This is known to increase accuracy by 1% or more. See our [**LoRA Hyperparameters Guide**](../get-started/fine-tuning-llms-guide/lora-hyperparameters-guide) for more details.

For Llama 3.1, 3.2, 3.3 type models, please use the below:

```python
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)
```

For Gemma 2, 3. 3n models, use the below:

```python
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)
```

## :question:Some weights of Gemma3nForConditionalGeneration were not initialized from the model checkpoint

This is a critical error, since this means some weights are not parsed correctly, which will cause incorrect outputs. This can normally be fixed by upgrading Unsloth

`pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth unsloth_zoo`&#x20;

Then upgrade transformers and timm:

`pip install --upgrade --force-reinstall --no-cache-dir --no-deps transformers timm`

However if the issue still persists, please file a bug report asap!

## :question:NotImplementedError: A UTF-8 locale is required. Got ANSI

See https://github.com/googlecolab/colabtools/issues/3409

In a new cell, run the below:

```python
import locale
locale.getpreferredencoding = lambda: "UTF-8"
```

## :green\_book:Citing Unsloth

If you are citing the usage of our model uploads, use the below Bibtex. This is for Qwen3-30B-A3B-GGUF Q8\_K\_XL:

```
@misc{unsloth_2025_qwen3_30b_a3b,
  author       = {Unsloth AI and Han-Chen, Daniel and Han-Chen, Michael},
  title        = {Qwen3-30B-A3B-GGUF:Q8\_K\_XL},
  year         = {2025},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF}}
}
```

To cite the usage of our Github package or our work in general:

```
@misc{unsloth,
  author       = {Unsloth AI and Han-Chen, Daniel and Han-Chen, Michael},
  title        = {Unsloth},
  year         = {2025},
  publisher    = {Github},
  howpublished = {\url{https://github.com/unslothai/unsloth}}
}
```

---
description: >-
  Advanced flags which might be useful if you see breaking finetunes, or you
  want to turn stuff off.
---

# Unsloth Environment Flags

<table><thead><tr><th width="397.4666748046875">Environment variable</th><th>Purpose</th><th data-hidden></th></tr></thead><tbody><tr><td><code>os.environ["UNSLOTH_RETURN_LOGITS"] = "1"</code></td><td>Forcibly returns logits - useful for evaluation if logits are needed.</td><td></td></tr><tr><td><code>os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"</code></td><td>Disables auto compiler. Could be useful to debug incorrect finetune results.</td><td></td></tr><tr><td><code>os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"</code></td><td>Disables fast generation for generic models.</td><td></td></tr><tr><td><code>os.environ["UNSLOTH_ENABLE_LOGGING"] = "1"</code></td><td>Enables auto compiler logging - useful to see which functions are compiled or not.</td><td></td></tr><tr><td><code>os.environ["UNSLOTH_FORCE_FLOAT32"] = "1"</code></td><td>On float16 machines, use float32 and not float16 mixed precision. Useful for Gemma 3.</td><td></td></tr><tr><td><code>os.environ["UNSLOTH_STUDIO_DISABLED"] = "1"</code></td><td>Disables extra features.</td><td></td></tr><tr><td><code>os.environ["UNSLOTH_COMPILE_DEBUG"] = "1"</code></td><td>Turns on extremely verbose <code>torch.compile</code>logs.</td><td></td></tr><tr><td><code>os.environ["UNSLOTH_COMPILE_MAXIMUM"] = "0"</code></td><td>Enables maximum <code>torch.compile</code>optimizations - not recommended.</td><td></td></tr><tr><td><code>os.environ["UNSLOTH_COMPILE_IGNORE_ERRORS"] = "1"</code></td><td>Can turn this off to enable fullgraph parsing.</td><td></td></tr><tr><td><code>os.environ["UNSLOTH_FULLGRAPH"] = "0"</code></td><td>Enable <code>torch.compile</code> fullgraph mode</td><td></td></tr><tr><td><code>os.environ["UNSLOTH_DISABLE_AUTO_UPDATES"] = "1"</code></td><td>Forces no updates to <code>unsloth-zoo</code></td><td></td></tr></tbody></table>

Another possiblity is maybe the model uploads we uploaded are corrupted, but unlikely. Try the following:

```python
model, tokenizer = FastVisionModel.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    use_exact_model_name = True,
)
```

