This README is designed to explain the "why" and "how" of the project, focusing on the technical innovation of using AirLLM for local inference.

InBox Lens: Privacy-First Local Email Intelligence
InBox Lens is a modular AI-powered assistant that processes your sensitive email data entirely offline. By leveraging AirLLM and the TinyLlama-1.1B-Chat model, this tool provides high-level executive assistant capabilities without the data ever leaving your local machine.

🚀 The Core Technology: AirLLM
The primary challenge of running Large Language Models (LLMs) locally is the massive VRAM requirement. Most consumer-grade GPUs cannot fit a model's entire weight set into memory.

How AirLLM Works
AirLLM utilizes a technique known as Layered Inference. Instead of loading the full model into memory, it:

Shards the model: Breaks the model into individual layers stored on your disk.

Sequential Execution: Loads a single layer into RAM/VRAM, performs the necessary calculations, and then replaces it with the next layer.

Memory Efficiency: This allows you to run models (like Llama 70B) that are 10x-50x larger than your actual VRAM capacity, albeit at a slower inference speed than full-memory loading.

AirLLM vs. Quantization
Quantization: Reduces the "weight" of the model (e.g., from 16-bit to 4-bit) to make it smaller. Use this when your model is almost small enough to fit in your GPU.

AirLLM: Optimizes the "loading process." Use this when even a quantized model is far too large for your hardware. It acts as a bridge for running enterprise-scale models on consumer hardware.

🛠️ Project Functionalities
This project transforms a raw LLM into a specialized worker for email management:

Action Item Extraction: Parses long, messy email threads to identify specific deadlines, tasks, and follow-ups, presenting them in a structured checklist.

Context-Aware Tone Shifting: Refines rough drafts into polished professional communications. It maintains the original intent while adjusting the vocabulary to match target tones like "Urgent," "Friendly," or "Persuasive."



