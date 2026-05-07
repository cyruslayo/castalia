"""Notebook 15 — Document Corpus (condensed for local search).

30 diverse articles across AI, science, history, and technology.
Condensed from the original notebook for file-size efficiency.
"""

CORPUS = [
    {"id": 0, "title": "The Transformer Architecture Revolution", "topic": "AI",
     "content": (
        "The transformer architecture was introduced in 2017 by Vaswani et al. "
        "in the paper 'Attention Is All You Need'. Unlike RNNs that process "
        "sequentially, transformers use self-attention for parallel processing. "
        "The scaled dot-product attention computes relevance scores between positions. "
        "This captures long-range dependencies without vanishing gradients. "
        "Transformers became the foundation for BERT, GPT, T5, and were adapted for "
        "vision (ViT), audio, and protein prediction."
     )},
    {"id": 1, "title": "Reinforcement Learning from Human Feedback", "topic": "AI",
     "content": (
        "RLHF aligns language models with human preferences through three stages: "
        "supervised fine-tuning on demonstrations, training a reward model on human "
        "comparisons, and optimizing with PPO. OpenAI used RLHF for InstructGPT and "
        "ChatGPT. The reward model predicts human preferences. Variations include DPO "
        "which eliminates the separate reward model, and Constitutional AI which uses "
        "AI feedback instead of human feedback for alignment."
     )},
    {"id": 2, "title": "Neural Network Scaling Laws", "topic": "AI",
     "content": (
        "Kaplan et al. at OpenAI discovered that language model performance follows "
        "power laws as model size, data size, and compute increase. Simply making "
        "models bigger improves capability predictably. The Chinchilla paper refined "
        "these laws, showing many models were undertrained relative to their size. "
        "Optimal compute allocation trains smaller models on more data. Organizations "
        "must balance model size against data availability and training cost."
     )},
    {"id": 3, "title": "Retrieval-Augmented Generation", "topic": "AI",
     "content": (
        "RAG combines generative LLMs with information retrieval. Instead of relying "
        "solely on parametric knowledge, RAG retrieves relevant documents from an "
        "external knowledge base and includes them in generation context. The original "
        "RAG paper by Lewis et al. used dense passage retrieval from Wikipedia. "
        "Modern RAG uses chunking strategies, hybrid retrieval, and re-ranking. "
        "It is the dominant approach for knowledge-grounded LLM applications."
     )},
    {"id": 4, "title": "AI Agent Architectures", "topic": "AI",
     "content": (
        "AI agents perceive their environment, reason about goals, and take actions. "
        "Modern LLM-based agents combine reasoning with tool use, memory, and planning. "
        "The ReAct framework interleaves reasoning traces with actions. Advanced "
        "architectures like Reflexion add self-evaluation and memory. Multi-agent "
        "systems have specialized agents collaborating, ranging from simple pipelines "
        "to complex hierarchies with manager-worker delegation."
     )},
    {"id": 5, "title": "Prompt Engineering Techniques", "topic": "AI",
     "content": (
        "Prompt engineering designs inputs to elicit desired outputs. Chain-of-thought "
        "prompting by Wei et al. improves reasoning by showing work step-by-step. "
        "Few-shot prompting provides examples of desired patterns. Tree-of-thought "
        "extends CoT by exploring multiple reasoning paths. Effective prompts are "
        "specific, contextual, and define output format. System prompts set overall "
        "behavior while user prompts provide specific tasks."
     )},
    {"id": 6, "title": "Embeddings and Vector Databases", "topic": "AI",
     "content": (
        "Text embeddings are dense vectors capturing semantic meaning. Similar texts "
        "produce similar vectors, enabling nearest-neighbor search. Embedding models "
        "like sentence-transformers use contrastive learning. Popular families include "
        "BGE, E5, and GTE. Vector databases like FAISS, Pinecone, and Weaviate provide "
        "efficient similarity search using approximate nearest neighbor algorithms like "
        "HNSW and IVF for sub-linear search time."
     )},
    {"id": 7, "title": "CRISPR Gene Editing Technology", "topic": "science",
     "content": (
        "CRISPR-Cas9 is a gene editing technology from a bacterial defense mechanism. "
        "Guide RNA directs Cas9 to a specific genome location for precise DNA cuts. "
        "Developed by Doudna and Charpentier, who won the 2020 Nobel Prize in Chemistry. "
        "CRISPR enables adding, removing, or modifying genetic material. Applications "
        "include treating sickle cell anemia and engineering crops. Ethical debates "
        "continue about human germline editing."
     )},
    {"id": 8, "title": "Quantum Computing Fundamentals", "topic": "science",
     "content": (
        "Quantum computers use qubits in superposition, representing 0 and 1 simultaneously. "
        "Combined with entanglement and interference, this enables parallel exploration of "
        "solutions. Current systems are in the NISQ era with dozens to hundreds of error-prone "
        "qubits. Google claimed quantum supremacy in 2019 with its 53-qubit Sycamore processor. "
        "Applications include drug discovery, cryptography via Shor's algorithm, and optimization. "
        "Error correction remains the major challenge."
     )},
    {"id": 9, "title": "mRNA Vaccine Technology", "topic": "science",
     "content": (
        "mRNA vaccines deliver genetic instructions that teach cells to produce a harmless "
        "pathogen piece, triggering immune response. Decades in development before COVID-19 "
        "deployment. Key innovations include modified nucleosides preventing immune destruction, "
        "and lipid nanoparticles protecting fragile mRNA. Speed advantage: once the viral "
        "sequence is known, a candidate can be designed in days. Future applications include "
        "personalized cancer vaccines and rare genetic disease treatments."
     )},
    {"id": 10, "title": "Dark Matter and Dark Energy", "topic": "science",
     "content": (
        "Approximately 85 percent of matter is dark matter, interacting gravitationally but not "
        "emitting light. Dark energy makes up 68 percent of total energy density and was "
        "inferred from the 1998 discovery of accelerating cosmic expansion. Together they "
        "constitute 95 percent of the universe. Candidate dark matter particles include WIMPs "
        "and axions. Dark energy might be Einstein's cosmological constant or a dynamic scalar "
        "field. Their nature remains unknown."
     )},
    {"id": 11, "title": "Climate Change Science", "topic": "science",
     "content": (
        "Human activities primarily burning fossil fuels caused global temperatures to rise "
        "approximately 1.1 degrees Celsius since pre-industrial times. The greenhouse effect "
        "is well-understood physics where CO2 absorbs infrared radiation. Ice cores show current "
        "CO2 levels over 420 ppm are higher than any point in 800,000 years. Impacts include "
        "rising sea levels, extreme weather, ocean acidification, and biodiversity loss. The "
        "IPCC recommends limiting warming to 1.5 degrees Celsius."
     )},
    {"id": 12, "title": "The Human Microbiome", "topic": "science",
     "content": (
        "The human body hosts trillions of microorganisms collectively called the microbiome. "
        "The gut microbiome contains an estimated 100 trillion bacteria across over 1,000 species. "
        "It plays roles in digestion, immune development, mental health via the gut-brain axis, "
        "and disease susceptibility. Dysbiosis is linked to obesity, inflammatory bowel disease, "
        "and depression. The Human Microbiome Project cataloged species across body sites. "
        "Therapies include fecal transplants and targeted probiotics."
     )},
    {"id": 13, "title": "The Industrial Revolution", "topic": "history",
     "content": (
        "Beginning in Britain in the late 18th century, the Industrial Revolution transformed "
        "civilization from agrarian to industrial economies. Key innovations included the steam "
        "engine, spinning jenny, and power loom. The first phase centered on textiles, steam, "
        "and iron. The second brought electricity, steel, and chemicals. Social consequences "
        "included urbanization, factory labor, child labor controversies, and trade unions. It "
        "set the foundation for modern capitalism."
     )},
    {"id": 14, "title": "The Space Race", "topic": "history",
     "content": (
        "The Cold War competition between the US and Soviet Union for spaceflight supremacy. "
        "The Soviets struck first with Sputnik in 1957 and Yuri Gagarin's orbital flight in 1961. "
        "Kennedy's 1961 declaration to land on the Moon galvanized the American program. The Apollo "
        "program consumed 4 percent of the federal budget at peak. On July 20, 1969, Neil Armstrong "
        "and Buzz Aldrin became the first humans to walk on the Moon."
     )},
    {"id": 15, "title": "The Printing Press and Information Revolution", "topic": "history",
     "content": (
        "Gutenberg's movable type printing press around 1440 is among the most important inventions "
        "in history. Before printing, books were hand-copied by scribes, making them extremely "
        "expensive. The press reduced book costs by 80 percent within decades. By 1500, an estimated "
        "20 million volumes had been printed in Europe. This democratization fueled the Renaissance, "
        "Reformation, and Scientific Revolution. Parallels to the modern internet are striking in "
        "reducing information distribution costs."
     )},
    {"id": 16, "title": "Ancient Greek Philosophy and Democracy", "topic": "history",
     "content": (
        "Ancient Athens in the 5th century BCE developed the world's first known democracy where "
        "citizens participated directly in governance. Simultaneously, philosophers laid foundations "
        "for Western thought. Socrates developed the dialectical method. Plato wrote dialogues exploring "
        "justice and the ideal state. Aristotle systematized logic, ethics, and natural philosophy. "
        "Greek contributions to mathematics by Euclid and Pythagoras, medicine by Hippocrates, and "
        "science by Archimedes established methodologies still used today."
     )},
    {"id": 17, "title": "The Silk Road Trade Network", "topic": "history",
     "content": (
        "The Silk Road was a network of trade routes connecting China to the Mediterranean from the "
        "2nd century BCE to the 15th century CE. Despite its name, silk was just one of many traded "
        "goods including spices, metals, and textiles. The routes facilitated exchange of ideas, "
        "religions, and technologies. Buddhism spread from India to China. Paper and gunpowder traveled "
        "from China westward. The Black Death likely spread along these routes in the 14th century."
     )},
    {"id": 18, "title": "The Renaissance", "topic": "history",
     "content": (
        "Meaning 'rebirth,' the Renaissance began in Italy in the 14th century and spread across "
        "Europe. It marked the transition from medieval to modern thinking. Key figures include "
        "Leonardo da Vinci in art and engineering, Michelangelo in sculpture, Galileo in astronomy, "
        "and Machiavelli in political philosophy. The movement emphasized humanism, observation, "
        "and rediscovering classical texts. It transformed art with linear perspective, science with "
        "empiricism, and commerce with banking and double-entry bookkeeping."
     )},
    {"id": 19, "title": "The Internet Protocol Suite", "topic": "technology",
     "content": (
        "TCP/IP is the foundational protocol suite of the internet, developed in the 1970s by Vint "
        "Cerf and Bob Kahn. Its design philosophy of end-to-end connectivity and packet switching created "
        "a resilient, scalable network. The stack has four layers: link, internet, transport, and "
        "application. The transition from IPv4 to IPv6 is ongoing. The internet now connects over "
        "5 billion people and an estimated 15 billion IoT devices worldwide."
     )},
    {"id": 20, "title": "Blockchain and Distributed Ledgers", "topic": "technology",
     "content": (
        "Blockchain technology, introduced by the Bitcoin whitepaper in 2008, provides a decentralized "
        "immutable ledger without requiring a trusted intermediary. Each block contains a cryptographic "
        "hash of the previous block creating a tamper-evident chain. Consensus mechanisms like Proof of "
        "Work and Proof of Stake ensure agreement. Smart contracts popularized by Ethereum allow "
        "programmable logic. Applications include supply chain tracking, digital identity, DeFi, and NFTs. "
        "Energy consumption of Proof of Work remains controversial."
     )},
    {"id": 21, "title": "Cloud Computing Architecture", "topic": "technology",
     "content": (
        "Cloud computing delivers resources like servers, storage, and databases over the internet on "
        "a pay-as-you-go basis. Service models are IaaS, PaaS, and SaaS offering increasing abstraction. "
        "Virtualization and containerization enable efficient resource sharing. Microservices decompose "
        "applications into independently deployable services. Major providers are AWS, Azure, and Google "
        "Cloud. The shift enables rapid scaling, global deployment, and reduced capital expenditure. "
        "Edge computing extends cloud capabilities closer to data sources."
     )},
    {"id": 22, "title": "Cybersecurity Threats and Defenses", "topic": "technology",
     "content": (
        "Cybersecurity protects computer systems, networks, and data from digital attacks. Common threats "
        "include malware, phishing, ransomware, SQL injection, XSS, and DDoS attacks. Defense operates "
        "at multiple layers: network security with firewalls, application security with input validation, "
        "data security with encryption, and organizational security with training. The zero-trust model "
        "assumes no user or system is inherently trustworthy. AI is used both by attackers for automated "
        "phishing and by defenders for anomaly detection."
     )},
    {"id": 23, "title": "Semiconductor Manufacturing", "topic": "technology",
     "content": (
        "Modern chips are manufactured using photolithography where patterns are projected onto silicon "
        "wafers coated with photosensitive material. Current processes use EUV lithography at 3nm and below. "
        "TSMC, Samsung, and Intel are the only companies at advanced nodes. A single EUV machine from ASML "
        "costs over 150 million dollars. Moore's Law drove exponential computing growth for decades. While "
        "physical limits are approaching, innovations like 3D stacking and chiplets continue pushing "
        "performance forward."
     )},
    {"id": 24, "title": "Open Source Software Movement", "topic": "technology",
     "content": (
        "The open source movement formalized by the Open Source Initiative in 1998 promotes freely "
        "available source code for inspection, modification, and distribution. It grew from the earlier "
        "free software movement led by Richard Stallman. Linux created by Linus Torvalds in 1991 powers "
        "most servers, all top supercomputers, and most mobile devices via Android. Major projects include "
        "Apache, PostgreSQL, Python, and Kubernetes. Companies like Red Hat built billion-dollar businesses "
        "around open source through support and services."
     )},
    {"id": 25, "title": "5G and Next-Generation Wireless", "topic": "technology",
     "content": (
        "5G is the fifth generation of cellular technology offering peak speeds up to 20 Gbps and latency "
        "under 1 millisecond. It uses low, mid, and millimeter wave spectrum bands. Applications include "
        "autonomous vehicles, remote surgery, industrial IoT, and augmented reality. Network slicing creates "
        "virtual networks optimized for different use cases. Deployment challenges include denser infrastructure "
        "for mmWave, high costs, and geopolitical concerns around equipment suppliers. Research into 6G is "
        "already underway."
     )},
    {"id": 26, "title": "Electric Vehicles and Battery Technology", "topic": "technology",
     "content": (
        "Electric vehicles have seen rapid adoption driven by falling battery costs and government incentives. "
        "Lithium-ion batteries dropped from over 1000 dollars per kWh in 2010 to under 140 dollars in 2023. "
        "Tesla pioneered the premium EV market. Challenges include charging infrastructure, range anxiety, "
        "and raw material sourcing. Solid-state batteries promise higher energy density and safety. Sodium-ion "
        "offers cheaper alternatives using abundant materials. Vehicle-to-grid technology could turn EVs into "
        "distributed energy storage."
     )},
    {"id": 27, "title": "The Ethics of Artificial Intelligence", "topic": "AI",
     "content": (
        "As AI becomes more capable, ethical concerns moved from academic discussion to urgent policy. Key "
        "issues include bias in training data, lack of transparency, privacy erosion, and job displacement. "
        "Studies show biases in facial recognition, hiring algorithms, and criminal justice risk assessment. "
        "Addressing bias requires diverse data and ongoing monitoring. Frameworks include the EU AI Act, NIST "
        "AI Risk Management Framework, and corporate guidelines. The tension between innovation speed and safety "
        "remains central."
     )},
    {"id": 28, "title": "Nuclear Fusion Research", "topic": "science",
     "content": (
        "Nuclear fusion powers the sun and promises virtually unlimited clean energy. It combines light atomic "
        "nuclei like deuterium and tritium to release enormous energy. Two approaches are magnetic confinement "
        "tokamaks like ITER and inertial confinement like NIF. In December 2022, NIF achieved fusion ignition "
        "producing more energy than the laser input. ITER in France aims to demonstrate sustained fusion power "
        "by the 2030s. Private companies including Commonwealth Fusion Systems pursue alternative designs."
     )},
    {"id": 29, "title": "The History of Computing", "topic": "technology",
     "content": (
        "Computing history spans from Babbage's Analytical Engine concept in the 1830s to modern cloud and "
        "quantum computing. Key milestones include Turing's theoretical foundations, ENIAC in 1945, the transistor "
        "in 1947, and the integrated circuit in 1958. The PC revolution of the 1970s-80s brought computing to "
        "homes. The World Wide Web invented by Tim Berners-Lee in 1989 transformed the internet. Moore's Law "
        "drove exponential improvement. Current frontiers include AI accelerators, quantum processors, and "
        "neuromorphic chips."
     )},
]
