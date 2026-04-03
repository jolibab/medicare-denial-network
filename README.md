# Medicare Denial Pattern Network
An interactive Streamlit app that shows which Medicare Fee-for-Service (FFS) CERT providers share similar denial patterns.
Link: https://medicare-denial-network-a3vv4vfstfpvhdpk37rax8.streamlit.app/

## Overview
This app uses Medicare FFS CERT claims data to identify which providers share similar denial patterns. Providers are connected when their error-code profiles are similar enough, making it easier to spot groups of providers who may be making the same types of billing errors.

---
## Features

- **Interactive network graph** zoom, pan, and hover over nodes and edges
- **Edge tooltips** hover over any edge to see the similarity score between two providers (calculated using cosine similarity)
- **Node tooltips** hover over any node to see the provider name, total denial count, and cluster number
- **Community detection** providers are automatically grouped into color-coded clusters based on which providers are most similar to each other (no manual labeling required)
- **Sidebar controls** adjust similarity threshold, max edges per node, label cutoff, and TF-IDF weighting in real time
- **Cluster panel** expandable list of each cluster with top error codes and provider breakdown
- **Error code table** summary of all error codes, total denials, and provider coverage
---
## Methodology

### Normalization
Raw denial counts are normalized in two steps:

1. **Proportional normalization** converts raw counts to a share of each provider's total denials. This removes volume bias so high-volume and low-volume providers are compared equally.
2. **TF-IDF weighting** down-weights error codes that appear across almost all providers since they are not useful for distinguishing patterns, and up-weights rare codes that are more specific to certain providers.

### Similarity
Cosine similarity is computed between every pair of provider profiles. An edge is drawn between two providers if their similarity score meets or exceeds the threshold set in the sidebar.

### Layout
The graph uses the Kamada-Kawai layout, which positions nodes based on edge weights. Providers with stronger similarity are placed closer together, which naturally reveals clusters.

### Deduplication
Duplicate rows are removed before any analysis using exact row matching.

---
## Parameters

| Parameter | Default | Description |
|---|---|---|
| Similarity threshold | 0.75 | Minimum cosine similarity to draw an edge |
| Max edges per node | 5 | Each provider keeps only its strongest connections |
| Label min denials | 50 | Only label providers above this denial count |
| TF-IDF weighting | On | Down-weights common error codes |

---
## Data

**Source:** [CMS Comprehensive Error Rate Testing (CERT) Program](https://www.cms.gov)  
**File:** `Medicare_FFS_CERT_2025.csv`

The CERT program measures improper payments in Medicare Fee-for-Service (FFS) by reviewing a stratified random sample of about 37,500 claims submitted to Medicare Administrative Contractors (MACs).
**Note:** The improper payment rate is not a fraud rate. It measures payments that did not meet Medicare requirements.

---
## Installation

```bash
pip install -r requirements.txt
```

## Running the App

```bash
streamlit run medicare_network_plotly.py
```
---

## Requirements
```
streamlit
networkx
plotly
pandas
scipy
numpy
```
---
## Project Structure
```
medicare-denial-network/
├── medicare_network_plotly.py   # Main Streamlit app
├── Medicare_FFS_CERT_2025.csv   # Source data
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```
---
## Built With
- [Streamlit](https://streamlit.io/) app framework
- [Plotly](https://plotly.com/python/) interactive graph visualization
- [NetworkX](https://networkx.org/) graph construction and layout
- [pandas](https://pandas.pydata.org/) data manipulation
- [SciPy](https://scipy.org/) required by NetworkX for Kamada-Kawai layout
