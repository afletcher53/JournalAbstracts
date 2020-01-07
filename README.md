# JournalAbstracts
## Pubmed Crawler

A Pubmed crawler which will gracefully downloaded species specific texts and abstracts, concatenates them, then creates a pickle file with columns of  "abstract, category_txt, category_id"

## inputs
#### Species
(-s [species]) (default="canine,feline") Specify the species for the download. Supported species are canine, feline and equine. You can pass through other species which are experimentally supported.

#### results
(-res [number]) (default = "100000") Specify the amount of results wanted

#### output
(-output) name of the output files

#### embedUSE
(-embedUSE) Encode results with Universal Sentence Embedding

#### embedELMO
(-embedELMO) Encode results with elmo

#### Species Stop words
(-speciesStopWords) (default = True ) Remove the supported species specific stop words from abstract corpus
